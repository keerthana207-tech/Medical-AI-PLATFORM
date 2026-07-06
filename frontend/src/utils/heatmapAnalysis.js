// src/utils/heatmapAnalysis.js
//
// Lightweight, client-side heatmap analysis for Grad-CAM (CNN) and
// ViT Grad-CAM (Vision Transformer attention) visualizations.
//
// IMPORTANT: This module never inspects tissue content or makes medical
// claims. It only reads the pixel colors of the already-rendered
// heatmap overlay image (base64 PNG returned by the backend) and derives
// simple, describable statistics about WHERE and HOW the colored
// overlay is distributed across the image. Those statistics are then
// turned into plain-English sentences describing the visualization
// itself (e.g. "localized", "diffuse", "central", "multiple hotspots").
//
// No backend changes are required for this — it works entirely on the
// image data that the existing /api/predict response already provides.

// ---- Tunable constants -----------------------------------------------

const GRID_SIZE = 48;          // downsample size used for analysis (fast + stable)
const ACTIVATION_THRESHOLD = 0.45; // fraction of max "warmth" counted as activated
const ENTROPY_BINS = 10;
const MIN_SIGNAL = 5;          // min (R-B) range (0-255) required to trust the heatmap

// ---- Image loading helpers --------------------------------------------

function toDataUrl(base64) {
  if (!base64) return null;
  return base64.startsWith('data:') ? base64 : `data:image/png;base64,${base64}`;
}

function loadImage(base64) {
  return new Promise((resolve, reject) => {
    const url = toDataUrl(base64);
    if (!url) {
      reject(new Error('No image data provided'));
      return;
    }
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load image for analysis'));
    img.src = url;
  });
}

function getPixelGrid(img, size = GRID_SIZE) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size).data; // Uint8ClampedArray RGBA
}

// ---- Core statistics ----------------------------------------------------

/**
 * Analyze a base64 heatmap overlay image and return grounded statistics
 * about the spatial distribution of "warm" (high-activation) coloring.
 *
 * Returns { valid: false } if there is no usable image or not enough
 * color signal to say anything meaningful (e.g. a flat/empty overlay).
 */
export async function analyzeHeatmap(base64) {
  if (!base64) return null; // no image at all — caller should treat as "unavailable"

  try {
    const img = await loadImage(base64);
    const size = GRID_SIZE;
    const data = getPixelGrid(img, size);
    const n = size * size;

    // "Warmth" proxy: standard heatmap colormaps (jet/inferno/turbo/etc.)
    // render high activation as red/yellow (high R, low B) and low
    // activation as blue/green (low R, high B). R - B is a simple,
    // colormap-agnostic proxy for activation strength.
    const activation = new Float32Array(n);
    let maxRaw = 0;
    for (let i = 0; i < n; i++) {
      const r = data[i * 4];
      const b = data[i * 4 + 2];
      let raw = r - b;
      if (raw < 0) raw = 0;
      activation[i] = raw;
      if (raw > maxRaw) maxRaw = raw;
    }

    if (maxRaw < MIN_SIGNAL) {
      return { valid: false };
    }

    for (let i = 0; i < n; i++) activation[i] = activation[i] / maxRaw;

    // Weighted center of mass + coverage
    let activatedCount = 0;
    let sumX = 0;
    let sumY = 0;
    let sumW = 0;
    const mask = new Uint8Array(n);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = y * size + x;
        const v = activation[idx];
        sumX += x * v;
        sumY += y * v;
        sumW += v;
        if (v >= ACTIVATION_THRESHOLD) {
          mask[idx] = 1;
          activatedCount++;
        }
      }
    }

    const activatedPct = (activatedCount / n) * 100;
    const cx = sumW > 0 ? sumX / sumW / (size - 1) : 0.5;
    const cy = sumW > 0 ? sumY / sumW / (size - 1) : 0.5;

    // Spread: weighted standard deviation of positions around the
    // center of mass. Small = tightly localized, large = diffuse.
    let varX = 0;
    let varY = 0;
    if (sumW > 0) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = y * size + x;
          const v = activation[idx];
          const dx = x / (size - 1) - cx;
          const dy = y / (size - 1) - cy;
          varX += v * dx * dx;
          varY += v * dy * dy;
        }
      }
      varX /= sumW;
      varY /= sumW;
    }
    const spread = Math.sqrt(varX + varY);

    // Distance of the center of mass from the image's geometric center,
    // normalized so ~0 = center, ~1 = near a corner/edge.
    const distanceFromCenter =
      Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / Math.SQRT1_2;

    // Entropy of the activation histogram: low = a sharp, well-defined
    // peak; high = intensity spread evenly across the affected area.
    const hist = new Array(ENTROPY_BINS).fill(0);
    let histTotal = 0;
    for (let i = 0; i < n; i++) {
      const v = activation[i];
      if (v <= 0) continue;
      const b = Math.min(ENTROPY_BINS - 1, Math.floor(v * ENTROPY_BINS));
      hist[b]++;
      histTotal++;
    }
    let entropy = 0;
    if (histTotal > 0) {
      for (const h of hist) {
        if (h === 0) continue;
        const p = h / histTotal;
        entropy -= p * Math.log2(p);
      }
    }
    const entropyNorm = histTotal > 0 ? entropy / Math.log2(ENTROPY_BINS) : 0;

    // Hotspot count: connected components in the thresholded mask
    // (4-connectivity flood fill), ignoring specks below a minimum size.
    const visited = new Uint8Array(n);
    let components = 0;
    const minComponentSize = Math.max(2, Math.round(n * 0.004));
    const stack = [];
    for (let start = 0; start < n; start++) {
      if (!mask[start] || visited[start]) continue;
      let compSize = 0;
      stack.push(start);
      visited[start] = 1;
      while (stack.length) {
        const cur = stack.pop();
        compSize++;
        const cx2 = cur % size;
        const cy2 = Math.floor(cur / size);
        const neighbors = [
          [cx2 - 1, cy2],
          [cx2 + 1, cy2],
          [cx2, cy2 - 1],
          [cx2, cy2 + 1],
        ];
        for (const [nx, ny] of neighbors) {
          if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
            const nIdx = ny * size + nx;
            if (mask[nIdx] && !visited[nIdx]) {
              visited[nIdx] = 1;
              stack.push(nIdx);
            }
          }
        }
      }
      if (compSize >= minComponentSize) components++;
    }
    const hotspotCount = Math.max(components, activatedCount > 0 ? 1 : 0);

    return {
      valid: true,
      activatedPct,
      centerX: cx,
      centerY: cy,
      distanceFromCenter,
      spread,
      entropyNorm,
      hotspotCount,
    };
  } catch (err) {
    console.error('Heatmap analysis failed:', err);
    return { valid: false };
  }
}

// ---- Qualitative descriptors --------------------------------------------

function describeDistribution(spread) {
  if (spread < 0.14) return { label: 'a tightly localized', diffuse: false };
  if (spread < 0.24) return { label: 'a moderately concentrated', diffuse: false };
  if (spread < 0.34) return { label: 'a moderately spread-out', diffuse: true };
  return { label: 'a broadly diffuse', diffuse: true };
}

function describePosition(distanceFromCenter) {
  if (distanceFromCenter < 0.22) return 'a central region of the image';
  if (distanceFromCenter < 0.5) return 'an off-center region of the image';
  return 'a region near the edge of the image';
}

function describeCoverage(pct) {
  if (pct < 12) return 'a small';
  if (pct < 35) return 'a moderate';
  return 'a large';
}

function describeIntensityUniformity(entropyNorm) {
  if (entropyNorm > 0.82) {
    return 'with intensity spread fairly evenly across the highlighted area';
  }
  if (entropyNorm > 0.55) {
    return 'with a moderate gradient between strong and weak regions';
  }
  return 'with a sharp, well-defined peak of intensity';
}

function describeHotspots(count) {
  if (count <= 1) return 'A single dominant region of interest was detected';
  if (count === 2) return 'Two separate regions of interest were detected';
  return `${count} separate regions of interest were detected`;
}

/**
 * Turn computed heatmap statistics into a plain-English explanation.
 * Every sentence is derived directly from the numeric stats — nothing
 * about tissue type, pathology, or diagnosis is asserted here.
 *
 * @param {Object} params
 * @param {string} params.modelLabel   e.g. "CNN" or "Vision Transformer"
 * @param {string} params.heatLabel    e.g. "Grad-CAM visualization" or "ViT Grad-CAM visualization"
 * @param {string} params.focusWord    e.g. "activation" or "attention"
 * @param {string} params.className    predicted class, human-readable
 * @param {string|number} params.confidencePct  confidence as a percentage string/number
 * @param {Object|null} params.stats   output of analyzeHeatmap(), or null if no image
 */
export function generateExplanation({
  modelLabel,
  heatLabel,
  focusWord,
  className,
  confidencePct,
  stats,
}) {
  const opening = `The ${modelLabel} predicted ${className} with ${confidencePct}% confidence.`;

  if (!stats) {
    return `${opening} No ${heatLabel.toLowerCase()} was returned for this prediction, so a visual explanation is not available.`;
  }

  if (!stats.valid) {
    return `${opening} The ${heatLabel} did not contain a strong enough ${focusWord} signal to describe a clear spatial pattern.`;
  }

  const dist = describeDistribution(stats.spread);
  const position = describePosition(stats.distanceFromCenter);
  const coverage = describeCoverage(stats.activatedPct);
  const uniformity = describeIntensityUniformity(stats.entropyNorm);
  const hotspotPhrase = describeHotspots(stats.hotspotCount);
  const pctStr = stats.activatedPct.toFixed(1);

  const spatialSentence = !dist.diffuse
    ? `The ${heatLabel} shows ${dist.label} area of ${focusWord} covering ${coverage} portion of the image (about ${pctStr}% of the area), concentrated around ${position}.`
    : `The ${heatLabel} shows ${dist.label} pattern of ${focusWord} spread across ${coverage} portion of the image (about ${pctStr}% of the area), centered loosely around ${position}.`;

  const hotspotSentence = `${hotspotPhrase}, ${uniformity}.`;

  const conclusion = !dist.diffuse
    ? 'This indicates the prediction relied primarily on a specific, well-defined region of the image rather than the entire tissue sample.'
    : 'This indicates the prediction was influenced by multiple or widely distributed regions of the image rather than one localized structure.';

  return `${opening} ${spatialSentence} ${hotspotSentence} ${conclusion}`;
}
