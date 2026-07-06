import React, { useState, useEffect } from 'react';
import {
  Activity,
  Upload,
  BarChart2,
  Database,
  Info,
  Cpu,
  Clock,
  TrendingUp,
  Image as ImageIcon,
  CheckCircle,
  FileText,
  AlertTriangle,
  Download,
  Award,
  ShieldAlert,
  Layers,
  BookOpen,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { analyzeHeatmap, generateExplanation } from './utils/heatmapAnalysis';

const API_BASE = 'http://localhost:8000/api';

// Turns "smooth_muscle" / "SMOOTH_MUSCLE" into "Smooth Muscle" for
// display inside generated explanation sentences.
const formatClassName = (name) => {
  if (!name) return 'an unknown class';
  return name
    .toString()
    .replace(/_/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
};

function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [metrics, setMetrics] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [loading, setLoading] = useState(false);

  // Prediction Page State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictResult, setPredictResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [error, setError] = useState(null);

  // Explainability State (client-side heatmap analysis + generated text)
  const [heatmapStats, setHeatmapStats] = useState({ cnn: null, vit: null });
  const [explanations, setExplanations] = useState({ cnn: '', vit: '' });
  const [explainLoading, setExplainLoading] = useState(false);

  // Fetch metrics and dataset info on load
  useEffect(() => {
    fetchMetrics();
    fetchDatasetInfo();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (err) {
      console.error("Failed to fetch metrics", err);
    }
  };

  const fetchDatasetInfo = async () => {
    try {
      const res = await fetch(`${API_BASE}/dataset`);
      if (res.ok) {
        const data = await res.json();
        setDatasetInfo(data);
      }
    } catch (err) {
      console.error("Failed to fetch dataset info", err);
    }
  };

  // Prediction upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictResult(null);
      setError(null);
      setHeatmapStats({ cnn: null, vit: null });
      setExplanations({ cnn: '', vit: '' });
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setPredictLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Prediction API call failed.");
      }

      const data = await res.json();
      setPredictResult(data);
      // Refresh metrics in case retraining happened
      fetchMetrics();
    } catch (err) {
      setError(err.message || "Failed to analyze image.");
    } finally {
      setPredictLoading(false);
    }
  };

  // Whenever a new prediction result arrives, analyze the returned
  // Grad-CAM / ViT Grad-CAM overlay images on the client and generate
  // grounded, statistics-based explanations. This does not call any
  // new backend endpoint — it reads the same base64 images already
  // present in predictResult.explainability.
  useEffect(() => {
    if (!predictResult) {
      setHeatmapStats({ cnn: null, vit: null });
      setExplanations({ cnn: '', vit: '' });
      return;
    }

    let cancelled = false;
    setExplainLoading(true);

    (async () => {
      const cnnBase64 = predictResult.explainability?.cnn_gradcam_base64 || null;
      const vitBase64 = predictResult.explainability?.vit_attention_base64 || null;

      const [cnnStats, vitStats] = await Promise.all([
        cnnBase64 ? analyzeHeatmap(cnnBase64) : Promise.resolve(null),
        vitBase64 ? analyzeHeatmap(vitBase64) : Promise.resolve(null),
      ]);

      if (cancelled) return;
      setHeatmapStats({ cnn: cnnStats, vit: vitStats });

      const cnnConf = ((predictResult.predictions?.cnn?.confidence || 0) * 100).toFixed(1);
      const vitConf = ((predictResult.predictions?.vit?.confidence || 0) * 100).toFixed(1);

      setExplanations({
        cnn: generateExplanation({
          modelLabel: 'CNN',
          heatLabel: 'Grad-CAM visualization',
          focusWord: 'activation',
          className: formatClassName(predictResult.predictions?.cnn?.class_name),
          confidencePct: cnnConf,
          stats: cnnStats,
        }),
        vit: generateExplanation({
          modelLabel: 'Vision Transformer',
          heatLabel: 'ViT Grad-CAM visualization',
          focusWord: 'attention',
          className: formatClassName(predictResult.predictions?.vit?.class_name),
          confidencePct: vitConf,
          stats: vitStats,
        }),
      });
      setExplainLoading(false);
    })();

    return () => {
      cancelled = true;
    };
  }, [predictResult]);

  // Helper to get confidence badge styles
  const getConfidenceBadge = (level) => {
    switch (level) {
      case 'High Confidence':
        return 'bg-emerald-950/60 text-emerald-400 border border-emerald-500/30';
      case 'Medium Confidence':
        return 'bg-amber-950/60 text-amber-400 border border-amber-500/30';
      default:
        return 'bg-rose-950/60 text-rose-400 border border-rose-500/30';
    }
  };

  // Prepare chart data
  const getLossChartData = () => {
    if (!metrics || !metrics.cnn || !metrics.vit) return [];
    const cnnHistory = metrics.cnn.history || {};
    const vitHistory = metrics.vit.history || {};

    const cnnLoss = cnnHistory.train_loss || [];
    const vitLoss = vitHistory.train_loss || [];

    const length = Math.max(cnnLoss.length, vitLoss.length);
    const data = [];
    for (let i = 0; i < length; i++) {
      data.push({
        epoch: i + 1,
        'CNN Train Loss': cnnLoss[i] || null,
        'CNN Val Loss': (cnnHistory.val_loss || [])[i] || null,
        'ViT Train Loss': vitLoss[i] || null,
        'ViT Val Loss': (vitHistory.val_loss || [])[i] || null,
      });
    }
    return data;
  };

  const getAccuracyChartData = () => {
    if (!metrics || !metrics.cnn || !metrics.vit) return [];
    const cnnHistory = metrics.cnn.history || {};
    const vitHistory = metrics.vit.history || {};

    const cnnAcc = cnnHistory.train_acc || [];
    const vitAcc = vitHistory.train_acc || [];

    const length = Math.max(cnnAcc.length, vitAcc.length);
    const data = [];
    for (let i = 0; i < length; i++) {
      data.push({
        epoch: i + 1,
        'CNN Train Acc': (cnnAcc[i] || 0) * 100,
        'CNN Val Acc': ((cnnHistory.val_acc || [])[i] || 0) * 100,
        'ViT Train Acc': (vitAcc[i] || 0) * 100,
        'ViT Val Acc': ((vitHistory.val_acc || [])[i] || 0) * 100,
      });
    }
    return data;
  };

  // ---- Model Comparison: winner-per-metric summary -----------------
  // Helper to format values elegantly or return "Not Evaluated"
  const formatMetricValue = (val, isPercentage = false, decimals = 4, suffix = "") => {
    if (val === null || val === undefined || val === "Not Evaluated") {
      return "Not Evaluated";
    }
    const num = Number(val);
    if (isNaN(num)) return "Not Evaluated";
    if (isPercentage) {
      return `${(num * 100).toFixed(2)}%`;
    }
    return `${num.toFixed(decimals)}${suffix}`;
  };

  // Determines, for each shared metric, whether CNN or ViT performs
  // better (lower-is-better for time/size), then rolls up an overall
  // verdict. Purely derived from the existing /api/metrics response.
  const getWinner = (cnnVal, vitVal, lowerIsBetter = false) => {
    if (
      cnnVal == null ||
      vitVal == null ||
      cnnVal === "Not Evaluated" ||
      vitVal === "Not Evaluated"
    ) return null;
    const cnnNum = Number(cnnVal);
    const vitNum = Number(vitVal);
    if (isNaN(cnnNum) || isNaN(vitNum)) return null;
    if (cnnNum === vitNum) return 'tie';
    if (lowerIsBetter) return cnnNum < vitNum ? 'cnn' : 'vit';
    return cnnNum > vitNum ? 'cnn' : 'vit';
  };

  const computeWinnerSummary = () => {
    if (!metrics || !metrics.cnn || !metrics.vit) return null;
    const rows = [
      { key: 'Accuracy', cnn: metrics.cnn.accuracy, vit: metrics.vit.accuracy },
      { key: 'Precision', cnn: metrics.cnn.precision, vit: metrics.vit.precision },
      { key: 'Recall', cnn: metrics.cnn.recall, vit: metrics.vit.recall },
      { key: 'F1 Score', cnn: metrics.cnn.f1_score, vit: metrics.vit.f1_score },
      {
        key: 'Inference Time',
        cnn: metrics.cnn.inference_time_ms_avg,
        vit: metrics.vit.inference_time_ms_avg,
        lowerIsBetter: true,
      },
      {
        key: 'Model Size',
        cnn: metrics.cnn.model_size_mb,
        vit: metrics.vit.model_size_mb,
        lowerIsBetter: true,
      },
    ];
    const scored = rows.map((r) => ({ ...r, winner: getWinner(r.cnn, r.vit, r.lowerIsBetter) }));
    const cnnWins = scored.filter((r) => r.winner === 'cnn').length;
    const vitWins = scored.filter((r) => r.winner === 'vit').length;
    let overall = 'tie';
    if (cnnWins > vitWins) overall = 'cnn';
    else if (vitWins > cnnWins) overall = 'vit';
    return { scored, cnnWins, vitWins, overall };
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans selection:bg-indigo-600 selection:text-white">
      {/* Top Banner */}
      <header className="border-b border-neutral-900 bg-neutral-950/80 backdrop-blur sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-indigo-600/10 text-indigo-400 rounded-lg border border-indigo-500/20">
            <Activity className="h-6 w-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-400 bg-clip-text text-transparent">
              PathExplain AI
            </h1>
            <p className="text-xs text-neutral-500">Colorectal Histology Analysis (PathMNIST)</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="hidden md:flex space-x-1">
          {[
            { id: 'home', label: 'Overview', icon: Cpu },
            { id: 'predict', label: 'Predictions', icon: Upload },
            { id: 'performance', label: 'Model Comparison', icon: BarChart2 },
            { id: 'about', label: 'About', icon: Info },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm transition-all ${
                  active
                    ? 'bg-indigo-600/10 text-indigo-400 border border-indigo-500/20'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900/50 border border-transparent'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {/* Main Container */}
      <main className="flex-grow max-w-7xl w-full mx-auto p-6 flex flex-col">

        {/* TAB 1: OVERVIEW / HOME (rewritten in plain English) */}
        {activeTab === 'home' && (
          <div className="space-y-8 animate-fade-in">
            {/* Hero Section */}
            <div className="relative rounded-2xl overflow-hidden border border-neutral-900 bg-gradient-to-b from-neutral-900/50 to-neutral-950 p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-950/20 via-transparent to-transparent pointer-events-none" />
              <div className="space-y-4 max-w-2xl">
                <span className="text-xs uppercase tracking-wider text-indigo-400 font-semibold px-2.5 py-1 bg-indigo-950/50 border border-indigo-500/20 rounded-full">
                  AI That Shows Its Work
                </span>
                <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight">
                  See what an AI "sees" in a tissue image — not just its answer
                </h2>
                <p className="text-neutral-400 text-sm leading-relaxed">
                  This project takes a small image of tissue and runs it through two different
                  AI models. Each model guesses what type of tissue it is looking at — and, more
                  importantly, this tool highlights exactly which part of the image each model
                  focused on to make that guess. You get the answer <em>and</em> the reasoning
                  behind it.
                </p>
                <div className="flex gap-4 pt-2">
                  <button
                    onClick={() => setActiveTab('predict')}
                    className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition"
                  >
                    Try It Yourself
                  </button>
                  <button
                    onClick={() => setActiveTab('performance')}
                    className="px-5 py-2.5 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 text-neutral-200 rounded-lg text-sm font-semibold transition"
                  >
                    Compare the Two Models
                  </button>
                </div>
                <p className="text-[11px] text-neutral-600 pt-1">
                  Built for education and research — not a medical diagnostic tool. See the About page for details.
                </p>
              </div>
              <div className="w-full md:w-80 flex flex-col space-y-3 bg-neutral-900/40 border border-neutral-800/60 p-5 rounded-xl backdrop-blur">
                <div className="text-xs text-neutral-500 uppercase tracking-wider font-semibold">Models Running</div>
                <div className="flex items-center justify-between py-2 border-b border-neutral-850">
                  <span className="text-sm font-medium">CNN (pattern detector)</span>
                  <span className="text-xs px-2 py-0.5 bg-emerald-950 text-emerald-400 border border-emerald-500/20 rounded">Active</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-sm font-medium">Vision Transformer (big-picture)</span>
                  <span className="text-xs px-2 py-0.5 bg-emerald-950 text-emerald-400 border border-emerald-500/20 rounded">Active</span>
                </div>
                <div className="text-xs text-neutral-500 pt-2 border-t border-neutral-850">
                  Runs on a regular computer processor — no special graphics hardware required.
                </div>
              </div>
            </div>

            {/* Architecture Overview — plain-English explanation of each model */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                <div className="flex items-center space-x-3 text-indigo-400">
                  <Cpu className="h-5 w-5" />
                  <h3 className="font-bold text-base">Model 1: CNN — the "pattern detector"</h3>
                </div>
                <p className="text-neutral-400 text-xs leading-relaxed">
                  This model scans the image in small pieces, looking for local textures, edges,
                  and shapes — a bit like noticing a specific pattern in a photo one section at a
                  time.
                </p>
                <div className="text-xs text-indigo-400 font-semibold">
                  Explains itself with Grad-CAM — a heatmap showing exactly which part of the image it focused on.
                </div>
              </div>

              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                <div className="flex items-center space-x-3 text-purple-400">
                  <TrendingUp className="h-5 w-5" />
                  <h3 className="font-bold text-base">Model 2: Vision Transformer — the "big-picture thinker"</h3>
                </div>
                <p className="text-neutral-400 text-xs leading-relaxed">
                  This model looks at the whole image at once and considers how different regions
                  relate to one another — more like stepping back to see the full picture instead
                  of one detail at a time.
                </p>
                <div className="text-xs text-purple-400 font-semibold">
                  Explains itself with ViT Grad-CAM — a heatmap showing which regions it paid the most attention to.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: PREDICTIONS / EXPLAINABILITY */}
        {activeTab === 'predict' && (
          <div className="space-y-6 animate-fade-in">
            {/* Grid layout */}
            <div className="grid lg:grid-cols-12 gap-6">
              {/* Uploader Card */}
              <div className="lg:col-span-4 border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl flex flex-col justify-between">
                <div className="space-y-4">
                  <h3 className="text-md font-bold">Upload Histology Slide</h3>
                  <p className="text-xs text-neutral-400">Upload tissue patch images (.png, .jpg) from PathMNIST classes to run comparative analysis.</p>

                  {/* File selector box */}
                  <label className="border-2 border-dashed border-neutral-800 hover:border-indigo-500/40 transition-all rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-neutral-950/40 relative">
                    <input type="file" onChange={handleFileChange} className="hidden" accept="image/*" />

                    {previewUrl ? (
                      <img src={previewUrl} alt="Preview" className="h-28 w-28 object-cover rounded-lg border border-neutral-850" />
                    ) : (
                      <>
                        <Upload className="h-8 w-8 text-neutral-500 mb-2" />
                        <span className="text-xs text-neutral-400 font-semibold">Select tissue patch</span>
                        <span className="text-[10px] text-neutral-600 mt-1">PNG, JPG up to 10MB</span>
                      </>
                    )}
                  </label>
                </div>

                <div className="pt-6 space-y-2">
                  {selectedFile && (
                    <button
                      onClick={handlePredict}
                      disabled={predictLoading}
                      className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-neutral-800 disabled:text-neutral-500 text-white rounded-lg text-sm font-semibold transition"
                    >
                      {predictLoading ? "Processing Inference..." : "Analyze Image"}
                    </button>
                  )}
                  {error && (
                    <div className="flex items-center space-x-2 text-rose-400 text-xs bg-rose-950/20 border border-rose-500/10 p-3 rounded-lg">
                      <AlertTriangle className="h-4 w-4" />
                      <span>{error}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Prediction Results Card */}
              <div className="lg:col-span-8 border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl min-h-[300px] flex flex-col justify-center">
                {!predictResult ? (
                  <div className="text-center space-y-2 text-neutral-500">
                    <ImageIcon className="h-10 w-10 mx-auto text-neutral-700" />
                    <p className="text-xs">No analysis running. Please upload and click Analyze.</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Visual Comparison row */}
                    <div className="grid grid-cols-3 gap-4">
                      {/* Original */}
                      <div className="space-y-2">
                        <div className="text-[10px] uppercase font-bold text-neutral-500 text-center">Original Input</div>
                        <div className="aspect-square bg-neutral-950 border border-neutral-900 rounded-lg overflow-hidden flex items-center justify-center">
                          <img src={`data:image/png;base64,${predictResult.explainability.original_image_base64}`} alt="Original" className="w-full h-full object-cover" />
                        </div>
                      </div>

                      {/* CNN Grad-CAM */}
                      <div className="space-y-2">
                        <div className="text-[10px] uppercase font-bold text-indigo-400 text-center">CNN Grad-CAM</div>
                        <div className="aspect-square bg-neutral-950 border border-neutral-900 rounded-lg overflow-hidden flex items-center justify-center relative group">
                          {predictResult.explainability.cnn_gradcam_base64 ? (
                            <>
                              <img src={`data:image/png;base64,${predictResult.explainability.cnn_gradcam_base64}`} alt="CNN Grad-CAM" className="w-full h-full object-cover" />
                              <a
                                href={`data:image/png;base64,${predictResult.explainability.cnn_gradcam_base64}`}
                                download={`gradcam_${predictResult.image_id}.png`}
                                className="absolute bottom-2 right-2 p-1.5 bg-neutral-900/80 hover:bg-indigo-600 text-white rounded transition opacity-0 group-hover:opacity-100"
                              >
                                <Download className="h-3.5 w-3.5" />
                              </a>
                            </>
                          ) : (
                            <span className="text-[10px] text-neutral-600">Grad-CAM unavailable</span>
                          )}
                        </div>
                      </div>

                      {/* ViT Grad-CAM (renamed from "Attention Map") */}
                      <div className="space-y-2">
                        <div className="text-[10px] uppercase font-bold text-purple-400 text-center">ViT Grad-CAM</div>
                        <div className="aspect-square bg-neutral-950 border border-neutral-900 rounded-lg overflow-hidden flex items-center justify-center relative group">
                          {predictResult.explainability.vit_attention_base64 ? (
                            <>
                              <img src={`data:image/png;base64,${predictResult.explainability.vit_attention_base64}`} alt="ViT Grad-CAM" className="w-full h-full object-cover" />
                              <a
                                href={`data:image/png;base64,${predictResult.explainability.vit_attention_base64}`}
                                download={`vit_gradcam_${predictResult.image_id}.png`}
                                className="absolute bottom-2 right-2 p-1.5 bg-neutral-900/80 hover:bg-indigo-600 text-white rounded transition opacity-0 group-hover:opacity-100"
                              >
                                <Download className="h-3.5 w-3.5" />
                              </a>
                            </>
                          ) : (
                            <span className="text-[10px] text-neutral-600">ViT Grad-CAM unavailable</span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Numerical Outputs comparison */}
                    <div className="grid md:grid-cols-2 gap-4">
                      {/* CNN prediction info */}
                      <div className="bg-neutral-950/60 border border-neutral-900 rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-indigo-400">Custom CNN</span>
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${getConfidenceBadge(predictResult.predictions.cnn.confidence_level)}`}>
                            {predictResult.predictions.cnn.confidence_level}
                          </span>
                        </div>
                        <div className="space-y-1">
                          <div className="text-neutral-500 text-[10px] uppercase tracking-wider font-semibold">Prediction Label</div>
                          <div className="text-md font-bold capitalize">{predictResult.predictions.cnn.class_name}</div>
                        </div>
                        <div className="flex justify-between text-xs border-t border-neutral-900 pt-2 text-neutral-400">
                          <span className="flex items-center"><Clock className="h-3 w-3 mr-1" /> {predictResult.predictions.cnn.inference_time_ms.toFixed(1)} ms</span>
                          <span>Confidence: {(predictResult.predictions.cnn.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>

                      {/* ViT prediction info */}
                      <div className="bg-neutral-950/60 border border-neutral-900 rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-purple-400">Vision Transformer</span>
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${getConfidenceBadge(predictResult.predictions.vit.confidence_level)}`}>
                            {predictResult.predictions.vit.confidence_level}
                          </span>
                        </div>
                        <div className="space-y-1">
                          <div className="text-neutral-500 text-[10px] uppercase tracking-wider font-semibold">Prediction Label</div>
                          <div className="text-md font-bold capitalize">{predictResult.predictions.vit.class_name}</div>
                        </div>
                        <div className="flex justify-between text-xs border-t border-neutral-900 pt-2 text-neutral-400">
                          <span className="flex items-center"><Clock className="h-3 w-3 mr-1" /> {predictResult.predictions.vit.inference_time_ms.toFixed(1)} ms</span>
                          <span>Confidence: {(predictResult.predictions.vit.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>

                    {/* Dynamic, statistics-grounded explanations */}
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="bg-neutral-950/60 border border-neutral-900 rounded-lg p-4 space-y-2">
                        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-bold uppercase tracking-wider">
                          <FileText className="h-3.5 w-3.5" />
                          <span>Why the CNN focused here</span>
                        </div>
                        <p className="text-xs text-neutral-400 leading-relaxed">
                          {explainLoading && !explanations.cnn
                            ? 'Analyzing Grad-CAM visualization…'
                            : explanations.cnn}
                        </p>
                      </div>
                      <div className="bg-neutral-950/60 border border-neutral-900 rounded-lg p-4 space-y-2">
                        <div className="flex items-center space-x-2 text-purple-400 text-xs font-bold uppercase tracking-wider">
                          <FileText className="h-3.5 w-3.5" />
                          <span>Why the ViT focused here</span>
                        </div>
                        <p className="text-xs text-neutral-400 leading-relaxed">
                          {explainLoading && !explanations.vit
                            ? 'Analyzing ViT Grad-CAM visualization…'
                            : explanations.vit}
                        </p>
                      </div>
                    </div>
                    <p className="text-[10px] text-neutral-600">
                      These explanations describe the visualization above (where the color is concentrated, how spread out it is, how many hotspots it has) — they are not medical findings.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: PERFORMANCE / MODEL COMPARISON */}
        {activeTab === 'performance' && (
          <div className="space-y-8 animate-fade-in">

            {/* Winner Summary */}
            {(() => {
              const summary = computeWinnerSummary();
              if (!summary) {
                return (
                  <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl text-center text-neutral-500 text-sm">
                    Loading comparison metrics...
                  </div>
                );
              }
              return (
                <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                  <h3 className="text-base font-bold flex items-center space-x-2">
                    <Award className="h-5 w-5 text-amber-400" />
                    <span>Winner Summary</span>
                  </h3>
                  <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
                    {summary.scored.map((r) => (
                      <div
                        key={r.key}
                        className="bg-neutral-950/40 border border-neutral-900 rounded-lg p-3 flex items-center justify-between text-xs"
                      >
                        <span className="text-neutral-400">{r.key}</span>
                        <span
                          className={`font-semibold px-2 py-0.5 rounded-full ${
                            r.winner === 'cnn'
                              ? 'text-indigo-400 bg-indigo-950/50 border border-indigo-500/20'
                              : r.winner === 'vit'
                              ? 'text-purple-400 bg-purple-950/50 border border-purple-500/20'
                              : 'text-neutral-500 bg-neutral-900 border border-neutral-800'
                          }`}
                        >
                          {r.winner === 'tie' ? 'Tie' : r.winner === 'cnn' ? 'CNN' : 'ViT'}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm text-neutral-400 leading-relaxed pt-3 border-t border-neutral-850">
                    {summary.overall === 'tie'
                      ? `Both models win on ${summary.cnnWins} metric(s) each — there's no single overall winner. The right pick depends on whether accuracy or speed/size matters more for your use case.`
                      : `The ${summary.overall === 'cnn' ? 'Custom CNN' : 'Vision Transformer'} leads on ${
                          summary.overall === 'cnn' ? summary.cnnWins : summary.vitWins
                        } out of ${summary.scored.length} metrics, making it the stronger overall choice between the two — though the other model may still be preferable depending on which single metric matters most for your deployment (e.g. raw accuracy vs. speed and size).`}
                  </p>
                </div>
              );
            })()}

            {/* Comparison Table */}
            <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
              <h3 className="text-base font-bold flex items-center space-x-2">
                <BarChart2 className="h-5 w-5 text-indigo-400" />
                <span>Model Comparison Dashboard</span>
              </h3>

              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm text-neutral-300">
                  <thead className="text-xs text-neutral-500 uppercase border-b border-neutral-850 bg-neutral-950/40">
                    <tr>
                      <th className="py-3 px-4">Model Name</th>
                      <th className="py-3 px-4 text-right">Accuracy</th>
                      <th className="py-3 px-4 text-right">Precision</th>
                      <th className="py-3 px-4 text-right">Recall</th>
                      <th className="py-3 px-4 text-right">F1 Score</th>
                      <th className="py-3 px-4 text-right">ROC-AUC</th>
                      <th className="py-3 px-4 text-right">Inference Time</th>
                      <th className="py-3 px-4 text-right">Parameters</th>
                      <th className="py-3 px-4 text-right">Model Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics ? (
                      <>
                        <tr className="border-b border-neutral-850 hover:bg-neutral-900/30">
                          <td className="py-3 px-4 font-semibold text-indigo-400">{metrics.cnn?.model_name || "Custom CNN"}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.accuracy, true)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.precision)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.recall)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.f1_score)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.roc_auc)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.inference_time_ms_avg, false, 2, " ms")}</td>
                          <td className="py-3 px-4 text-right">{metrics.cnn?.num_parameters != null && metrics.cnn?.num_parameters !== "Not Evaluated" ? metrics.cnn.num_parameters.toLocaleString() : "Not Evaluated"}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.cnn?.model_size_mb, false, 2, " MB")}</td>
                        </tr>
                        <tr className="hover:bg-neutral-900/30">
                          <td className="py-3 px-4 font-semibold text-purple-400">{metrics.vit?.model_name || "Vision Transformer"}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.accuracy, true)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.precision)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.recall)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.f1_score)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.roc_auc)}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.inference_time_ms_avg, false, 2, " ms")}</td>
                          <td className="py-3 px-4 text-right">{metrics.vit?.num_parameters != null && metrics.vit?.num_parameters !== "Not Evaluated" ? metrics.vit.num_parameters.toLocaleString() : "Not Evaluated"}</td>
                          <td className="py-3 px-4 text-right">{formatMetricValue(metrics.vit?.model_size_mb, false, 2, " MB")}</td>
                        </tr>
                      </>
                    ) : (
                      <tr>
                        <td colSpan="9" className="text-center py-4 text-neutral-500">Loading comparison metrics...</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Interactive Charts */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Chart 1: Training curves */}
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                <h4 className="text-sm font-bold text-neutral-300">Loss Curves</h4>
                <div className="h-64">
                  {metrics ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={getLossChartData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
                        <XAxis dataKey="epoch" stroke="#525252" fontSize={11} />
                        <YAxis stroke="#525252" fontSize={11} />
                        <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#1f1f1f' }} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line type="monotone" dataKey="CNN Val Loss" stroke="#6366f1" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="ViT Val Loss" stroke="#a855f7" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-neutral-500">Loading curve charts...</div>
                  )}
                </div>
              </div>

              {/* Chart 2: Accuracy curves */}
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                <h4 className="text-sm font-bold text-neutral-300">Validation Accuracy (%)</h4>
                <div className="h-64">
                  {metrics ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={getAccuracyChartData()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
                        <XAxis dataKey="epoch" stroke="#525252" fontSize={11} />
                        <YAxis stroke="#525252" fontSize={11} domain={[0, 100]} />
                        <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', borderColor: '#1f1f1f' }} />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Line type="monotone" dataKey="CNN Val Acc" stroke="#6366f1" strokeWidth={2} name="CNN Val Acc" dot={false} />
                        <Line type="monotone" dataKey="ViT Val Acc" stroke="#a855f7" strokeWidth={2} name="ViT Val Acc" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-neutral-500">Loading curve charts...</div>
                  )}
                </div>
              </div>
            </div>

          </div>
        )}

        {/* TAB 4: ABOUT (rewritten completely) */}
        {activeTab === 'about' && (
          <div className="space-y-6 animate-fade-in">

            {/* Disclaimer banner */}
            <div className="border border-amber-500/20 bg-amber-950/20 p-5 rounded-xl flex items-start space-x-3">
              <ShieldAlert className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-amber-200/90 leading-relaxed">
                <strong>This platform is for education and research only.</strong> It is not a
                clinical diagnostic system, has not been validated for medical use, and must never
                be used to diagnose or treat real patients. Always consult a qualified pathologist
                or physician for any medical decision.
              </p>
            </div>

            {/* What is Histopathology */}
            <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
              <h3 className="text-lg font-bold flex items-center space-x-2">
                <BookOpen className="h-5 w-5 text-indigo-400" />
                <span>What is Histopathology?</span>
              </h3>
              <p className="text-sm text-neutral-400 leading-relaxed max-w-4xl">
                Histopathology is the study of tissue under a microscope to understand disease.
                A small sample of tissue is stained and examined so that its cell structure and
                organization can be assessed. It's one of the most common ways diseases,
                including cancer, are studied and understood.
              </p>
            </div>

            {/* What is PathMNIST */}
            <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
              <h3 className="text-lg font-bold flex items-center space-x-2">
                <Database className="h-5 w-5 text-indigo-400" />
                <span>What is PathMNIST?</span>
              </h3>
              <p className="text-sm text-neutral-400 leading-relaxed max-w-4xl">
                PathMNIST is a publicly available benchmark dataset of small colorectal tissue
                image patches, built from a larger histology image collection and distributed as
                part of the MedMNIST collection for evaluating machine learning models on medical
                images.{' '}
                {datasetInfo?.num_classes ? (
                  <>
                    This project's copy of the dataset has <strong>{datasetInfo.num_classes}</strong> tissue
                    categories{datasetInfo?.classes?.length ? (
                      <> — {datasetInfo.classes.slice(0, 3).join(', ')}{datasetInfo.classes.length > 3 ? ', and others' : ''}</>
                    ) : null}.
                  </>
                ) : (
                  <>It typically includes several tissue categories such as normal mucosa, tumor
                  epithelium, stroma, muscle, and lymphocytes.</>
                )}
              </p>
            </div>

            {/* CNN vs ViT */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
                <h3 className="text-base font-bold flex items-center space-x-2 text-indigo-400">
                  <Cpu className="h-5 w-5" />
                  <span>What is a CNN?</span>
                </h3>
                <p className="text-sm text-neutral-400 leading-relaxed">
                  A Convolutional Neural Network (CNN) learns to recognize small local patterns —
                  edges, textures, shapes — by sliding filters across an image. Stacking several
                  layers lets it build up from simple patterns to more complex ones. CNNs are a
                  long-standing, efficient choice for image classification.
                </p>
              </div>
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
                <h3 className="text-base font-bold flex items-center space-x-2 text-purple-400">
                  <TrendingUp className="h-5 w-5" />
                  <span>What is a Vision Transformer?</span>
                </h3>
                <p className="text-sm text-neutral-400 leading-relaxed">
                  A Vision Transformer (ViT) splits an image into small patches and treats them
                  like a sequence of tokens, using a self-attention mechanism to learn how every
                  patch relates to every other patch. This lets it reason about relationships
                  across the whole image rather than one local region at a time.
                </p>
              </div>
            </div>

            {/* Explainable AI */}
            <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
              <h3 className="text-lg font-bold flex items-center space-x-2">
                <FileText className="h-5 w-5 text-indigo-400" />
                <span>What is Explainable AI (XAI), and why does it matter?</span>
              </h3>
              <p className="text-sm text-neutral-400 leading-relaxed max-w-4xl">
                Most deep learning models are "black boxes" — they give an answer without showing
                their reasoning. Explainable AI techniques try to open that box up a little. This
                project uses two such techniques:
              </p>
              <ul className="list-disc pl-5 space-y-2 text-sm text-neutral-400 max-w-4xl">
                <li>
                  <strong className="text-neutral-300">Grad-CAM</strong> (for the CNN) highlights
                  which regions of the image most influenced the model's convolutional layers when
                  making its prediction.
                </li>
                <li>
                  <strong className="text-neutral-300">ViT Grad-CAM</strong> (for the Vision
                  Transformer) highlights which regions received the most attention from the
                  model's self-attention layers.
                </li>
              </ul>
              <p className="text-sm text-neutral-400 leading-relaxed max-w-4xl pt-2">
                Explainability matters because a confident-sounding prediction isn't automatically
                a trustworthy one. Being able to see <em>where</em> a model focused makes it far
                easier to catch mistakes, spot models that are focusing on the wrong part of an
                image, and build justified trust before any AI system is ever considered for a
                sensitive domain like medicine.
              </p>
            </div>

            {/* Tech Stack */}
            <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
              <h3 className="text-lg font-bold flex items-center space-x-2">
                <Layers className="h-5 w-5 text-indigo-400" />
                <span>Tech Stack</span>
              </h3>
              <div className="grid md:grid-cols-2 gap-4 text-sm text-neutral-400">
                <ul className="list-disc pl-5 space-y-1.5">
                  <li><strong className="text-neutral-300">Backend:</strong> FastAPI (Python)</li>
                  <li><strong className="text-neutral-300">Models:</strong> Custom CNN and a Vision Transformer</li>
                  <li><strong className="text-neutral-300">Explainability:</strong> Grad-CAM and ViT Grad-CAM (attention-based)</li>
                </ul>
                <ul className="list-disc pl-5 space-y-1.5">
                  <li><strong className="text-neutral-300">Frontend:</strong> React + Vite</li>
                  <li><strong className="text-neutral-300">Styling:</strong> Tailwind CSS</li>
                  <li><strong className="text-neutral-300">Charts:</strong> Recharts</li>
                </ul>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-neutral-900 bg-neutral-950/40 px-6 py-4 text-center text-xs text-neutral-600">
        &copy; {new Date().getFullYear()} PathExplain AI. All rights reserved.
      </footer>
    </div>
  );
}

export default App;
