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
  Download
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

const API_BASE = 'http://localhost:8000/api';

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
              Medical AI Platform
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
            { id: 'dataset', label: 'PathMNIST Dataset', icon: Database },
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
        
        {/* TAB 1: OVERVIEW */}
        {activeTab === 'home' && (
          <div className="space-y-8 animate-fade-in">
            {/* Hero Section */}
            <div className="relative rounded-2xl overflow-hidden border border-neutral-900 bg-gradient-to-b from-neutral-900/50 to-neutral-950 p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-950/20 via-transparent to-transparent pointer-events-none" />
              <div className="space-y-4 max-w-2xl">
                <span className="text-xs uppercase tracking-wider text-indigo-400 font-semibold px-2.5 py-1 bg-indigo-950/50 border border-indigo-500/20 rounded-full">
                  Explainable AI (XAI)
                </span>
                <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight">
                  Comparing Convolutional Networks and Vision Transformers
                </h2>
                <p className="text-neutral-400 text-sm leading-relaxed">
                  Histology tissue classification requires precision, robustness, and trust. Our platform runs dual inference on a Custom CNN and a pretrained timm Vision Transformer (ViT) to compare their structural differences, visual focuses, and accuracy metrics.
                </p>
                <div className="flex gap-4 pt-2">
                  <button 
                    onClick={() => setActiveTab('predict')}
                    className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-semibold transition"
                  >
                    Start Analysis
                  </button>
                  <button 
                    onClick={() => setActiveTab('performance')}
                    className="px-5 py-2.5 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 text-neutral-200 rounded-lg text-sm font-semibold transition"
                  >
                    View Metrics
                  </button>
                </div>
              </div>
              <div className="w-full md:w-80 flex flex-col space-y-3 bg-neutral-900/40 border border-neutral-800/60 p-5 rounded-xl backdrop-blur">
                <div className="text-xs text-neutral-500 uppercase tracking-wider font-semibold">Active Models</div>
                <div className="flex items-center justify-between py-2 border-b border-neutral-850">
                  <span className="text-sm font-medium">Custom CNN</span>
                  <span className="text-xs px-2 py-0.5 bg-emerald-950 text-emerald-400 border border-emerald-500/20 rounded">Active</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-sm font-medium">timm ViT Tiny</span>
                  <span className="text-xs px-2 py-0.5 bg-emerald-950 text-emerald-400 border border-emerald-500/20 rounded">Active</span>
                </div>
                <div className="text-xs text-neutral-500 pt-2 border-t border-neutral-850">
                  AMD GPU unsupported; executing on CPU (linear-probed ViT).
                </div>
              </div>
            </div>

            {/* Architecture Overview */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                <div className="flex items-center space-x-3 text-indigo-400">
                  <Cpu className="h-5 w-5" />
                  <h3 className="font-bold text-base">Model 1: Custom CNN</h3>
                </div>
                <p className="text-neutral-400 text-xs leading-relaxed">
                  Our custom Convolutional Neural Network uses spatial convolutions to capture regional textures in tissue patches. Equipped with three Conv2D blocks, Batch Normalization, and Max Pooling, it excels at local pattern extraction.
                </p>
                <div className="text-xs text-indigo-400 font-semibold">Grad-CAM explainability hooks into the last convolutional layers.</div>
              </div>

              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-4">
                <div className="flex items-center space-x-3 text-purple-400">
                  <TrendingUp className="h-5 w-5" />
                  <h3 className="font-bold text-base">Model 2: Vision Transformer</h3>
                </div>
                <p className="text-neutral-400 text-xs leading-relaxed">
                  Based on the `vit_tiny_patch16_224` backbone from HuggingFace (`timm`), it slices input images into patches and maps them as sequence tokens. Self-attention layers capture global associations across the slide.
                </p>
                <div className="text-xs text-purple-400 font-semibold">Self-attention hook extracts the raw Multi-Head CLS attention matrix.</div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: PREDICTIONS */}
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
                              <img src={`data:image/png;base64,${predictResult.explainability.cnn_gradcam_base64}`} alt="Grad-CAM" className="w-full h-full object-cover" />
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

                      {/* ViT Attention */}
                      <div className="space-y-2">
                        <div className="text-[10px] uppercase font-bold text-purple-400 text-center">ViT Attention Map</div>
                        <div className="aspect-square bg-neutral-950 border border-neutral-900 rounded-lg overflow-hidden flex items-center justify-center relative group">
                          {predictResult.explainability.vit_attention_base64 ? (
                            <>
                              <img src={`data:image/png;base64,${predictResult.explainability.vit_attention_base64}`} alt="Attention Map" className="w-full h-full object-cover" />
                              <a
                                href={`data:image/png;base64,${predictResult.explainability.vit_attention_base64}`}
                                download={`attention_${predictResult.image_id}.png`}
                                className="absolute bottom-2 right-2 p-1.5 bg-neutral-900/80 hover:bg-indigo-600 text-white rounded transition opacity-0 group-hover:opacity-100"
                              >
                                <Download className="h-3.5 w-3.5" />
                              </a>
                            </>
                          ) : (
                            <span className="text-[10px] text-neutral-600">Attention Map unavailable</span>
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
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: PERFORMANCE COMPARISON */}
        {activeTab === 'performance' && (
          <div className="space-y-8 animate-fade-in">
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
                          <td className="py-3 px-4 text-right">{((metrics.cnn?.accuracy || 0) * 100).toFixed(2)}%</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.precision || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.recall || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.f1_score || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.roc_auc || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.inference_time_ms_avg || 0).toFixed(2)} ms</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.num_parameters || 0).toLocaleString()}</td>
                          <td className="py-3 px-4 text-right">{(metrics.cnn?.model_size_mb || 0).toFixed(2)} MB</td>
                        </tr>
                        <tr className="hover:bg-neutral-900/30">
                          <td className="py-3 px-4 font-semibold text-purple-400">{metrics.vit?.model_name || "Vision Transformer"}</td>
                          <td className="py-3 px-4 text-right">{((metrics.vit?.accuracy || 0) * 100).toFixed(2)}%</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.precision || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.recall || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.f1_score || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.roc_auc || 0).toFixed(4)}</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.inference_time_ms_avg || 0).toFixed(2)} ms</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.num_parameters || 0).toLocaleString()}</td>
                          <td className="py-3 px-4 text-right">{(metrics.vit?.model_size_mb || 0).toFixed(2)} MB</td>
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
            
            {/* Confusion Matrices */}
            {metrics && (
              <div className="grid md:grid-cols-2 gap-6">
                {/* CNN Matrix */}
                <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
                  <h4 className="text-sm font-bold text-indigo-400">CNN Confusion Matrix (Colorectal Tissue Classifications)</h4>
                  <div className="grid grid-cols-9 gap-1 bg-neutral-950 p-2 rounded-lg aspect-square">
                    {metrics.cnn?.confusion_matrix?.map((row, rIdx) => 
                      row.map((val, cIdx) => {
                        const isDiagonal = rIdx === cIdx;
                        const opacity = val > 0 ? Math.min(val / 100 + 0.15, 0.95) : 0.05;
                        return (
                          <div 
                            key={`cnn-cm-${rIdx}-${cIdx}`}
                            style={{ backgroundColor: isDiagonal ? `rgba(99, 102, 241, ${opacity})` : `rgba(244, 63, 94, ${opacity})` }}
                            className="aspect-square flex items-center justify-center text-[8px] font-bold border border-neutral-900/30 rounded"
                            title={`True: ${settings.CLASS_NAMES[rIdx]}, Pred: ${settings.CLASS_NAMES[cIdx]} = ${val}`}
                          >
                            {val}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

                {/* ViT Matrix */}
                <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-3">
                  <h4 className="text-sm font-bold text-purple-400">ViT Confusion Matrix (Colorectal Tissue Classifications)</h4>
                  <div className="grid grid-cols-9 gap-1 bg-neutral-950 p-2 rounded-lg aspect-square">
                    {metrics.vit?.confusion_matrix?.map((row, rIdx) => 
                      row.map((val, cIdx) => {
                        const isDiagonal = rIdx === cIdx;
                        const opacity = val > 0 ? Math.min(val / 100 + 0.15, 0.95) : 0.05;
                        return (
                          <div 
                            key={`vit-cm-${rIdx}-${cIdx}`}
                            style={{ backgroundColor: isDiagonal ? `rgba(168, 85, 247, ${opacity})` : `rgba(244, 63, 94, ${opacity})` }}
                            className="aspect-square flex items-center justify-center text-[8px] font-bold border border-neutral-900/30 rounded"
                            title={`True: ${settings.CLASS_NAMES[rIdx]}, Pred: ${settings.CLASS_NAMES[cIdx]} = ${val}`}
                          >
                            {val}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: DATASET DETAILS */}
        {activeTab === 'dataset' && (
          <div className="space-y-6 animate-fade-in">
            {datasetInfo ? (
              <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-6">
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">{datasetInfo.name}</span>
                  <h3 className="text-xl font-bold">Colorectal Histology Patches</h3>
                  <p className="text-neutral-400 text-sm leading-relaxed max-w-4xl">{datasetInfo.description}</p>
                </div>
                
                <div className="grid md:grid-cols-3 gap-6 pt-4 border-t border-neutral-850">
                  <div className="bg-neutral-950/40 p-4 border border-neutral-900 rounded-lg space-y-1">
                    <span className="text-[10px] text-neutral-500 uppercase font-semibold">Image Dimensions</span>
                    <div className="text-lg font-bold">{datasetInfo.image_dimension}</div>
                  </div>
                  <div className="bg-neutral-950/40 p-4 border border-neutral-900 rounded-lg space-y-1">
                    <span className="text-[10px] text-neutral-500 uppercase font-semibold">Total Slide Classes</span>
                    <div className="text-lg font-bold">{datasetInfo.num_classes} Distinct Tissues</div>
                  </div>
                  <div className="bg-neutral-950/40 p-4 border border-neutral-900 rounded-lg space-y-1">
                    <span className="text-[10px] text-neutral-500 uppercase font-semibold">Splits (Train / Val / Test)</span>
                    <div className="text-lg font-bold">
                      {datasetInfo.splits?.train.toLocaleString()} / {datasetInfo.splits?.validation.toLocaleString()} / {datasetInfo.splits?.test.toLocaleString()}
                    </div>
                  </div>
                </div>

                <div className="space-y-3 pt-4 border-t border-neutral-850">
                  <h4 className="text-sm font-bold text-neutral-300">Tissue Classification Labels</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {datasetInfo.classes?.map((cls, idx) => (
                      <div key={idx} className="flex items-center space-x-2 text-xs bg-neutral-950/20 border border-neutral-900 p-2.5 rounded-lg">
                        <span className="h-2 w-2 rounded-full bg-indigo-500/80" />
                        <span className="font-semibold text-neutral-500">[{idx}]</span>
                        <span className="capitalize text-neutral-300 font-medium">{cls}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-neutral-500">Loading dataset details...</div>
            )}
          </div>
        )}

        {/* TAB 5: ABOUT & CREDENTIALS */}
        {activeTab === 'about' && (
          <div className="border border-neutral-900 bg-neutral-900/20 p-6 rounded-xl space-y-6 animate-fade-in">
            <h3 className="text-lg font-bold">Architecture & MLOps Infrastructure</h3>
            
            <div className="space-y-4 text-sm text-neutral-400 leading-relaxed max-w-4xl">
              <p>
                This platform is built as a complete medical validation stack comparing standard convolutions with attention-based tokens. 
              </p>
              <h4 className="font-bold text-neutral-200">MLOps Infostructure:</h4>
              <ul className="list-disc pl-5 space-y-2">
                <li><strong className="text-neutral-300">FastAPI Async Server:</strong> Resolves CORS headers, registers Multipart file uploads, and calls python image transformation utilities on-demand.</li>
                <li><strong className="text-neutral-300">Custom CNN Architecture:</strong> 3 blocks of convolutions featuring batch normalization and Adaptive Pooling. Designed to yield very lightweight models suitable for edge device inference.</li>
                <li><strong className="text-neutral-300">Vision Transformer (ViT):</strong> Loads `vit_tiny_patch16_224` from `timm` libraries. Backbones are frozen to keep fine-tuning compute requirements minimal, utilizing attention hook mapping on the last layer for pixel-level attention explainability.</li>
                <li><strong className="text-neutral-300">Docker Containerization:</strong> Bundled using `docker-compose` to package front-end Nginx servers and backend FastAPI services together, preparing them for cloud hosts like Render, Railway, or Google Cloud Run.</li>
              </ul>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-neutral-900 bg-neutral-950/40 px-6 py-4 text-center text-xs text-neutral-600">
        &copy; {new Date().getFullYear()} Medical AI Model Comparison and Explainability Platform. All rights reserved.
      </footer>
    </div>
  );
}

export default App;
