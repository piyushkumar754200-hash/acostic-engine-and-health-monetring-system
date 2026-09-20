import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Pause, Disc, Cpu, CheckCircle2, AlertTriangle, ShieldCheck, Download, RefreshCw, BarChart2, Layers, Info, Car, Wrench } from 'lucide-react';
import { api } from '../services/api';

export default function AnalyzePage({ setActivePage }) {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [activeTab, setActiveTab] = useState('waveform'); // 'waveform', 'spectrogram', 'mfcc', 'features'
  const [errorMsg, setErrorMsg] = useState(null);

  const audioRef = useRef(null);
  const canvasRef = useRef(null);

  // Form state
  const [vehicleInfo, setVehicleInfo] = useState({
    brand: 'Toyota',
    model: 'Camry V6',
    year: 2021,
    engine_type: '3.5L V6 DOHC',
    fuel_type: 'Gasoline',
    mileage: 52000,
    notes: 'Slight metallic vibration noticed around 2500 RPM.'
  });

  const processingSteps = [
    'Uploading audio recording...',
    'Preprocessing audio signal & resampling to 22.05 kHz...',
    'Extracting 20 MFCCs, Mel Spectrogram & Spectral Centroids...',
    'Running Random Forest & SVM feature classifiers...',
    'Running 1D CNN, 2D Mel-CNN & LSTM recurrent models...',
    'Executing Weighted Soft Voting Ensemble consensus...',
    'Generating diagnostic report & explainable AI breakdown...'
  ];

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setErrorMsg(null);
    setAnalysisResult(null);
    const url = URL.createObjectURL(selectedFile);
    setAudioUrl(url);
  };

  // Synthetic sample generators in browser as fallback if user has no local audio file
  const generateSampleAudio = (type) => {
    const sr = 22050;
    const duration = 3.0;
    const numSamples = sr * duration;
    const buffer = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
      const t = i / sr;
      let s = 0.4 * Math.sin(2 * Math.PI * 35 * t) + 0.2 * Math.sin(2 * Math.PI * 70 * t);
      
      if (type === 'misfire' && (t % 0.5 < 0.08)) {
        s *= 0.1;
        s += 0.3 * (Math.random() * 2 - 1);
      } else if (type === 'bearing' && (t % 0.3 < 0.15)) {
        s += 0.3 * Math.sin(2 * Math.PI * 2500 * t);
      } else if (type === 'knocking' && (t % 0.4 < 0.03)) {
        s += 0.7 * Math.sin(2 * Math.PI * 5000 * t) * Math.exp(-(t % 0.4) * 100);
      }
      buffer[i] = Math.max(-1, Math.min(1, s));
    }

    // Convert float buffer to WAV blob
    const wavBlob = bufferToWav(buffer, sr);
    const sampleFile = new File([wavBlob], `sample_engine_${type}.wav`, { type: 'audio/wav' });
    handleFileChange(sampleFile);
  };

  // Helper function to turn Float32Array to WAV Blob
  function bufferToWav(buffer, sampleRate) {
    const numOfChan = 1;
    const length = buffer.length * 2 + 44;
    const out = new DataView(new ArrayBuffer(length));
    let channels = [buffer], sample, offset = 0, pos = 0;

    function setUint16(data) { out.setUint16(pos, data, true); pos += 2; }
    function setUint32(data) { out.setUint32(pos, data, true); pos += 4; }

    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"
    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16);         // length = 16
    setUint16(1);          // PCM
    setUint16(numOfChan);  // mono
    setUint32(sampleRate);
    setUint32(sampleRate * 2);
    setUint16(2);          // block align
    setUint16(16);         // bits per sample
    setUint32(0x61746164); // "data" chunk
    setUint32(buffer.length * 2);

    for (let i = 0; i < buffer.length; i++) {
      sample = Math.max(-1, Math.min(1, buffer[i]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      out.setInt16(44 + offset, sample, true);
      offset += 2;
    }

    return new Blob([out], { type: 'audio/wav' });
  }

  // Draw Audio Waveform on Canvas
  useEffect(() => {
    if (!canvasRef.current || !analysisResult?.visualizations?.waveform) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const data = analysisResult.visualizations.waveform;

    ctx.clearRect(0, 0, width, height);

    // Draw background grid
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw waveform gradient
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, '#00f0ff');
    gradient.addColorStop(0.5, '#3b82f6');
    gradient.addColorStop(1, '#8b5cf6');

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.beginPath();

    const sliceWidth = width / data.length;
    let x = 0;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const y = (1 - v) * (height / 2);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      x += sliceWidth;
    }

    ctx.stroke();
  }, [analysisResult, activeTab]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleStartAnalysis = async () => {
    if (!file) {
      setErrorMsg('Please select or upload an engine audio recording first.');
      return;
    }

    setIsProcessing(true);
    setCurrentStep(0);
    setErrorMsg(null);

    // Simulate multi-step progress UI indicator
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < processingSteps.length - 1) return prev + 1;
        clearInterval(interval);
        return prev;
      });
    }, 450);

    try {
      const result = await api.analyzeEngineSound(file, vehicleInfo);
      clearInterval(interval);
      setAnalysisResult(result);
    } catch (err) {
      clearInterval(interval);
      setErrorMsg(err.message || 'Failed to analyze sound file.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Disc className="w-8 h-8 text-cyan-400" />
            Engine Sound Analysis
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Upload acoustic recordings for multi-model feature extraction & AI fault diagnosis.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 font-mono bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg">
            Supported Formats: WAV • MP3 • FLAC • OGG
          </span>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          {errorMsg}
        </div>
      )}

      {/* Main Analysis Input Section (Shown when no result yet or re-analyzing) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left 7 Columns: File Upload & Preloader */}
        <div className="lg:col-span-7 space-y-6">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Upload className="w-5 h-5 text-cyan-400" />
              Audio Input & Preview
            </h3>

            {/* Drag & Drop Dropzone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files?.[0]) handleFileChange(e.dataTransfer.files[0]);
              }}
              className="border-2 border-dashed border-slate-700 hover:border-cyan-500/50 rounded-xl p-8 text-center transition-colors bg-slate-950/40 cursor-pointer group"
            >
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
                className="hidden"
                id="audio-upload-input"
              />
              <label htmlFor="audio-upload-input" className="cursor-pointer space-y-3 block">
                <div className="w-14 h-14 mx-auto rounded-full bg-cyan-950/80 border border-cyan-800/50 flex items-center justify-center group-hover:scale-110 transition-transform">
                  <Upload className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">Click or drag audio file here</p>
                  <p className="text-xs text-slate-400 mt-1">Supports WAV, MP3, FLAC, OGG up to 50MB</p>
                </div>
              </label>
            </div>

            {/* Quick Demo Preloaded Sample Triggers */}
            <div className="pt-2">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Don't have an audio file? Test with synthetic sample recordings:
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {[
                  { label: 'Normal Engine', type: 'normal', color: 'border-emerald-800/50 text-emerald-400' },
                  { label: 'Misfire Sound', type: 'misfire', color: 'border-amber-800/50 text-amber-400' },
                  { label: 'Bearing Wear', type: 'bearing', color: 'border-rose-800/50 text-rose-400' },
                  { label: 'Detonation Knock', type: 'knocking', color: 'border-purple-800/50 text-purple-400' }
                ].map((sample) => (
                  <button
                    key={sample.type}
                    type="button"
                    onClick={() => generateSampleAudio(sample.type)}
                    className={`px-3 py-2 rounded-lg bg-slate-900 border text-xs font-medium hover:bg-slate-800 transition-all ${sample.color}`}
                  >
                    + {sample.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Active Selected File Preview Box */}
            {file && (
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={togglePlay}
                      className="w-10 h-10 rounded-full bg-cyan-500 hover:bg-cyan-400 text-slate-950 flex items-center justify-center shadow-lg shadow-cyan-500/30 transition-transform active:scale-95"
                    >
                      {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
                    </button>
                    <div>
                      <p className="text-sm font-semibold text-white truncate max-w-[240px]">{file.name}</p>
                      <p className="text-xs text-slate-400 font-mono">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                    </div>
                  </div>
                  <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-cyan-400 font-mono">Ready</span>
                </div>
                {audioUrl && (
                  <audio
                    ref={audioRef}
                    src={audioUrl}
                    onEnded={() => setIsPlaying(false)}
                    className="w-full h-8 rounded"
                    controlsList="nodownload"
                  />
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right 5 Columns: Vehicle Metadata Form */}
        <div className="lg:col-span-5 space-y-6">
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Car className="w-5 h-5 text-cyan-400" />
              Vehicle Specifications (Optional)
            </h3>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Make / Brand</label>
                <input
                  type="text"
                  value={vehicleInfo.brand}
                  onChange={(e) => setVehicleInfo({ ...vehicleInfo, brand: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Model Name</label>
                <input
                  type="text"
                  value={vehicleInfo.model}
                  onChange={(e) => setVehicleInfo({ ...vehicleInfo, model: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Manufacturing Year</label>
                <input
                  type="number"
                  value={vehicleInfo.year}
                  onChange={(e) => setVehicleInfo({ ...vehicleInfo, year: parseInt(e.target.value) || 2020 })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Engine Configuration</label>
                <input
                  type="text"
                  value={vehicleInfo.engine_type}
                  onChange={(e) => setVehicleInfo({ ...vehicleInfo, engine_type: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Fuel Type</label>
                <select
                  value={vehicleInfo.fuel_type}
                  onChange={(e) => setVehicleInfo({ ...vehicleInfo, fuel_type: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                >
                  <option value="Gasoline">Gasoline / Petrol</option>
                  <option value="Diesel">Diesel</option>
                  <option value="Hybrid">Hybrid Electric</option>
                </select>
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Odometer (KM)</label>
                <input
                  type="number"
                  value={vehicleInfo.mileage}
                  onChange={(e) => setVehicleInfo({ ...vehicleInfo, mileage: parseInt(e.target.value) || 0 })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-white focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Symptoms / Notes</label>
              <textarea
                rows={2}
                value={vehicleInfo.notes}
                onChange={(e) => setVehicleInfo({ ...vehicleInfo, notes: e.target.value })}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <button
              onClick={handleStartAnalysis}
              disabled={isProcessing || !file}
              className={`w-full py-3.5 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-xl transition-all ${
                isProcessing || !file
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-cyan-500/25 hover:scale-[1.02]'
              }`}
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
                  Running Neural Pipeline...
                </>
              ) : (
                <>
                  <Cpu className="w-5 h-5 text-white" />
                  Start AI Diagnosis
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Animated Multi-Step Processing Progress Box */}
      {isProcessing && (
        <div className="glass-panel p-8 rounded-2xl border border-cyan-500/30 glow-cyan space-y-6 animate-pulse">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Disc className="w-5 h-5 text-cyan-400 animate-spin" />
              EngineSense AI Processing Pipeline
            </h3>
            <span className="text-xs font-mono text-cyan-400">Step {currentStep + 1} of {processingSteps.length}</span>
          </div>

          <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-600 transition-all duration-300"
              style={{ width: `${((currentStep + 1) / processingSteps.length) * 100}%` }}
            />
          </div>

          <p className="text-sm font-mono text-cyan-300">{processingSteps[currentStep]}</p>
        </div>
      )}

      {/* Diagnostic Results Section (Rendered upon complete analysis) */}
      {analysisResult && !isProcessing && (
        <div className="space-y-8 animate-fade-in">
          {/* Top Banner: Overall Diagnosis */}
          <div className={`glass-panel p-8 rounded-2xl border ${
            analysisResult.predicted_class_name === 'normal'
              ? 'border-emerald-500/40 bg-emerald-950/20'
              : 'border-rose-500/40 bg-rose-950/20'
          } relative overflow-hidden`}>
            
            {analysisResult.is_demo && (
              <div className="mb-4 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950/80 border border-amber-800/50 text-amber-400 text-xs font-mono">
                <Info className="w-3.5 h-3.5" />
                Demo Acoustic Fallback (Run Training Pipeline in Model Performance page for full neural weights)
              </div>
            )}

            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  {analysisResult.predicted_class_name === 'normal' ? (
                    <span className="p-2 rounded-xl bg-emerald-950 border border-emerald-800 text-emerald-400">
                      <CheckCircle2 className="w-8 h-8" />
                    </span>
                  ) : (
                    <span className="p-2 rounded-xl bg-rose-950 border border-rose-800 text-rose-400">
                      <AlertTriangle className="w-8 h-8" />
                    </span>
                  )}
                  <div>
                    <p className="text-xs uppercase font-mono tracking-widest text-slate-400">Ensemble Diagnosis</p>
                    <h2 className={`text-3xl font-extrabold tracking-tight ${
                      analysisResult.predicted_class_name === 'normal' ? 'text-emerald-400' : 'text-rose-400'
                    }`}>
                      {analysisResult.predicted_condition.toUpperCase()}
                    </h2>
                  </div>
                </div>
                <p className="text-sm text-slate-300 max-w-2xl mt-2 leading-relaxed">
                  {analysisResult.recommendation}
                </p>
              </div>

              {/* Confidence Badge & Agreement */}
              <div className="flex items-center gap-6 bg-slate-950/80 p-4 rounded-xl border border-slate-800 shrink-0">
                <div className="text-center">
                  <p className="text-[11px] text-slate-400 uppercase font-mono">Confidence</p>
                  <p className="text-3xl font-black text-cyan-400 font-mono">
                    {(analysisResult.overall_confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="h-10 w-px bg-slate-800" />
                <div className="text-center">
                  <p className="text-[11px] text-slate-400 uppercase font-mono">Severity</p>
                  <span className={`inline-block px-2.5 py-1 rounded text-xs font-bold font-mono mt-1 ${
                    analysisResult.severity === 'Critical' || analysisResult.severity === 'High'
                      ? 'bg-rose-950 text-rose-400 border border-rose-800'
                      : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                  }`}>
                    {analysisResult.severity}
                  </span>
                </div>
              </div>
            </div>

            {/* Action Bar */}
            <div className="mt-6 pt-6 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
                <span>Analysis ID: <strong className="text-white">{analysisResult.analysis_id}</strong></span>
                <span>Time: <strong className="text-white">{analysisResult.processing_time_seconds}s</strong></span>
              </div>
              <div className="flex items-center gap-3">
                <a
                  href={api.getReportUrl(analysisResult.analysis_id)}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all"
                >
                  <Download className="w-4 h-4" />
                  Download PDF Report
                </a>
              </div>
            </div>
          </div>

          {/* Model Predictions Breakdown Grid */}
          <div className="space-y-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-cyan-400" />
              Multi-Model Prediction Breakdown
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(analysisResult.model_predictions).map(([mId, mRes]) => (
                <div key={mId} className="glass-panel p-5 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-bold text-white">{mRes.model_name}</h4>
                      <p className="text-[11px] text-slate-400">{mRes.type}</p>
                    </div>
                    <span className="text-xs font-mono font-bold text-cyan-400">
                      {(mRes.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="w-full h-1.5 rounded-full bg-slate-950 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                      style={{ width: `${mRes.confidence * 100}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs pt-1 border-t border-slate-800/50">
                    <span className="text-slate-400">Class Output:</span>
                    <span className="font-mono font-semibold text-white uppercase">{mRes.prediction}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Audio Visualization Tabs */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-cyan-400" />
                Acoustic Signal Visualizations
              </h3>

              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs font-medium">
                {['waveform', 'features'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-3 py-1.5 rounded-md capitalize transition-colors ${
                      activeTab === tab ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
            </div>

            {/* Waveform Tab */}
            {activeTab === 'waveform' && (
              <div className="space-y-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 relative">
                  <canvas ref={canvasRef} width={800} height={180} className="w-full h-44 rounded" />
                </div>
                <p className="text-xs text-slate-400 text-center font-mono">
                  Time-Domain Signal Waveform Amplitude (Downsampled 300 Frames)
                </p>
              </div>
            )}

            {/* Features Stats Tab */}
            {activeTab === 'features' && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <p className="text-slate-400 mb-1">RMS Energy (Mean)</p>
                  <p className="text-lg font-bold text-cyan-400">{analysisResult.feature_stats.rms_mean.toFixed(4)}</p>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <p className="text-slate-400 mb-1">Zero Crossing Rate</p>
                  <p className="text-lg font-bold text-blue-400">{analysisResult.feature_stats.zcr_mean.toFixed(4)}</p>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <p className="text-slate-400 mb-1">Spectral Centroid</p>
                  <p className="text-lg font-bold text-emerald-400">{analysisResult.feature_stats.spectral_centroid_mean.toFixed(1)} Hz</p>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <p className="text-slate-400 mb-1">Spectral Bandwidth</p>
                  <p className="text-lg font-bold text-purple-400">{analysisResult.feature_stats.spectral_bandwidth_mean.toFixed(1)} Hz</p>
                </div>
              </div>
            )}
          </div>

          {/* Explainable AI Section */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              Explainable AI — Key Contributing Acoustic Features
            </h3>
            <p className="text-xs text-slate-400">
              Top acoustic feature importances derived from Random Forest feature trees & spectral perturbation.
            </p>

            <div className="space-y-3 pt-2">
              {analysisResult.explainable_ai.top_acoustic_features.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-200 font-medium">{item.feature}</span>
                    <span className="font-mono text-cyan-400 font-bold">{(item.importance * 100).toFixed(0)}% contribution</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                      style={{ width: `${item.importance * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
