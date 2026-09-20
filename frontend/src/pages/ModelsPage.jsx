import React, { useState, useEffect } from 'react';
import { Cpu, RefreshCw, BarChart2, CheckCircle2, Sliders, AlertCircle, Info, Database } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { api } from '../services/api';

export default function ModelsPage() {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [msg, setMsg] = useState(null);

  const fetchPerformance = async () => {
    setLoading(true);
    try {
      const data = await api.getModelPerformance();
      setPerformanceData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPerformance();
  }, []);

  const handleTrain = async (runTuning = false) => {
    setIsTraining(true);
    setMsg(null);
    try {
      const res = await api.trainModels(runTuning);
      setMsg(res.message || 'Training pipeline initiated.');
      // Poll after 5 seconds to get updated metrics
      setTimeout(() => {
        fetchPerformance();
        setIsTraining(false);
      }, 5000);
    } catch (err) {
      setMsg(`Error: ${err.message}`);
      setIsTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center space-y-4">
        <RefreshCw className="w-10 h-10 text-cyan-400 animate-spin mx-auto" />
        <p className="text-slate-400 text-sm">Loading Neural Model Evaluation Metrics...</p>
      </div>
    );
  }

  const isTrained = performanceData?.is_trained;
  const modelsList = performanceData?.models || [];
  const confusionMatrix = performanceData?.overall_confusion_matrix || performanceData?.models?.[0]?.confusion_matrix || [];
  const faultClasses = performanceData?.fault_classes || [
    { id: 'normal', name: 'Normal Engine' },
    { id: 'misfire', name: 'Misfire' },
    { id: 'bearing_fault', name: 'Bearing Fault' },
    { id: 'valve_fault', name: 'Valve Fault' },
    { id: 'knocking', name: 'Knocking' }
  ];

  // Recharts Chart Data
  const chartData = modelsList.map((m) => ({
    name: m.model_name.replace('Classifier', '').replace('Recurrent Neural Network', 'RNN'),
    Accuracy: +(m.accuracy * 100).toFixed(1),
    F1Score: +(m.f1_score * 100).toFixed(1),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    InferenceMs: m.inference_time_ms
  }));

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <Cpu className="w-8 h-8 text-cyan-400" />
            Model Performance & Comparison
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Empirical evaluation results loaded directly from cross-validation testing.
          </p>
        </div>

        {/* Action Triggers */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => handleTrain(true)}
            disabled={isTraining}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-semibold transition-all"
          >
            <Sliders className="w-4 h-4 text-cyan-400" />
            Run Hyperparameter Tuning
          </button>
          <button
            onClick={() => handleTrain(false)}
            disabled={isTraining}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all"
          >
            <RefreshCw className={`w-4 h-4 ${isTraining ? 'animate-spin' : ''}`} />
            {isTraining ? 'Training Models...' : 'Retrain All Models'}
          </button>
        </div>
      </div>

      {msg && (
        <div className="p-4 rounded-xl bg-cyan-950/60 border border-cyan-800 text-cyan-300 text-sm flex items-center gap-3">
          <Info className="w-5 h-5 text-cyan-400 shrink-0" />
          {msg}
        </div>
      )}

      {/* Untrained Banner */}
      {!isTrained && (
        <div className="glass-panel p-8 rounded-2xl border border-amber-500/40 bg-amber-950/20 text-center space-y-4">
          <AlertCircle className="w-12 h-12 text-amber-400 mx-auto" />
          <h3 className="text-xl font-bold text-white">Models Not Yet Trained</h3>
          <p className="text-slate-400 text-sm max-w-xl mx-auto">
            No evaluation metrics file was found. Click the button below to generate synthetic dataset samples and run full training pipeline.
          </p>
          <button
            onClick={() => handleTrain(false)}
            disabled={isTraining}
            className="px-6 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-sm shadow-lg shadow-amber-500/25"
          >
            Run Training Pipeline Now
          </button>
        </div>
      )}

      {/* Dataset Summary Banner */}
      {isTrained && performanceData?.dataset_summary && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
          <div>
            <p className="text-slate-400">Total Dataset Samples</p>
            <p className="text-2xl font-bold text-white mt-1">{performanceData.dataset_summary.total_samples}</p>
          </div>
          <div>
            <p className="text-slate-400">Training Samples</p>
            <p className="text-2xl font-bold text-cyan-400 mt-1">{performanceData.dataset_summary.train_samples}</p>
          </div>
          <div>
            <p className="text-slate-400">Test Samples (Stratified)</p>
            <p className="text-2xl font-bold text-blue-400 mt-1">{performanceData.dataset_summary.test_samples}</p>
          </div>
          <div>
            <p className="text-slate-400">Target Classes</p>
            <p className="text-2xl font-bold text-emerald-400 mt-1">{performanceData.dataset_summary.classes?.length || 5} Classes</p>
          </div>
        </div>
      )}

      {/* Model Performance Table */}
      {isTrained && (
        <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden space-y-4 p-6">
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            Model Evaluation Metrics Summary Table
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase bg-slate-900/60">
                  <th className="p-3">Model Architecture</th>
                  <th className="p-3">Accuracy</th>
                  <th className="p-3">Precision</th>
                  <th className="p-3">Recall</th>
                  <th className="p-3">F1 Score</th>
                  <th className="p-3">Inference Speed</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-200">
                {modelsList.map((m, idx) => (
                  <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                    <td className="p-3 font-semibold text-white flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      {m.model_name}
                    </td>
                    <td className="p-3 font-mono text-cyan-400 font-bold">{(m.accuracy * 100).toFixed(1)}%</td>
                    <td className="p-3 font-mono text-blue-400">{(m.precision * 100).toFixed(1)}%</td>
                    <td className="p-3 font-mono text-emerald-400">{(m.recall * 100).toFixed(1)}%</td>
                    <td className="p-3 font-mono text-purple-400 font-bold">{(m.f1_score * 100).toFixed(1)}%</td>
                    <td className="p-3 font-mono text-slate-300">{m.inference_time_ms} ms</td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">
                        Validated
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Performance Bar Charts */}
      {isTrained && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Accuracy & F1 Score Comparison */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-cyan-400" />
              Accuracy & F1-Score Comparison (%)
            </h3>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="Accuracy" fill="#00f0ff" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="F1Score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Inference Time Comparison */}
          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-purple-400" />
              Inference Latency per Sample (ms)
            </h3>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Bar dataKey="InferenceMs" fill="#a855f7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Interactive Confusion Matrix Section */}
      {isTrained && confusionMatrix.length > 0 && (
        <div className="glass-panel p-8 rounded-2xl border border-slate-800 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-emerald-400" />
                Multi-Class Confusion Matrix Heatmap
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Actual Ground Truth vs Ensemble Predicted Class
              </p>
            </div>
            <span className="text-xs px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-slate-400 italic">
              "Diagonal values represent correctly classified samples."
            </span>
          </div>

          <div className="overflow-x-auto pt-4">
            <div className="min-w-[450px] max-w-2xl mx-auto space-y-2">
              <div className="grid grid-cols-6 gap-2 text-center text-xs font-mono font-bold text-slate-400 pb-2 border-b border-slate-800">
                <div className="text-left text-slate-500">Actual ↓ / Pred →</div>
                {faultClasses.map((fc) => (
                  <div key={fc.id} className="truncate px-1" title={fc.name}>
                    {fc.id}
                  </div>
                ))}
              </div>

              {confusionMatrix.map((row, rIdx) => (
                <div key={rIdx} className="grid grid-cols-6 gap-2 items-center text-center text-xs font-mono">
                  <div className="text-left font-semibold text-slate-300 truncate" title={faultClasses[rIdx]?.name}>
                    {faultClasses[rIdx]?.id || `Class ${rIdx}`}
                  </div>

                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const maxVal = Math.max(...row.flat(), 1);
                    const intensity = Math.min(1.0, val / maxVal);
                    
                    return (
                      <div
                        key={cIdx}
                        className={`p-3 rounded-lg border font-bold transition-transform hover:scale-105 ${
                          isDiagonal
                            ? 'bg-emerald-950/80 text-emerald-300 border-emerald-600/50'
                            : val > 0
                            ? 'bg-rose-950/80 text-rose-300 border-rose-800/50'
                            : 'bg-slate-950/40 text-slate-600 border-slate-900'
                        }`}
                      >
                        {val}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
