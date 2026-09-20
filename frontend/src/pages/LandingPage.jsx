import React, { useState } from 'react';
import { Activity, Disc, Cpu, ShieldCheck, Zap, BarChart2, Layers, CheckCircle, ArrowRight, HelpCircle, ChevronDown, Radio, FileText } from 'lucide-react';

export default function LandingPage({ setActivePage }) {
  const [openFaq, setOpenFaq] = useState(0);

  const features = [
    {
      icon: Activity,
      title: 'AI-Powered Acoustic Diagnosis',
      desc: 'Deep signal processing and pattern recognition algorithms to detect early mechanical anomalies from sound vibration signatures.'
    },
    {
      icon: Cpu,
      title: 'Multi-Model Neural Pipeline',
      desc: 'Simultaneously executes Random Forest, SVM, 1D CNN, 2D Mel-Spectrogram CNN, and LSTM recurrent networks.'
    },
    {
      icon: Layers,
      title: 'Weighted Soft Voting Ensemble',
      desc: 'Combines independent model probability vectors into a unified consensus diagnosis with model agreement metrics.'
    },
    {
      icon: BarChart2,
      title: 'Explainable Acoustic AI',
      desc: 'Transparent diagnostic reasoning showing top contributing frequency bands, MFCC coefficients, and RMS energy levels.'
    },
    {
      icon: Radio,
      title: 'Real-Time Microphone Stream',
      desc: 'Experimental browser-based Web Audio API recorder for live engine acoustic evaluation in workshop environments.'
    },
    {
      icon: FileText,
      title: 'Automated PDF Diagnostic Reports',
      desc: 'Generates comprehensive engineering reports with embedded spectral figures, vehicle metadata, and repair recommendations.'
    }
  ];

  const workflowSteps = [
    { step: '01', title: 'Audio Capture', desc: 'Upload engine sound recording (WAV, MP3, FLAC, OGG, M4A) or stream via live microphone.' },
    { step: '02', title: 'Feature Extraction', desc: 'Compute 20 MFCCs, Mel Spectrogram, Spectral Centroid, Bandwidth, Rolloff, and Zero Crossing Rate.' },
    { step: '03', title: 'Multi-Model Inference', desc: 'Parallel evaluation across classical machine learning classifiers and deep convolutional/recurrent neural networks.' },
    { step: '04', title: 'Ensemble Diagnosis', desc: 'Generate overall health condition, confidence score, model agreement percentage, and repair report.' }
  ];

  const faqs = [
    {
      q: 'How does EngineSense AI detect vehicle engine faults from sound?',
      a: 'Mechanical engine faults (misfires, bearing wear, valve lash, knocking) alter the acoustic vibration patterns. EngineSense AI uses Librosa to decompose audio into spectral heatmaps and MFCCs, which neural networks match against learned fault acoustic fingerprints.'
    },
    {
      q: 'What engine fault categories are currently supported?',
      a: 'The system categorizes sounds into Normal Engine Operation, Cylinder Misfire, Crankshaft/Bearing Wear, Valve Train Tappet Noise, and Detonation Engine Knocking.'
    },
    {
      q: 'Can I run EngineSense AI without internet connectivity?',
      a: 'Yes. The backend runs completely locally using standard Python libraries (FastAPI, PyTorch, Scikit-learn) with SQLite storage fallback.'
    },
    {
      q: 'How accurate is the multi-model ensemble?',
      a: 'The weighted ensemble combines traditional statistical classifiers with deep learning models, cross-validating predictions across all 5 models to minimize false positives.'
    }
  ];

  return (
    <div className="space-y-24 py-8">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-8 pb-12">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[350px] bg-cyan-500/10 blur-[140px] rounded-full pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-900/80 border border-cyan-500/30 text-cyan-400 text-xs font-semibold tracking-wider uppercase mb-8 shadow-lg shadow-cyan-500/10">
            <Zap className="w-4 h-4 text-cyan-400" />
            Next-Gen AI Vehicle Acoustic Diagnostics
          </div>

          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white mb-6 leading-tight">
            Listen to Your Engine.<br />
            <span className="text-gradient">Understand Its Acoustic Health.</span>
          </h1>

          <p className="max-w-3xl mx-auto text-slate-400 text-base sm:text-lg mb-10 leading-relaxed">
            EngineSense AI analyzes vehicle engine recordings using signal processing and an ensemble of 5 Machine Learning & Deep Learning models to detect mechanical anomalies before critical failures occur.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={() => setActivePage('analyze')}
              className="flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-base shadow-xl shadow-cyan-500/30 transition-all hover:scale-105"
            >
              <Disc className="w-5 h-5 animate-spin-slow" />
              Analyze Engine Sound
            </button>
            <button
              onClick={() => setActivePage('models')}
              className="flex items-center gap-2 px-6 py-4 rounded-xl glass-panel hover:bg-slate-800/60 text-slate-200 font-semibold text-base transition-all border border-slate-700/80"
            >
              Explore Technology & Models
              <ArrowRight className="w-4 h-4 text-cyan-400" />
            </button>
          </div>

          {/* Interactive Audio Hero Graphic */}
          <div className="mt-16 max-w-4xl mx-auto glass-panel p-6 rounded-2xl border border-slate-800 shadow-2xl relative">
            <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                Live Diagnostic Waveform Frequency Analyzer
              </div>
              <span className="text-xs text-slate-400 font-mono">22,050 Hz • 16-Bit PCM • 5 Models Active</span>
            </div>

            {/* Wave Equalizer Animation */}
            <div className="h-28 flex items-center justify-center gap-1.5 px-4 bg-slate-950/80 rounded-xl border border-slate-800/80 overflow-hidden">
              {[...Array(40)].map((_, i) => {
                const anims = ['animate-equalizer-1', 'animate-equalizer-2', 'animate-equalizer-3', 'animate-equalizer-4', 'animate-equalizer-5'];
                return (
                  <div
                    key={i}
                    className={`w-1.5 rounded-full bg-gradient-to-t from-cyan-500 via-blue-500 to-indigo-400 ${anims[i % 5]}`}
                    style={{ animationDelay: `${(i % 7) * 0.15}s` }}
                  />
                );
              })}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 text-left">
              <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                <p className="text-[11px] text-slate-400">Feature Dimensions</p>
                <p className="text-sm font-bold text-white font-mono">20 MFCC + Mel 128</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                <p className="text-[11px] text-slate-400">Supported Models</p>
                <p className="text-sm font-bold text-cyan-400 font-mono">5 ML/DL Networks</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                <p className="text-[11px] text-slate-400">Ensemble Voting</p>
                <p className="text-sm font-bold text-emerald-400 font-mono">Weighted Soft Voting</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
                <p className="text-[11px] text-slate-400">Average Speed</p>
                <p className="text-sm font-bold text-indigo-400 font-mono">&lt; 0.45 sec</p>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* Key Stats Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { label: 'Trained ML & DL Models', val: '5 Neural Nets', sub: 'RF, SVM, 1D/2D CNN, LSTM' },
            { label: 'Acoustic Features Extracted', val: '45+ Metrics', sub: 'MFCCs, Mel, ZCR, RMS' },
            { label: 'Inference Processing Time', val: '< 0.5 sec', sub: 'Real-time classification' },
            { label: 'Supported Audio Formats', val: '5 Formats', sub: 'WAV, MP3, FLAC, OGG, M4A' }
          ].map((stat, i) => (
            <div key={i} className="glass-panel p-6 rounded-xl border border-slate-800/80 glass-panel-hover">
              <p className="text-xs font-semibold text-cyan-400 uppercase tracking-wider mb-1">{stat.label}</p>
              <p className="text-2xl font-bold text-white mb-1">{stat.val}</p>
              <p className="text-xs text-slate-400">{stat.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-white mb-4">Production AI Acoustic Diagnostics</h2>
          <p className="text-slate-400 max-w-2xl mx-auto text-sm">
            Designed for automotive technicians, fleet managers, and mechanical research engineers.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div key={idx} className="glass-panel p-6 rounded-2xl border border-slate-800 glass-panel-hover group">
                <div className="w-12 h-12 rounded-xl bg-cyan-950/60 border border-cyan-800/50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <Icon className="w-6 h-6 text-cyan-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{item.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Technology Workflow Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="glass-panel p-8 sm:p-12 rounded-3xl border border-slate-800 relative overflow-hidden">
          <div className="text-center mb-12">
            <span className="text-xs font-mono text-cyan-400 uppercase tracking-widest px-3 py-1 rounded-full bg-cyan-950 border border-cyan-800/50">
              End-to-End Pipeline
            </span>
            <h2 className="text-3xl font-bold text-white mt-4">How EngineSense AI Works</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 relative">
            {workflowSteps.map((s, idx) => (
              <div key={idx} className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 relative">
                <span className="text-3xl font-extrabold text-cyan-500/30 font-mono block mb-2">{s.step}</span>
                <h4 className="text-base font-bold text-white mb-2">{s.title}</h4>
                <p className="text-xs text-slate-400 leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Accordion */}
      <section className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-white mb-3">Frequently Asked Questions</h2>
          <p className="text-slate-400 text-sm">Everything you need to know about EngineSense AI</p>
        </div>

        <div className="space-y-4">
          {faqs.map((faq, i) => (
            <div key={i} className="glass-panel rounded-xl border border-slate-800 overflow-hidden">
              <button
                onClick={() => setOpenFaq(openFaq === i ? -1 : i)}
                className="w-full p-5 text-left flex items-center justify-between text-white font-semibold text-sm hover:text-cyan-400 transition-colors"
              >
                <span>{faq.q}</span>
                <ChevronDown className={`w-5 h-5 text-slate-400 transition-transform ${openFaq === i ? 'rotate-180 text-cyan-400' : ''}`} />
              </button>
              {openFaq === i && (
                <div className="p-5 pt-0 text-xs text-slate-400 leading-relaxed border-t border-slate-800/50">
                  {faq.a}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
