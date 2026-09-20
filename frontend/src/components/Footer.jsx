import React from 'react';
import { Activity, ShieldCheck, Cpu, GitBranch } from 'lucide-react';

export default function Footer({ setActivePage }) {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 text-slate-400 py-12 text-sm mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <Activity className="w-6 h-6 text-cyan-400" />
              <span className="font-bold text-lg text-white tracking-wider">ENGINE<span className="text-cyan-400">SENSE</span> AI</span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed">
              State-of-the-art acoustic signal processing and deep multi-model neural network architecture for automotive vehicle diagnosis.
            </p>
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-400/80">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              Production Diagnostic Pipeline v1.0
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-white mb-4 text-xs tracking-wider uppercase">Platform Features</h4>
            <ul className="space-y-2 text-xs">
              <li><button onClick={() => setActivePage('analyze')} className="hover:text-cyan-400 transition-colors">Engine Sound Diagnosis</button></li>
              <li><button onClick={() => setActivePage('models')} className="hover:text-cyan-400 transition-colors">Multi-Model Performance Table</button></li>
              <li><button onClick={() => setActivePage('models')} className="hover:text-cyan-400 transition-colors">Interactive Confusion Matrix</button></li>
              <li><button onClick={() => setActivePage('live')} className="hover:text-cyan-400 transition-colors">Live Microphone Recorder</button></li>
              <li><button onClick={() => setActivePage('history')} className="hover:text-cyan-400 transition-colors">PDF Report History</button></li>
            </ul>
          </div>

          {/* Supported Fault Categories */}
          <div>
            <h4 className="font-semibold text-white mb-4 text-xs tracking-wider uppercase">Supported Fault Classes</h4>
            <ul className="space-y-2 text-xs font-mono">
              <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> Normal Engine Resonance</li>
              <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span> Cylinder Misfire</li>
              <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-rose-400"></span> Bearing & Crankshaft Wear</li>
              <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span> Valve Train Tappet Noise</li>
              <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-purple-400"></span> Detonation Engine Knock</li>
            </ul>
          </div>

          {/* Tech Specs */}
          <div>
            <h4 className="font-semibold text-white mb-4 text-xs tracking-wider uppercase">Machine Learning Architecture</h4>
            <div className="space-y-2 text-xs">
              <div className="p-2 rounded bg-slate-900 border border-slate-800 font-mono text-[11px]">
                Random Forest • SVM • 1D CNN • 2D Mel-CNN • LSTM • Soft Voting Ensemble
              </div>
              <p className="text-[11px] text-slate-400">
                Librosa MFCC acoustic feature extraction & PyTorch deep neural models.
              </p>
            </div>
          </div>
        </div>

        {/* Disclaimer & Bottom Bar */}
        <div className="pt-8 border-t border-slate-900 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-slate-400">
          <p>© 2026 EngineSense AI. All rights reserved.</p>
          <p className="text-center italic max-w-xl text-[11px]">
            Disclaimer: EngineSense AI provides computer-assisted acoustic diagnostic evaluations. Results should be verified by a certified mechanic prior to major repairs.
          </p>
        </div>
      </div>
    </footer>
  );
}
