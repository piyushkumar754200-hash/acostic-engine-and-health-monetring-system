import React, { useState, useRef, useEffect } from 'react';
import { Radio, Mic, MicOff, Activity, AlertTriangle, ShieldCheck, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

export default function LivePage() {
  const [isRecording, setIsRecording] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [micPermission, setMicPermission] = useState(null); // 'granted', 'denied', null
  const [errorMsg, setErrorMsg] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);

  const startRecording = async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setMicPermission('granted');
      setIsRecording(true);

      // Setup Web Audio API Analyser for Live Waveform Canvas
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 256;

      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      drawLiveWaveform();

      // Setup MediaRecorder
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = async () => {
          const base64Audio = reader.result;
          try {
            const res = await api.analyzeLiveChunk(base64Audio);
            setLiveResult(res);
          } catch (err) {
            setErrorMsg(err.message || 'Live chunk analysis failed.');
          }
        };
      };

      mediaRecorder.start();

      // Automatically capture & process 3.5 second audio slices
      const interval = setInterval(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
          audioChunksRef.current = [];
          mediaRecorderRef.current.start();
        } else {
          clearInterval(interval);
        }
      }, 3500);

    } catch (err) {
      setMicPermission('denied');
      setErrorMsg('Microphone access permission denied or unavailable in this browser environment.');
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (audioCtxRef.current) audioCtxRef.current.close();
  };

  const drawLiveWaveform = () => {
    if (!canvasRef.current || !analyserRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyserRef.current.getByteTimeDomainData(dataArray);

      ctx.fillStyle = '#090d16';
      ctx.fillRect(0, 0, width, height);

      ctx.lineWidth = 2;
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, '#00f0ff');
      gradient.addColorStop(0.5, '#3b82f6');
      gradient.addColorStop(1, '#10b981');
      ctx.strokeStyle = gradient;

      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);

        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    draw();
  };

  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (audioCtxRef.current) audioCtxRef.current.close();
    };
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
              <Radio className="w-8 h-8 text-cyan-400 animate-pulse" />
              Live Engine Microphone Analysis
            </h1>
          </div>
          <p className="text-slate-400 text-sm">
            Continuous real-time audio chunk processing via Web Audio API.
          </p>
        </div>

        <span className="px-3 py-1.5 rounded-full bg-amber-950/80 border border-amber-800/50 text-amber-400 text-xs font-mono inline-flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5" />
          Experimental Live Stream Mode
        </span>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-300 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          {errorMsg}
        </div>
      )}

      {/* Main Microphone Control Console */}
      <div className="glass-panel p-8 rounded-3xl border border-slate-800 text-center space-y-8">
        <div className="max-w-md mx-auto space-y-4">
          <div className="relative inline-block">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-24 h-24 rounded-full flex items-center justify-center transition-all transform active:scale-95 shadow-2xl ${
                isRecording
                  ? 'bg-rose-500 hover:bg-rose-600 text-white shadow-rose-500/40 animate-pulse scale-105'
                  : 'bg-gradient-to-tr from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-cyan-500/30'
              }`}
            >
              {isRecording ? <MicOff className="w-10 h-10" /> : <Mic className="w-10 h-10" />}
            </button>
          </div>

          <div>
            <p className="text-base font-bold text-white">
              {isRecording ? 'Listening & Analyzing Stream...' : 'Click Microphone to Start Live Diagnosis'}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Position device microphone near the engine compartment for optimal acoustic sensitivity.
            </p>
          </div>
        </div>

        {/* Live Canvas Visualizer */}
        <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 relative">
          <canvas ref={canvasRef} width={750} height={160} className="w-full h-40 rounded-xl" />
          {!isRecording && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-950/70 text-slate-500 text-xs font-mono">
              Microphone Inactive — Press Start Button Above
            </div>
          )}
        </div>
      </div>

      {/* Real-time Result Card */}
      {liveResult && (
        <div className={`glass-panel p-6 rounded-2xl border ${
          liveResult.predicted_class_name === 'normal'
            ? 'border-emerald-500/40 bg-emerald-950/20'
            : 'border-rose-500/40 bg-rose-950/20'
        } space-y-4 animate-fade-in`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-6 h-6 text-cyan-400" />
              <div>
                <p className="text-xs text-slate-400 uppercase font-mono">Live Acoustic Output</p>
                <h3 className="text-xl font-extrabold text-white">{liveResult.predicted_condition}</h3>
              </div>
            </div>
            <span className="text-lg font-bold font-mono text-cyan-400">
              {(liveResult.confidence * 100).toFixed(1)}% Conf
            </span>
          </div>

          <div className="grid grid-cols-3 gap-4 text-xs font-mono bg-slate-950/60 p-4 rounded-xl border border-slate-800">
            <div>
              <p className="text-slate-400">RMS Intensity</p>
              <p className="text-base font-bold text-cyan-400 mt-0.5">{liveResult.metrics.rms}</p>
            </div>
            <div>
              <p className="text-slate-400">ZCR Frequency</p>
              <p className="text-base font-bold text-blue-400 mt-0.5">{liveResult.metrics.zcr}</p>
            </div>
            <div>
              <p className="text-slate-400">Centroid Hz</p>
              <p className="text-base font-bold text-emerald-400 mt-0.5">{liveResult.metrics.spectral_centroid} Hz</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
