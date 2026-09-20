import React, { useState, useEffect } from 'react';
import { History, Search, Download, Trash2, Eye, RefreshCw, X, Car, Disc, ShieldCheck } from 'lucide-react';
import { api } from '../services/api';

export default function HistoryPage() {
  const [historyItems, setHistoryItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [filterCondition, setFilterCondition] = useState('ALL');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const data = await api.getHistory();
      setHistoryItems(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete analysis record '${id}'?`)) return;
    try {
      await api.deleteHistory(id);
      setHistoryItems((prev) => prev.filter((item) => item.id !== id));
      if (selectedDetail?.analysis_id === id) setSelectedDetail(null);
    } catch (err) {
      alert(err.message);
    }
  };

  const handleViewDetail = async (id) => {
    try {
      const detail = await api.getHistoryDetail(id);
      setSelectedDetail(detail);
    } catch (err) {
      alert(err.message);
    }
  };

  const filteredItems = historyItems.filter((item) => {
    const matchSearch =
      item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.vehicle_brand.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.vehicle_model.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.id.toLowerCase().includes(searchTerm.toLowerCase());

    const matchFilter =
      filterCondition === 'ALL' ||
      (filterCondition === 'NORMAL' && item.predicted_condition.toLowerCase().includes('normal')) ||
      (filterCondition === 'ABNORMAL' && !item.predicted_condition.toLowerCase().includes('normal'));

    return matchSearch && matchFilter;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
            <History className="w-8 h-8 text-cyan-400" />
            Analysis History Log
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Persisted diagnostic history records and downloadable PDF engineering reports.
          </p>
        </div>

        <button
          onClick={fetchHistory}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 text-xs font-semibold transition-colors self-start"
        >
          <RefreshCw className={`w-4 h-4 text-cyan-400 ${loading ? 'animate-spin' : ''}`} />
          Refresh History
        </button>
      </div>

      {/* Search & Filter Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-4">
        <div className="sm:col-span-8 relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by analysis ID, file name, vehicle make or model..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2.5 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
          />
        </div>

        <div className="sm:col-span-4">
          <select
            value={filterCondition}
            onChange={(e) => setFilterCondition(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white focus:border-cyan-500 focus:outline-none"
          >
            <option value="ALL">Filter: All Conditions</option>
            <option value="NORMAL">Filter: Normal Engine Sounds Only</option>
            <option value="ABNORMAL">Filter: Fault Anomalies Only</option>
          </select>
        </div>
      </div>

      {/* History Data Table */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-slate-400 text-xs">Loading saved analysis records...</div>
        ) : filteredItems.length === 0 ? (
          <div className="p-16 text-center space-y-3">
            <Disc className="w-12 h-12 text-slate-600 mx-auto" />
            <p className="text-slate-300 font-semibold text-sm">No analysis history found</p>
            <p className="text-slate-500 text-xs max-w-sm mx-auto">
              {historyItems.length === 0
                ? "No sound recordings have been analyzed yet. Go to Sound Analysis to run your first diagnostic."
                : "No history records match your search filter."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-mono uppercase bg-slate-900/80">
                  <th className="p-3.5">ID / Date</th>
                  <th className="p-3.5">Vehicle</th>
                  <th className="p-3.5">Audio File</th>
                  <th className="p-3.5">Diagnosis Output</th>
                  <th className="p-3.5">Confidence</th>
                  <th className="p-3.5">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredItems.map((item) => {
                  const isNormal = item.predicted_condition.toLowerCase().includes('normal');
                  return (
                    <tr
                      key={item.id}
                      onClick={() => handleViewDetail(item.id)}
                      className="hover:bg-slate-900/50 cursor-pointer transition-colors"
                    >
                      <td className="p-3.5 font-mono">
                        <span className="font-bold text-white block">{item.id}</span>
                        <span className="text-[11px] text-slate-400">{item.timestamp}</span>
                      </td>
                      <td className="p-3.5">
                        <span className="font-semibold text-white block">{item.vehicle_brand} {item.vehicle_model}</span>
                        <span className="text-[11px] text-slate-400">{item.engine_type}</span>
                      </td>
                      <td className="p-3.5 font-mono text-cyan-400 truncate max-w-[160px]">
                        {item.filename}
                      </td>
                      <td className="p-3.5">
                        <span className={`px-2.5 py-1 rounded text-[11px] font-bold font-mono ${
                          isNormal
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : 'bg-rose-950 text-rose-400 border border-rose-800'
                        }`}>
                          {item.predicted_condition}
                        </span>
                      </td>
                      <td className="p-3.5 font-mono font-bold text-white">
                        {(item.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="p-3.5" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleViewDetail(item.id)}
                            title="View Detail"
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-cyan-400 border border-slate-800"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <a
                            href={api.getReportUrl(item.id)}
                            target="_blank"
                            rel="noreferrer"
                            title="Download PDF"
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-emerald-400 border border-slate-800"
                          >
                            <Download className="w-4 h-4" />
                          </a>
                          <button
                            onClick={(e) => handleDelete(item.id, e)}
                            title="Delete Record"
                            className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-rose-400 border border-slate-800"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Analysis Detail Modal */}
      {selectedDetail && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
          <div className="glass-panel max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 rounded-2xl border border-slate-800 space-y-6 relative text-xs">
            <button
              onClick={() => setSelectedDetail(null)}
              className="absolute top-4 right-4 p-2 rounded-lg bg-slate-900 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="border-b border-slate-800 pb-4">
              <span className="text-[11px] font-mono text-cyan-400 uppercase">Analysis ID: {selectedDetail.analysis_id}</span>
              <h3 className="text-xl font-bold text-white mt-1">{selectedDetail.predicted_condition}</h3>
              <p className="text-slate-400 mt-1">Recorded: {selectedDetail.timestamp}</p>
            </div>

            <div className="grid grid-cols-2 gap-4 font-mono bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div>
                <p className="text-slate-400">Vehicle Make / Model</p>
                <p className="text-white font-bold">{selectedDetail.vehicle_info.brand} {selectedDetail.vehicle_info.model}</p>
              </div>
              <div>
                <p className="text-slate-400">Overall Confidence</p>
                <p className="text-cyan-400 font-bold">{(selectedDetail.overall_confidence * 100).toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-slate-400">Engine Configuration</p>
                <p className="text-white">{selectedDetail.vehicle_info.engine_type}</p>
              </div>
              <div>
                <p className="text-slate-400">Severity Level</p>
                <p className="text-rose-400 font-bold">{selectedDetail.severity}</p>
              </div>
            </div>

            <div>
              <h4 className="font-bold text-white mb-2">Mechanical Recommendation</h4>
              <p className="text-slate-300 leading-relaxed bg-slate-900 p-3 rounded-lg border border-slate-800">
                {selectedDetail.recommendation}
              </p>
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t border-slate-800">
              <a
                href={api.getReportUrl(selectedDetail.analysis_id)}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download PDF Report
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
