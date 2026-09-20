const API_BASE = `${import.meta.env.VITE_API_URL}/api`;

export const api = {
  // Analyze audio file with vehicle metadata
  async analyzeEngineSound(file, vehicleData) {
    const formData = new FormData();
    formData.append('file', file);
    if (vehicleData) {
      if (vehicleData.brand) formData.append('brand', vehicleData.brand);
      if (vehicleData.model) formData.append('model', vehicleData.model);
      if (vehicleData.year) formData.append('year', vehicleData.year);
      if (vehicleData.engine_type) formData.append('engine_type', vehicleData.engine_type);
      if (vehicleData.fuel_type) formData.append('fuel_type', vehicleData.fuel_type);
      if (vehicleData.mileage) formData.append('mileage', vehicleData.mileage);
      if (vehicleData.notes) formData.append('notes', vehicleData.notes);
    }

    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(err.detail || 'Failed to analyze engine sound');
    }

    return await response.json();
  },

  // Fetch models list and status
  async getModels() {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error('Failed to fetch models');
    return await res.json();
  },

  // Fetch performance metrics and confusion matrix
  async getModelPerformance() {
    const res = await fetch(`${API_BASE}/model-performance`);
    if (!res.ok) throw new Error('Failed to fetch model performance');
    return await res.json();
  },

  // Trigger training pipeline
  async trainModels(runTuning = false) {
    const res = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_tuning: runTuning }),
    });
    if (!res.ok) throw new Error('Failed to initiate training');
    return await res.json();
  },

  // Trigger hyperparameter tuning
  async tuneModels() {
    const res = await fetch(`${API_BASE}/tune`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to initiate hyperparameter tuning');
    return await res.json();
  },

  // Fetch history list
  async getHistory() {
    const res = await fetch(`${API_BASE}/history`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return await res.json();
  },

  // Fetch detail by ID
  async getHistoryDetail(id) {
    const res = await fetch(`${API_BASE}/history/${id}`);
    if (!res.ok) throw new Error('Failed to fetch analysis detail');
    return await res.json();
  },

  // Delete history item
  async deleteHistory(id) {
    const res = await fetch(`${API_BASE}/history/${id}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete history record');
    return await res.json();
  },

  // Get PDF report URL
  getReportUrl(id) {
    return `${API_BASE}/report/${id}`;
  },

  // Live microphone chunk diagnosis
  async analyzeLiveChunk(base64Audio) {
    const res = await fetch(`${API_BASE}/live/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_base64: base64Audio }),
    });
    if (!res.ok) throw new Error('Live chunk analysis failed');
    return await res.json();
  }
};
