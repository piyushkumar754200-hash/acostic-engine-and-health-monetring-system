from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class VehicleInfo(BaseModel):
    brand: Optional[str] = "Generic / Unknown"
    model: Optional[str] = "Standard V6/I4"
    year: Optional[int] = 2020
    engine_type: Optional[str] = "2.0L Inline-4 Turbo"
    fuel_type: Optional[str] = "Gasoline"
    mileage: Optional[int] = 45000
    notes: Optional[str] = ""

class FeatureStats(BaseModel):
    rms_mean: float
    rms_std: float
    zcr_mean: float
    zcr_std: float
    spectral_centroid_mean: float
    spectral_bandwidth_mean: float
    spectral_rolloff_mean: float
    chroma_mean: float
    spectral_contrast_mean: float
    mfcc_means: List[float]

class AudioAnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    timestamp: str
    duration_seconds: float
    sample_rate: int
    channels: int
    vehicle_info: VehicleInfo
    predicted_condition: str
    predicted_class_name: str
    overall_confidence: float
    severity: str
    recommendation: str
    processing_time_seconds: float
    model_predictions: Dict[str, Dict[str, Any]]
    ensemble_results: Dict[str, Any]
    feature_stats: FeatureStats
    visualizations: Dict[str, Any]
    explainable_ai: Dict[str, Any]
    is_demo: bool = False

class AnalysisHistoryItem(BaseModel):
    id: str
    timestamp: str
    filename: str
    vehicle_brand: str
    vehicle_model: str
    engine_type: str
    predicted_condition: str
    confidence: float
    processing_time: float

class ModelPerformanceItem(BaseModel):
    model_id: str
    model_name: str
    type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    status: str
    last_trained: str

class ModelPerformanceResponse(BaseModel):
    is_trained: bool
    dataset_summary: Dict[str, Any]
    models: List[ModelPerformanceItem]
    confusion_matrix: Dict[str, Any]
    fault_classes: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 16
    test_size: Optional[float] = 0.2
    run_tuning: Optional[bool] = False

class LiveAudioRequest(BaseModel):
    audio_base64: str
    vehicle_info: Optional[VehicleInfo] = None
