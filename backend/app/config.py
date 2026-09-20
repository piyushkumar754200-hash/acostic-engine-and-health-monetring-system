import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"

UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
DB_DIR = DATA_DIR / "db"

# Create directories if not existing
for directory in [UPLOADS_DIR, MODELS_DIR, REPORTS_DIR, DB_DIR, DATASET_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Application Settings
APP_NAME = "EngineSense AI — Intelligent Vehicle Sound Diagnosis"
API_PREFIX = "/api"
MAX_UPLOAD_SIZE_MB = 50
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# Database settings
SQLITE_DB_PATH = DB_DIR / "enginesense.db"
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "enginesense_db")

# Fault Classes Configuration (Configurable per dataset)
FAULT_CLASSES = [
    {
        "id": "normal",
        "name": "Normal Engine Sound",
        "severity": "Low",
        "description": "Engine sounds operating within optimal acoustic parameters with harmonic rhythms and balanced combustion resonance.",
        "recommendation": "Routine maintenance schedule recommended. No immediate mechanical intervention required."
    },
    {
        "id": "misfire",
        "name": "Cylinder Misfire",
        "severity": "Medium-High",
        "description": "Irregular rhythm and sharp transient dips caused by incomplete combustion in one or more cylinders.",
        "recommendation": "Inspect spark plugs, ignition coils, fuel injectors, and compression ratio across cylinders."
    },
    {
        "id": "bearing_fault",
        "name": "Bearing Wear / Shaft Wear",
        "severity": "High",
        "description": "High-frequency friction continuous whining or grinding noise indicating worn crankshaft or connecting rod bearings.",
        "recommendation": "Urgent oil pressure test & bearing tolerance inspection. Continued operation risks catastrophic engine failure."
    },
    {
        "id": "valve_fault",
        "name": "Valve Train Tappet Noise",
        "severity": "Medium",
        "description": "Rapid metallic tapping at half the crankshaft rotational frequency due to excessive valve clearance or sticky lifters.",
        "recommendation": "Check valve clearance lash, hydraulic lifters, and engine oil viscosity."
    },
    {
        "id": "knocking",
        "name": "Engine Knock / Detonation",
        "severity": "Critical",
        "description": "Heavy metallic pinging or knocking sound caused by abnormal premature auto-ignition in the combustion chamber.",
        "recommendation": "Check fuel octane rating, ignition timing, knock sensors, and carbon deposit buildup in cylinder head."
    }
]

FAULT_CLASS_NAMES = [fc["id"] for fc in FAULT_CLASSES]
FAULT_MAP = {fc["id"]: fc for fc in FAULT_CLASSES}

# Model List Configuration
AVAILABLE_MODELS = [
    {"id": "rf", "name": "Random Forest", "type": "Traditional ML", "input": "Standardized Acoustic Features"},
    {"id": "svm", "name": "Support Vector Machine (SVM)", "type": "Traditional ML", "input": "Standardized Acoustic Features"},
    {"id": "cnn_1d", "name": "1D Convolutional Neural Network", "type": "Deep Learning", "input": "1D Sequential Feature Arrays"},
    {"id": "cnn_2d", "name": "2D Mel-Spectrogram CNN", "type": "Deep Learning", "input": "2D Mel Spectrogram Heatmap"},
    {"id": "lstm", "name": "LSTM / Recurrent Neural Net", "type": "Deep Learning", "input": "Temporal Feature Sequences"},
    {"id": "ensemble", "name": "Weighted Soft Voting Ensemble", "type": "Ensemble", "input": "Multi-Model Output Probability Vectors"}
]

# Feature Extraction Parameters
AUDIO_SAMPLE_RATE = 22050
AUDIO_DURATION = 5.0  # seconds segment standard
N_MFCC = 20
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
