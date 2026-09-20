import os
import json
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.config import MODELS_DIR, AVAILABLE_MODELS, FAULT_CLASSES
from app.schemas.schemas import TrainRequest
from train import train_pipeline

router = APIRouter()

@router.get("/models")
async def get_models_list():
    """Returns available models metadata."""
    metrics_file = MODELS_DIR / "metrics.json"
    is_trained = metrics_file.exists()
    
    models_status = []
    metrics_map = {}

    if is_trained:
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
                metrics_map = {m["model_id"]: m for m in data.get("models", [])}
        except Exception:
            pass

    for m in AVAILABLE_MODELS:
        m_id = m["id"]
        perf = metrics_map.get(m_id, {})
        models_status.append({
            "model_id": m_id,
            "name": m["name"],
            "type": m["type"],
            "input": m["input"],
            "is_trained": is_trained and m_id in metrics_map,
            "accuracy": perf.get("accuracy", 0.0),
            "f1_score": perf.get("f1_score", 0.0),
            "inference_time_ms": perf.get("inference_time_ms", 0.0)
        })

    return {
        "is_system_trained": is_trained,
        "models": models_status
    }

@router.get("/model-performance")
async def get_model_performance():
    """Returns real training evaluation metrics, precision/recall, and confusion matrix."""
    metrics_file = MODELS_DIR / "metrics.json"
    if not metrics_file.exists():
        return {
            "is_trained": False,
            "message": "Model not trained yet. Run training pipeline to generate metrics.",
            "dataset_summary": {"total_samples": 0, "classes": [fc["id"] for fc in FAULT_CLASSES]},
            "models": [],
            "confusion_matrix": [],
            "fault_classes": FAULT_CLASSES
        }

    with open(metrics_file, "r") as f:
        data = json.load(f)

    data["fault_classes"] = FAULT_CLASSES
    return data

@router.post("/train")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Triggers the training pipeline in background task."""
    def run_train():
        train_pipeline(run_tuning=req.run_tuning)

    background_tasks.add_task(run_train)
    return {
        "status": "training_started",
        "message": "Model training pipeline initiated successfully. Metrics will update upon completion."
    }

@router.post("/tune")
async def trigger_hyperparameter_tuning(background_tasks: BackgroundTasks):
    """Triggers hyperparameter tuning across Random Forest and SVM."""
    def run_tune():
        train_pipeline(run_tuning=True)

    background_tasks.add_task(run_tune)
    return {
        "status": "tuning_started",
        "message": "GridSearchCV hyperparameter tuning initiated."
    }
