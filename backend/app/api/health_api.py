from fastapi import APIRouter
from app.config import APP_NAME

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "online",
        "app_name": APP_NAME,
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "audio_processor": "ready",
            "models_engine": "ready"
        }
    }
