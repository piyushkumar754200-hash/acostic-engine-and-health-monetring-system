import base64
import numpy as np
import librosa
import io
import soundfile as sf
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import LiveAudioRequest
from app.audio.processor import AudioProcessor
from app.config import FAULT_CLASSES, FAULT_MAP

router = APIRouter()
processor = AudioProcessor()

@router.post("/live/analyze")
async def analyze_live_audio_chunk(req: LiveAudioRequest):
    """
    Decodes base64 Web Audio API buffer and performs real-time acoustic evaluation.
    """
    try:
        # Decode base64 audio
        header, encoded = req.audio_base64.split(",", 1) if "," in req.audio_base64 else ("", req.audio_base64)
        audio_bytes = base64.b64decode(encoded)

        y_audio, sr = processor.load_audio(io.BytesIO(audio_bytes))
        feat = processor.extract_features(y_audio, sr)

        rms_val = feat["stats"]["rms_mean"]
        zcr_val = feat["stats"]["zcr_mean"]
        centroid = feat["stats"]["spectral_centroid_mean"]

        # Real-time condition evaluation
        if centroid > 3200 and zcr_val > 0.12:
            pred_id = "bearing_fault"
            conf = 0.88
        elif rms_val > 0.28:
            pred_id = "knocking"
            conf = 0.91
        elif zcr_val > 0.14:
            pred_id = "valve_fault"
            conf = 0.84
        elif rms_val < 0.07:
            pred_id = "misfire"
            conf = 0.82
        else:
            pred_id = "normal"
            conf = 0.95

        fault_info = FAULT_MAP.get(pred_id, FAULT_CLASSES[0])

        return {
            "status": "success",
            "predicted_condition": fault_info["name"],
            "predicted_class_name": fault_info["id"],
            "confidence": conf,
            "severity": fault_info["severity"],
            "metrics": {
                "rms": round(rms_val, 4),
                "zcr": round(zcr_val, 4),
                "spectral_centroid": round(centroid, 1)
            },
            "waveform_preview": feat["visualizations"]["waveform"][:100],
            "experimental_notice": "Live Audio Diagnosis (Experimental Real-Time Stream Mode)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process live stream audio chunk: {str(e)}")
