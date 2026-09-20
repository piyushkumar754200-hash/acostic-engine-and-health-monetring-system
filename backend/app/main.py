import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure backend package is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import APP_NAME, API_PREFIX, UPLOADS_DIR, REPORTS_DIR
from app.api.analyze import router as analyze_router
from app.api.models_api import router as models_router
from app.api.history_api import router as history_router
from app.api.report_api import router as report_router
from app.api.health_api import router as health_router
from app.api.live_api import router as live_router

app = FastAPI(
    title=APP_NAME,
    description="Full-stack AI acoustic diagnostic system for intelligent vehicle and engine health monitoring.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for direct reports or uploads viewing if needed
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# Include Routers
app.include_router(health_router, prefix=API_PREFIX, tags=["Health"])
app.include_router(analyze_router, prefix=API_PREFIX, tags=["Analysis"])
app.include_router(models_router, prefix=API_PREFIX, tags=["Models"])
app.include_router(history_router, prefix=API_PREFIX, tags=["History"])
app.include_router(report_router, prefix=API_PREFIX, tags=["Reports"])
app.include_router(live_router, prefix=API_PREFIX, tags=["Live Audio"])

@app.get("/")
async def root():
    return {
        "app": APP_NAME,
        "status": "online",
        "docs_url": "/docs",
        "api_prefix": API_PREFIX
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
