import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.database.db import db
from app.services.report_generator import ReportGenerator
from app.config import REPORTS_DIR

router = APIRouter()

@router.get("/report/{analysis_id}")
async def download_pdf_report(analysis_id: str):
    """Generates and returns PDF report for a given analysis ID."""
    data = db.get_analysis_by_id(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis record not found.")

    pdf_filename = f"EngineSense_Report_{analysis_id}.pdf"
    pdf_path = REPORTS_DIR / pdf_filename

    if not pdf_path.exists():
        ReportGenerator.generate_pdf_report(data, str(pdf_path))

    return FileResponse(
        path=str(pdf_path),
        filename=pdf_filename,
        media_type="application/pdf"
    )
