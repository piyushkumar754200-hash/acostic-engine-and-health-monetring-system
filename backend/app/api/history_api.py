from fastapi import APIRouter, HTTPException
from app.database.db import db

router = APIRouter()

@router.get("/history")
async def get_analysis_history():
    """Returns list of past diagnostic analyses."""
    return db.get_history()

@router.get("/history/{analysis_id}")
async def get_analysis_detail(analysis_id: str):
    """Returns full JSON analysis result for a specific ID."""
    res = db.get_analysis_by_id(analysis_id)
    if not res:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
    return res

@router.delete("/history/{analysis_id}")
async def delete_analysis_record(analysis_id: str):
    """Deletes an analysis record."""
    success = db.delete_analysis(analysis_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
    return {"status": "success", "message": f"Deleted analysis record '{analysis_id}'"}
