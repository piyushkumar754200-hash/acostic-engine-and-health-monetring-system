import json
import sqlite3
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.config import SQLITE_DB_PATH, DB_DIR, MONGODB_URI, MONGODB_DB_NAME

class Database:
    def __init__(self):
        self.use_mongo = False
        self.mongo_client = None
        
        # Check if MongoDB URI is set and motor can be imported
        if MONGODB_URI:
            try:
                import motor.motor_asyncio
                self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
                self.mongo_db = self.mongo_client[MONGODB_DB_NAME]
                self.use_mongo = True
                print("Connected to MongoDB database.")
            except Exception as e:
                print(f"MongoDB connection failed: {e}. Falling back to SQLite.")

        # Initialize SQLite fallback
        self._init_sqlite()

    def _init_sqlite(self):
        SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                vehicle_brand TEXT,
                vehicle_model TEXT,
                engine_type TEXT,
                predicted_condition TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT,
                processing_time REAL,
                full_result_json TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def save_analysis(self, analysis_data: Dict[str, Any]) -> str:
        analysis_id = analysis_data.get("analysis_id", str(uuid.uuid4()))
        filename = analysis_data.get("filename", "unknown.wav")
        timestamp = analysis_data.get("timestamp", datetime.now().isoformat())
        
        vehicle = analysis_data.get("vehicle_info", {})
        v_brand = vehicle.get("brand", "")
        v_model = vehicle.get("model", "")
        v_engine = vehicle.get("engine_type", "")

        pred_cond = analysis_data.get("predicted_condition", "Unknown")
        confidence = float(analysis_data.get("overall_confidence", 0.0))
        severity = analysis_data.get("severity", "Low")
        proc_time = float(analysis_data.get("processing_time_seconds", 0.0))

        json_str = json.dumps(analysis_data)

        if self.use_mongo:
            try:
                # Synchronous wrapper or store in mongo
                doc = dict(analysis_data)
                doc["_id"] = analysis_id
                # Fire and forget / store sync if loop present
            except Exception:
                pass

        # Save to SQLite
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO analysis_history 
            (id, filename, timestamp, vehicle_brand, vehicle_model, engine_type, predicted_condition, confidence, severity, processing_time, full_result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (analysis_id, filename, timestamp, v_brand, v_model, v_engine, pred_cond, confidence, severity, proc_time, json_str))
        conn.commit()
        conn.close()

        return analysis_id

    def get_history(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, timestamp, vehicle_brand, vehicle_model, engine_type, predicted_condition, confidence, processing_time FROM analysis_history ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "filename": r["filename"],
                "timestamp": r["timestamp"],
                "vehicle_brand": r["vehicle_brand"],
                "vehicle_model": r["vehicle_model"],
                "engine_type": r["engine_type"],
                "predicted_condition": r["predicted_condition"],
                "confidence": r["confidence"],
                "processing_time": r["processing_time"]
            })
        return results

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT full_result_json FROM analysis_history WHERE id = ?', (analysis_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row["full_result_json"])
        return None

    def delete_analysis(self, analysis_id: str) -> bool:
        conn = sqlite3.connect(str(SQLITE_DB_PATH))
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analysis_history WHERE id = ?', (analysis_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

db = Database()
