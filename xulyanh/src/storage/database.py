import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parents[3] / "database" / "plate_logs.db"


class PlateDatabase:

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS plate_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_file TEXT,
                    plate_text TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    bbox TEXT,
                    province_code TEXT,
                    province_name TEXT,
                    plate_type TEXT,
                    is_valid INTEGER DEFAULT 0,
                    processing_time_ms REAL DEFAULT 0.0,
                    image_path TEXT,
                    crop_path TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset TEXT,
                    total_samples INTEGER,
                    exact_match_rate REAL,
                    char_accuracy REAL,
                    detection_precision REAL,
                    detection_recall REAL,
                    mean_confidence REAL,
                    mean_processing_time_ms REAL,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_timestamp ON plate_logs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_plate_text ON plate_logs(plate_text);
                CREATE INDEX IF NOT EXISTS idx_province ON plate_logs(province_code);
            """)
        logger.info(f"Database initialized at {self.db_path}")

    def log_detection(
        self,
        plate_text: str,
        confidence: float,
        source_type: str = "image",
        source_file: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        province_code: Optional[str] = None,
        province_name: Optional[str] = None,
        plate_type: Optional[str] = None,
        is_valid: bool = False,
        processing_time_ms: float = 0.0,
        image_path: Optional[str] = None,
        crop_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Log a single plate detection result."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO plate_logs
                   (timestamp, source_type, source_file, plate_text, confidence,
                    bbox, province_code, province_name, plate_type, is_valid,
                    processing_time_ms, image_path, crop_path, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    source_type,
                    source_file,
                    plate_text,
                    confidence,
                    json.dumps(bbox) if bbox else None,
                    province_code,
                    province_name,
                    plate_type,
                    int(is_valid),
                    processing_time_ms,
                    image_path,
                    crop_path,
                    json.dumps(metadata) if metadata else None,
                )
            )
            return cursor.lastrowid

    def get_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        source_type: Optional[str] = None,
        plate_text: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve plate recognition logs with optional filters."""
        query = "SELECT * FROM plate_logs WHERE 1=1"
        params = []

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if plate_text:
            query += " AND plate_text LIKE ?"
            params.append(f"%{plate_text}%")
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_log_by_id(self, log_id: int) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM plate_logs WHERE id = ?", (log_id,)).fetchone()
            return dict(row) if row else None

    def delete_log(self, log_id: int) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM plate_logs WHERE id = ?", (log_id,))
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM plate_logs").fetchone()[0]
            valid = conn.execute("SELECT COUNT(*) FROM plate_logs WHERE is_valid = 1").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM plate_logs").fetchone()[0] or 0.0
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = conn.execute(
                "SELECT COUNT(*) FROM plate_logs WHERE timestamp LIKE ?", (f"{today}%",)
            ).fetchone()[0]

            province_counts = conn.execute(
                """SELECT province_code, province_name, COUNT(*) as count
                   FROM plate_logs WHERE province_code IS NOT NULL
                   GROUP BY province_code ORDER BY count DESC LIMIT 10"""
            ).fetchall()

            return {
                "total_detections": total,
                "valid_plates": valid,
                "validation_rate": valid / total if total > 0 else 0,
                "avg_confidence": round(avg_conf, 3),
                "today_detections": today_count,
                "top_provinces": [dict(r) for r in province_counts],
            }

    def save_evaluation(self, metrics: dict, dataset: str = "", notes: str = "") -> int:
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO evaluation_runs
                   (timestamp, dataset, total_samples, exact_match_rate, char_accuracy,
                    detection_precision, detection_recall, mean_confidence,
                    mean_processing_time_ms, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    dataset,
                    metrics.get("total_samples", 0),
                    metrics.get("exact_match_rate", 0),
                    metrics.get("mean_char_accuracy", 0),
                    metrics.get("detection_precision", 0),
                    metrics.get("detection_recall", 0),
                    metrics.get("mean_confidence", 0),
                    metrics.get("mean_processing_time_ms", 0),
                    notes,
                )
            )
            return cursor.lastrowid

    def get_evaluations(self, limit: int = 20) -> List[Dict]:
        """Get past evaluation runs."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def clear_all_logs(self) -> int:
        """Delete all plate logs. Returns count deleted."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM plate_logs")
            return cursor.rowcount
