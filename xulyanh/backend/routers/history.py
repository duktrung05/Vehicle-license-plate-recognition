"""
History API routes
"""

from fastapi import APIRouter, Query, HTTPException
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2] / "src"))

from storage.database import PlateDatabase
from backend.schemas.response_schema import HistoryResponse, StatsResponse, PlateLogEntry

router = APIRouter(prefix="/history", tags=["History"])
db = PlateDatabase()


@router.get("/", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    source_type: str = Query(None),
    plate_text: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    """Get plate recognition history with optional filters."""
    logs = db.get_logs(
        limit=limit, offset=offset,
        source_type=source_type,
        plate_text=plate_text,
        start_date=start_date,
        end_date=end_date,
    )
    total = len(logs)
    return HistoryResponse(total=total, offset=offset, limit=limit, logs=logs)


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get overall recognition statistics."""
    stats = db.get_stats()
    return StatsResponse(**stats)


@router.get("/{log_id}", response_model=PlateLogEntry)
async def get_log(log_id: int):
    """Get a single log entry by ID."""
    log = db.get_log_by_id(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    return PlateLogEntry(**log)


@router.delete("/{log_id}")
async def delete_log(log_id: int):
    """Delete a single log entry."""
    success = db.delete_log(log_id)
    if not success:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"message": f"Log {log_id} deleted"}


@router.delete("/")
async def clear_all_logs():
    """Clear all log entries."""
    count = db.clear_all_logs()
    return {"message": f"Deleted {count} log entries"}
