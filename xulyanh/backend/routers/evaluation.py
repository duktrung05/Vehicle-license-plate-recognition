"""
Evaluation API routes
"""

from fastapi import APIRouter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2] / "src"))

from evaluation.metrics import plate_accuracy
from storage.database import PlateDatabase
from backend.schemas.response_schema import EvaluationRequest, EvaluationResponse

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
db = PlateDatabase()


@router.post("/", response_model=EvaluationResponse)
async def run_evaluation(req: EvaluationRequest):
    """Run evaluation on a set of predictions vs ground truths."""
    metrics = plate_accuracy(req.predictions, req.ground_truths)

    eval_id = db.save_evaluation(
        metrics=metrics,
        dataset=req.dataset_name,
        notes=req.notes,
    )

    return EvaluationResponse(
        exact_match_rate=metrics.get("exact_match_rate", 0),
        mean_char_accuracy=metrics.get("mean_char_accuracy", 0),
        mean_edit_distance=metrics.get("mean_edit_distance", 0),
        total_samples=metrics.get("total_samples", 0),
        exact_matches=metrics.get("exact_matches", 0),
        evaluation_id=eval_id,
    )


@router.get("/history")
async def get_evaluation_history(limit: int = 20):
    """Get past evaluation runs."""
    return db.get_evaluations(limit=limit)
