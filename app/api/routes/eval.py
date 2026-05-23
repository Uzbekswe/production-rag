"""
Evaluation API endpoints.

GET /api/v1/eval/run      — trigger a background RAGAS evaluation run
GET /api/v1/eval/history  — fetch last N eval runs from Postgres
"""

import subprocess
import sys
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories.eval_repo import list_eval_runs

router = APIRouter(prefix="/eval", tags=["evaluation"])
logger = get_logger(__name__)


def _run_eval_subprocess(run_id: str, category: str | None) -> None:
    """
    Launch evaluation/runner.py as a subprocess so it doesn't block
    the FastAPI event loop (RAGAS makes many blocking LLM calls).
    """
    cmd = [sys.executable, "evaluation/runner.py", "--save"]
    if category:
        cmd += ["--category", category]

    logger.info("eval_run_started", run_id=run_id, category=category or "all")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            logger.info("eval_run_complete", run_id=run_id)
        else:
            logger.warning("eval_run_failed", run_id=run_id, stderr=result.stderr[-500:])
    except subprocess.TimeoutExpired:
        logger.error("eval_run_timeout", run_id=run_id)
    except Exception as e:
        logger.error("eval_run_error", run_id=run_id, error=str(e))


@router.get("/run")
async def trigger_eval(
    category: str | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Trigger a RAGAS evaluation run in the background.
    Returns immediately — poll GET /eval/history for results.

    Optional query param: ?category=factual|analytical|multi_hop|adversarial
    """
    run_id = str(uuid4())
    background_tasks.add_task(_run_eval_subprocess, run_id, category)
    logger.info("eval_triggered", run_id=run_id, category=category or "all")
    return {
        "run_id": run_id,
        "status": "started",
        "category": category or "all",
        "message": "Evaluation running in background. Poll GET /api/v1/eval/history for results. Expect 5-15 minutes.",
    }


@router.get("/history")
async def eval_history(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the last N evaluation runs with their RAGAS scores."""
    runs = await list_eval_runs(db, limit=limit)
    return [
        {
            "id": str(r.id),
            "run_at": r.run_at.isoformat(),
            "faithfulness": r.faithfulness,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
            "answer_relevancy": r.answer_relevancy,
            "passed_ci": r.passed_ci,
            "question_count": r.question_count,
            "git_sha": r.git_sha,
        }
        for r in runs
    ]
