from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.eval_run import EvalRun, GoldenQuestion


async def create_eval_run(
    db: AsyncSession,
    *,
    git_sha: str | None = None,
    faithfulness: float | None = None,
    context_precision: float | None = None,
    context_recall: float | None = None,
    answer_relevancy: float | None = None,
    passed_ci: bool = False,
    question_count: int | None = None,
) -> EvalRun:
    run = EvalRun(
        git_sha=git_sha,
        faithfulness=faithfulness,
        context_precision=context_precision,
        context_recall=context_recall,
        answer_relevancy=answer_relevancy,
        passed_ci=passed_ci,
        question_count=question_count,
    )
    db.add(run)
    await db.flush()
    return run


async def list_eval_runs(db: AsyncSession, limit: int = 10) -> list[EvalRun]:
    result = await db.execute(
        select(EvalRun).order_by(EvalRun.run_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


async def create_golden_question(
    db: AsyncSession,
    *,
    question: str,
    ground_truth: str,
    relevant_chunks: list[str] | None = None,
    category: str | None = None,
) -> GoldenQuestion:
    gq = GoldenQuestion(
        question=question,
        ground_truth=ground_truth,
        relevant_chunks=relevant_chunks,
        category=category,
    )
    db.add(gq)
    await db.flush()
    return gq


async def get_golden_questions(
    db: AsyncSession,
    category: str | None = None,
) -> list[GoldenQuestion]:
    stmt = select(GoldenQuestion)
    if category:
        stmt = stmt.where(GoldenQuestion.category == category)
    result = await db.execute(stmt.order_by(GoldenQuestion.created_at))
    return list(result.scalars().all())


async def get_golden_question(db: AsyncSession, gq_id: UUID) -> GoldenQuestion | None:
    result = await db.execute(
        select(GoldenQuestion).where(GoldenQuestion.id == gq_id)
    )
    return result.scalar_one_or_none()
