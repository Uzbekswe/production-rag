import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


@router.post("", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    start = time.monotonic()
    query_id = str(uuid.uuid4())
    logger.info("query_received", query_id=query_id, query=payload.query[:100])

    # TODO: replace stub with real agent invocation
    # result = await rag_agent.ainvoke(RAGState(query=payload.query, trace_id=query_id))

    latency_ms = int((time.monotonic() - start) * 1000)

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Query pipeline not yet implemented. Coming in Phase 2.",
    )
