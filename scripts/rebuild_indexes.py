"""
Manual BM25 index rebuild from Postgres.

Run this any time the BM25 index needs to be recreated without restarting
the server — after a machine migration, after restoring from a pg_dump, or
after a manual Postgres restore.

Usage (from project root with venv active):
    .venv/bin/python scripts/rebuild_indexes.py

Requires:
    - Postgres running and reachable (docker compose up -d)
    - DATABASE_URL set in .env
    - At least one completed ingestion in the chunks table

The server auto-rebuilds on startup if the pickle is missing, so this
script is only needed for out-of-band rebuilds (e.g. during migration).
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from app.core.database import AsyncSessionLocal
    from app.repositories.chunk_repo import get_all_chunks_for_bm25
    from app.services.ingestion.bm25_indexer import BM25Index

    print("BM25 Index Rebuild")
    print("=" * 40)

    print("Connecting to Postgres...")
    t0 = time.perf_counter()
    async with AsyncSessionLocal() as db:
        chunks = await get_all_chunks_for_bm25(db)
    fetch_ms = (time.perf_counter() - t0) * 1000

    if not chunks:
        print("No chunks found in Postgres. Run an ingestion first.")
        sys.exit(1)

    print(f"Fetched {len(chunks):,} chunks in {fetch_ms:.0f}ms")
    print("Building BM25 index...")

    t1 = time.perf_counter()
    index = BM25Index()
    index.build(chunks)
    build_ms = (time.perf_counter() - t1) * 1000

    from app.services.ingestion.bm25_indexer import INDEX_PATH
    size_mb = INDEX_PATH.stat().st_size / 1_000_000

    print(f"Done in {build_ms:.0f}ms")
    print(f"Written → {INDEX_PATH}  ({size_mb:.1f} MB)")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
