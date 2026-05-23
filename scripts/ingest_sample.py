"""
Batch-ingest all downloaded SEC 10-K filings into the running RAG server.

Usage:
    python scripts/ingest_sample.py [--base-url http://localhost:8000]

Reads every PDF/HTM file under data/sample_docs/, POSTs each one to
POST /api/v1/ingest, then polls until all jobs complete (or fail).
Prints a summary table at the end.
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx

SAMPLE_DIR = Path("data/sample_docs")
POLL_INTERVAL = 5   # seconds between status polls
POLL_TIMEOUT = 3600  # give up after 1 hour per document


async def ingest_file(client: httpx.AsyncClient, base_url: str, path: Path) -> dict:
    with open(path, "rb") as f:
        resp = await client.post(
            f"{base_url}/api/v1/ingest",
            files={"file": (path.name, f, "application/octet-stream")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()


async def poll_until_done(
    client: httpx.AsyncClient, base_url: str, job_id: str, filename: str
) -> dict:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = await client.get(f"{base_url}/api/v1/ingest/{job_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        if status in ("ready", "failed"):
            return data
        print(f"  [{filename}] status={status} … waiting {POLL_INTERVAL}s")
        await asyncio.sleep(POLL_INTERVAL)
    return {"status": "timeout", "job_id": job_id, "chunk_count": None, "error_msg": "timed out"}


async def main(base_url: str) -> None:
    files = sorted(SAMPLE_DIR.rglob("*.pdf")) + sorted(SAMPLE_DIR.rglob("*.htm"))
    if not files:
        print(f"No files found in {SAMPLE_DIR}. Run scripts/download_sec_filings.py first.")
        return

    print(f"Found {len(files)} files to ingest against {base_url}")
    print("-" * 60)

    results = []
    async with httpx.AsyncClient(base_url=base_url) as client:
        # Submit all jobs first (non-blocking)
        jobs: list[tuple[Path, dict]] = []
        for path in files:
            try:
                job = await ingest_file(client, base_url, path)
                print(f"  Queued: {path.name}  →  job_id={job['job_id']}")
                jobs.append((path, job))
            except Exception as e:
                print(f"  FAILED to queue {path.name}: {e}")
                results.append({"file": path.name, "status": "queue_error", "chunks": None})

        # Poll all jobs concurrently
        print(f"\nPolling {len(jobs)} jobs …")
        poll_tasks = [
            poll_until_done(client, base_url, job["job_id"], path.name)
            for path, job in jobs
        ]
        statuses = await asyncio.gather(*poll_tasks, return_exceptions=True)

        for (path, job), status_data in zip(jobs, statuses):
            if isinstance(status_data, Exception):
                results.append({"file": path.name, "status": "poll_error", "chunks": None})
            else:
                results.append({
                    "file": path.name,
                    "status": status_data["status"],
                    "chunks": status_data.get("chunk_count"),
                    "error": status_data.get("error_msg"),
                })

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    total_chunks = 0
    for r in results:
        icon = "✓" if r["status"] == "ready" else "✗"
        chunks = r.get("chunks") or 0
        total_chunks += chunks
        print(f"  {icon} {r['file']:40s}  {r['status']:10s}  {chunks:>5} chunks")
        if r.get("error"):
            print(f"      error: {r['error']}")

    ready = sum(1 for r in results if r["status"] == "ready")
    print(f"\nReady: {ready}/{len(results)}  |  Total chunks: {total_chunks:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    asyncio.run(main(args.base_url))
