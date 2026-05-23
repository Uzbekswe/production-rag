"""
Retrieval quality benchmark against the live hybrid+rerank pipeline.

Runs golden questions through the live /api/v1/query endpoint and measures:
  - Hit rate: fraction of answers containing key numbers from ground truth
  - Latency: per-question and average end-to-end response time
  - Citation depth: average number of citations returned

Results are broken down by category (factual / analytical / multi_hop / adversarial).

Usage:
  python scripts/benchmark.py                       # all 50 golden questions
  python scripts/benchmark.py --category factual    # subset
  python scripts/benchmark.py --limit 10            # first N questions
  python scripts/benchmark.py --output bench.json   # write JSON results
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH = Path(__file__).parent.parent / "evaluation" / "golden_dataset.json"
API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")


def load_golden(category: str | None = None, limit: int | None = None) -> list[dict]:
    with open(GOLDEN_PATH) as f:
        dataset = json.load(f)
    if category:
        dataset = [q for q in dataset if q["category"] == category]
    if limit:
        dataset = dataset[:limit]
    return dataset


def query_live(question: str, client: httpx.Client) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.post(
            f"{API_BASE}/api/v1/query",
            json={"query": question},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - t0
        return {
            "ok": True,
            "latency": latency,
            "answer": data.get("answer", ""),
            "num_citations": len(data.get("citations", [])),
            "from_cache": data.get("from_cache", False),
            "retrieval_method": data.get("retrieval_method", "?"),
        }
    except Exception as e:
        return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}


def keyword_hit(answer: str, ground_truth: str) -> bool:
    """Proxy for correctness: key numbers from ground_truth appear in answer."""
    numbers = re.findall(r"\$?[\d,]+\.?\d*%?", ground_truth)
    if not numbers:
        return True  # non-numeric ground truth: skip numeric check
    answer_norm = answer.lower().replace(",", "")
    return any(n.replace("$", "").replace(",", "") in answer_norm for n in numbers[:3])


def run_benchmark(golden: list[dict]) -> list[dict]:
    results = []
    print(f"\nBenchmarking {len(golden)} questions against {API_BASE}/api/v1/query\n")

    with httpx.Client() as client:
        for i, gq in enumerate(golden, 1):
            q_short = gq["question"][:65]
            print(f"  [{i:02d}/{len(golden)}] [{gq['category']:<12}] {q_short}...")
            r = query_live(gq["question"], client)
            hit = keyword_hit(r.get("answer", ""), gq["ground_truth"]) if r["ok"] else False
            cached = " (cached)" if r.get("from_cache") else ""
            status = f"{'✓' if hit else '✗'}  {r.get('latency', 0):.2f}s  {r.get('num_citations', 0)} citations{cached}"
            print(f"           → {status}")
            results.append({
                "question": gq["question"],
                "category": gq["category"],
                "ok": r["ok"],
                "latency": round(r.get("latency", 0), 3),
                "num_citations": r.get("num_citations", 0),
                "hit": hit,
                "from_cache": r.get("from_cache", False),
                "retrieval_method": r.get("retrieval_method", "?"),
                "error": r.get("error"),
            })
            time.sleep(0.3)

    return results


def print_report(results: list[dict]) -> None:
    width = 62
    ok_results = [r for r in results if r["ok"]]

    print()
    print("=" * width)
    print("  PIPELINE BENCHMARK RESULTS")
    print("=" * width)

    if ok_results:
        overall_hit = sum(1 for r in ok_results if r["hit"]) / len(ok_results)
        avg_latency = sum(r["latency"] for r in ok_results) / len(ok_results)
        avg_cit = sum(r["num_citations"] for r in ok_results) / len(ok_results)
        p95_latency = sorted(r["latency"] for r in ok_results)[int(len(ok_results) * 0.95)]
        cached_pct = sum(1 for r in ok_results if r["from_cache"]) / len(ok_results)
        retrieval_method = ok_results[0].get("retrieval_method", "?")

        print(f"  Strategy:          {retrieval_method}")
        print(f"  Questions:         {len(results)}  ({len(ok_results)} successful)")
        print(f"  Overall hit rate:  {overall_hit:.1%}")
        print(f"  Avg latency:       {avg_latency:.2f}s")
        print(f"  P95 latency:       {p95_latency:.2f}s")
        print(f"  Avg citations:     {avg_cit:.1f}")
        print(f"  Cache hit rate:    {cached_pct:.1%}")

    print()
    print(f"  {'Category':<18} {'Questions':>9} {'Hit Rate':>9} {'Avg Latency':>12} {'Avg Cit':>8}")
    print("-" * width)

    categories = sorted(set(r["category"] for r in results))
    by_cat: dict[str, list] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    for cat in categories:
        cat_ok = [r for r in by_cat[cat] if r["ok"]]
        if not cat_ok:
            print(f"  {cat:<18}  {'N/A':>9}")
            continue
        hit = sum(1 for r in cat_ok if r["hit"]) / len(cat_ok)
        lat = sum(r["latency"] for r in cat_ok) / len(cat_ok)
        cit = sum(r["num_citations"] for r in cat_ok) / len(cat_ok)
        print(f"  {cat:<18} {len(cat_ok):>9} {hit:>8.1%} {lat:>11.2f}s {cit:>7.1f}")

    print("=" * width)

    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\n  {len(failures)} failed requests:")
        for r in failures:
            print(f"    {r['question'][:50]}... → {r.get('error', '?')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark live RAG pipeline quality")
    parser.add_argument("--category", help="Filter: factual, analytical, multi_hop, adversarial")
    parser.add_argument("--limit", type=int, help="Only test first N questions")
    parser.add_argument("--output", help="Write JSON results to this path")
    args = parser.parse_args()

    golden = load_golden(category=args.category, limit=args.limit)
    if not golden:
        print("No questions matched filter.")
        sys.exit(0)

    results = run_benchmark(golden)
    print_report(results)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(results, indent=2))
        print(f"  Results written → {out}")


if __name__ == "__main__":
    main()
