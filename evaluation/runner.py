"""
RAGAS evaluation runner.

Calls the live /api/v1/query endpoint for each golden question, builds a
RAGAS EvaluationDataset from the answers + retrieved citations, then scores
4 metrics using Groq Llama (free) as the judge LLM.

Usage:
  python evaluation/runner.py                          # run all questions
  python evaluation/runner.py --category factual       # subset by category
  python evaluation/runner.py --output results.json    # write JSON file
  python evaluation/runner.py --commit abc123          # tag with git SHA
  python evaluation/runner.py --save                   # persist to Postgres
  python evaluation/runner.py --limit 10               # first N questions
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import httpx

# Load .env so the runner works as a standalone script (server uses pydantic-settings
# to load .env automatically, but standalone scripts need explicit loading)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_golden(category: str | None = None, limit: int | None = None) -> list[dict]:
    with open(GOLDEN_PATH) as f:
        dataset = json.load(f)
    if category:
        dataset = [q for q in dataset if q["category"] == category]
    if limit:
        dataset = dataset[:limit]
    return dataset


def query_rag(question: str, client: httpx.Client) -> dict:
    """
    Call the live /api/v1/query endpoint.
    Returns {"answer": str, "contexts": list[str]} or raises on failure.
    """
    resp = client.post(
        f"{API_BASE}/api/v1/query",
        json={"query": question},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()

    # Citations from our API become retrieved_contexts for RAGAS.
    # We include the filename so the judge knows which document was cited.
    contexts = [
        f"[{c['filename']}] {c['cited_text']}"
        for c in data.get("citations", [])
    ]
    return {
        "answer": data.get("answer", ""),
        "contexts": contexts,
        "from_cache": data.get("from_cache", False),
    }


def collect_responses(golden: list[dict]) -> list[dict]:
    responses = []
    print(f"\nQuerying live API for {len(golden)} questions...")
    print(f"  API: {API_BASE}/api/v1/query")
    print()

    with httpx.Client() as client:
        for i, gq in enumerate(golden, 1):
            cat = gq.get("category", "?")
            q_short = gq["question"][:65]
            print(f"  [{i:02d}/{len(golden)}] [{cat:<12}] {q_short}...")
            try:
                resp = query_rag(gq["question"], client)
                cache_label = "(cached)" if resp["from_cache"] else ""
                ctx_count = len(resp["contexts"])
                print(f"          → {ctx_count} chunks  answer: {resp['answer'][:60]}... {cache_label}")
                responses.append(resp)
            except Exception as e:
                print(f"          → ERROR: {e}")
                responses.append({"answer": "", "contexts": []})
            time.sleep(0.3)

    return responses


def build_ragas_dataset(golden: list[dict], responses: list[dict]):
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = []
    for gq, resp in zip(golden, responses):
        samples.append(SingleTurnSample(
            user_input=gq["question"],
            retrieved_contexts=resp["contexts"] or ["[No context retrieved]"],
            response=resp["answer"] or "[No answer generated]",
            reference=gq["ground_truth"],
        ))
    return EvaluationDataset(samples=samples)


def run_ragas(dataset, evaluator_llm) -> dict:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextRecall,
        FactualCorrectness,
    )

    print("\nRunning RAGAS evaluation (Groq Llama judge)...")
    print("  This takes 2-5 minutes — one LLM call per question per metric.\n")

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
        ],
    )
    return result


def make_evaluator_llm():
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set — needed for RAGAS judge LLM")

    return LangchainLLMWrapper(
        ChatOpenAI(
            model="llama-3.1-8b-instant",
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0,
        )
    )


async def save_to_postgres(metrics: dict, git_sha: str | None, passed: bool) -> None:
    """Persist results to the eval_runs table."""
    try:
        from app.core.database import AsyncSessionLocal
        from app.repositories.eval_repo import create_eval_run
        async with AsyncSessionLocal() as db:
            await create_eval_run(
                db,
                git_sha=git_sha,
                faithfulness=metrics.get("faithfulness"),
                context_precision=metrics.get("factual_correctness"),
                context_recall=metrics.get("context_recall"),
                answer_relevancy=metrics.get("semantic_similarity"),
                passed_ci=passed,
                question_count=metrics.get("question_count"),
            )
            await db.commit()
        print("  Saved to Postgres eval_runs table.")
    except Exception as e:
        print(f"  Warning: could not save to Postgres: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against live RAG pipeline")
    parser.add_argument("--category", help="Filter by category: factual, analytical, multi_hop, adversarial")
    parser.add_argument("--output", help="Write JSON results to this file path")
    parser.add_argument("--commit", help="Git SHA to tag this run with")
    parser.add_argument("--save", action="store_true", help="Persist results to Postgres eval_runs table")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions")
    parser.add_argument("--fail-threshold", default="faithfulness=0.90,context_recall=0.75",
                        help="Comma-separated metric=threshold pairs for CI gate")
    args = parser.parse_args()

    # Parse CI gate thresholds
    thresholds = {}
    for part in args.fail_threshold.split(","):
        k, v = part.split("=")
        thresholds[k.strip()] = float(v.strip())

    # Load golden dataset
    golden = load_golden(category=args.category, limit=args.limit)
    if not golden:
        print("No questions matched the filter. Check --category value.")
        sys.exit(0)

    category_counts = dict(Counter(q["category"] for q in golden))
    print(f"\nLoaded {len(golden)} golden questions")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:<18} {count}")

    # Step 1: collect responses from live API
    responses = collect_responses(golden)

    # Step 2: build RAGAS dataset
    dataset = build_ragas_dataset(golden, responses)

    # Step 3: run RAGAS with Groq judge
    evaluator_llm = make_evaluator_llm()
    ragas_result = run_ragas(dataset, evaluator_llm)

    # Step 4: extract scores
    # ragas_result[key] returns a list of per-sample scores in RAGAS 0.2.x;
    # we aggregate to the mean (ignoring None values from timeout/parse failures)
    def _mean_score(val) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        scores = [v for v in val if v is not None]
        return sum(scores) / len(scores) if scores else 0.0

    metrics = {
        "faithfulness":        _mean_score(ragas_result["faithfulness"]),
        "context_recall":      _mean_score(ragas_result["context_recall"]),
        "factual_correctness": _mean_score(ragas_result["factual_correctness"]),
        "question_count":      len(golden),
        "category_counts":     category_counts,
    }
    if args.commit:
        metrics["git_sha"] = args.commit

    # Step 5: report
    from evaluation.report import write_report
    output_path = Path(args.output) if args.output else None
    failures = write_report(metrics, output_path=output_path, git_sha=args.commit)

    # Step 6: check CI gate thresholds
    gate_failures = [
        f"{m} {metrics[m]:.3f} < {t}"
        for m, t in thresholds.items()
        if m in metrics and metrics[m] < t
    ]
    passed = len(gate_failures) == 0

    # Step 7: optionally save to Postgres
    if args.save:
        asyncio.run(save_to_postgres(metrics, args.commit, passed))

    if gate_failures:
        print(f"CI GATE FAILED: {', '.join(gate_failures)}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
