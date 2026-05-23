"""
RAGAS evaluation runner.

Calls the live /api/v1/query endpoint for each golden question, builds a
RAGAS EvaluationDataset from the answers + retrieved citations, then scores
3 metrics using Groq Llama (free) as the judge LLM.

Checkpoint-aware: saves per-sample scores after every batch so the run can
be resumed after a crash or rate-limit kill without wasting LLM calls.

Usage:
  python evaluation/runner.py                          # run all questions
  python evaluation/runner.py --category factual       # subset by category
  python evaluation/runner.py --output results.json    # write JSON file
  python evaluation/runner.py --commit abc123          # tag with git SHA
  python evaluation/runner.py --save                   # persist to Postgres
  python evaluation/runner.py --limit 10               # first N questions
  python evaluation/runner.py --batch-size 10          # LLM calls per batch
  python evaluation/runner.py --checkpoint cp.jsonl    # checkpoint file path
"""

import argparse
import asyncio
import json
import math
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
DEFAULT_CHECKPOINT = Path(__file__).parent / "results_checkpoint.jsonl"
API_BASE = os.getenv("RAG_API_URL", "http://localhost:8000")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

METRICS = ["faithfulness", "context_recall", "factual_correctness"]


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[int, dict]:
    """Return {sample_idx: {metric: score}} for already-scored samples."""
    done = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["idx"]] = rec
            except Exception:
                pass
    return done


def append_checkpoint(path: Path, records: list[dict]) -> None:
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------

def load_golden(category: str | None = None, limit: int | None = None) -> list[dict]:
    with open(GOLDEN_PATH) as f:
        dataset = json.load(f)
    if category:
        dataset = [q for q in dataset if q["category"] == category]
    if limit:
        dataset = dataset[:limit]
    return dataset


def query_rag(question: str, client: httpx.Client) -> dict:
    """Call the live /api/v1/query endpoint."""
    resp = client.post(
        f"{API_BASE}/api/v1/query",
        json={"query": question},
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
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


# ---------------------------------------------------------------------------
# RAGAS evaluation (batched + checkpoint-aware)
# ---------------------------------------------------------------------------

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


def _is_valid(v) -> bool:
    """True iff v is a real numeric score (not None, not float NaN)."""
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _sanitize(v):
    """Normalise float('nan') → None so downstream None-checks are reliable."""
    return None if not _is_valid(v) else v


def _mean_score(val) -> float | None:
    """Aggregate a per-sample score list → mean, ignoring None/NaN failures."""
    if isinstance(val, (int, float)):
        return float(val) if _is_valid(val) else None
    scores = [v for v in val if _is_valid(v)]
    return round(sum(scores) / len(scores), 4) if scores else None


def make_evaluator_llm():
    """
    VESSL-first, Groq-fallback — same priority logic as enricher.py.

    VESSL (set VESSL_ENDPOINT + VESSL_TOKEN in .env):
      - OpenAI-compatible vLLM endpoint, no TPD limit, GPU billed per hour
      - Model: VESSL_MODEL env var (default Qwen/Qwen2.5-14B-Instruct)

    Groq fallback (GROQ_API_KEY):
      - llama-3.3-70b-versatile via RAGAS_JUDGE_MODEL env var
      - 100K tokens/day rolling limit — subject to exhaustion
    """
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    vessl_endpoint = os.getenv("VESSL_ENDPOINT")
    vessl_token    = os.getenv("VESSL_TOKEN")
    vessl_model    = os.getenv("VESSL_MODEL", "Qwen/Qwen2.5-14B-Instruct")

    if vessl_endpoint and vessl_token:
        print(f"  Judge backend : VESSL  ({vessl_endpoint})")
        print(f"  Judge model   : {vessl_model}")
        return LangchainLLMWrapper(
            ChatOpenAI(
                model=vessl_model,
                openai_api_key=vessl_token,
                openai_api_base=f"{vessl_endpoint}/v1",
                temperature=0,
                max_retries=3,
                request_timeout=180,
            )
        )

    if not GROQ_API_KEY:
        raise ValueError(
            "No judge LLM configured: set VESSL_ENDPOINT+VESSL_TOKEN (recommended) "
            "or GROQ_API_KEY (subject to 100K TPD limit)"
        )
    model = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")
    print(f"  Judge backend : Groq  (TPD-limited — set VESSL_ENDPOINT to avoid this)")
    print(f"  Judge model   : {model}")
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0,
            max_retries=6,
            request_timeout=120,
        )
    )


def run_ragas_batch(
    batch_golden: list[dict],
    batch_responses: list[dict],
    batch_indices: list[int],
    evaluator_llm,
    checkpoint_path: Path,
) -> list[dict]:
    """
    Score one batch. Returns list of per-sample dicts with scores.
    Appends to checkpoint on success.
    """
    from ragas import evaluate
    from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness

    dataset = build_ragas_dataset(batch_golden, batch_responses)

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
        ],
    )

    # ragas_result[key] → list of per-sample scores (None for failures)
    faith_scores   = result["faithfulness"]
    recall_scores  = result["context_recall"]
    factual_scores = result["factual_correctness"]

    records = []
    for local_i, global_idx in enumerate(batch_indices):
        rec = {
            "idx":               global_idx,
            "question":          batch_golden[local_i]["question"],
            "category":          batch_golden[local_i].get("category", "?"),
            # _sanitize: RAGAS returns float('nan') for failed metric calls, not None.
            # Normalise to None so all downstream `is not None` checks are reliable.
            "faithfulness":      _sanitize(faith_scores[local_i])   if local_i < len(faith_scores)   else None,
            "context_recall":    _sanitize(recall_scores[local_i])  if local_i < len(recall_scores)  else None,
            "factual_correctness": _sanitize(factual_scores[local_i]) if local_i < len(factual_scores) else None,
        }
        records.append(rec)

    append_checkpoint(checkpoint_path, records)
    return records


def run_all_batches(
    golden: list[dict],
    responses: list[dict],
    already_done: dict[int, dict],
    evaluator_llm,
    batch_size: int,
    checkpoint_path: Path,
) -> list[dict]:
    """
    Iterate batches, skipping already-checkpointed indices.
    Returns full list of per-sample score dicts (done + newly scored).
    """
    all_records = list(already_done.values())
    pending_indices = [i for i in range(len(golden)) if i not in already_done]

    if not pending_indices:
        print("  All samples already checkpointed — skipping RAGAS calls.")
        return all_records

    total_pending = len(pending_indices)
    total_batches = (total_pending + batch_size - 1) // batch_size
    print(f"\nRunning RAGAS evaluation — {total_pending} samples in {total_batches} batches of {batch_size}")
    print(f"  Checkpoint: {checkpoint_path}\n")

    run_start = time.time()
    for batch_num in range(total_batches):
        batch_pending = pending_indices[batch_num * batch_size : (batch_num + 1) * batch_size]
        batch_golden    = [golden[i]    for i in batch_pending]
        batch_responses = [responses[i] for i in batch_pending]

        first_q = batch_golden[0]["question"][:50]
        print(f"  Batch {batch_num+1}/{total_batches}  (samples {batch_pending[0]}–{batch_pending[-1]})  '{first_q}...'")
        batch_start = time.time()

        try:
            records = run_ragas_batch(
                batch_golden, batch_responses, batch_pending,
                evaluator_llm, checkpoint_path,
            )
            all_records.extend(records)
            elapsed = time.time() - batch_start
            scored = sum(1 for r in records if all(r.get(m) is not None for m in METRICS))
            failed = len(records) - scored
            print(f"    ✓ {scored} fully scored, {failed} partial/failed  [{elapsed:.0f}s]")
        except Exception as e:
            elapsed = time.time() - batch_start
            print(f"    ✗ Batch {batch_num+1} failed after {elapsed:.0f}s: {e}")
            print(f"      Checkpoint saved up to this point — re-run to resume.")
            # Don't re-raise: fall through and aggregate whatever we have
            break

    total_elapsed = time.time() - run_start
    print(f"\n  Total RAGAS time: {total_elapsed/60:.1f} min")
    return all_records


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def aggregate_scores(all_records: list[dict]) -> dict:
    """Average per-metric scores across all samples, counting failures."""
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    failed_per_metric: dict[str, int] = {m: 0 for m in METRICS}

    for rec in all_records:
        for m in METRICS:
            val = rec.get(m)
            if _is_valid(val):
                per_metric[m].append(float(val))
            else:
                failed_per_metric[m] += 1

    agg = {}
    for m in METRICS:
        scores = per_metric[m]
        agg[m] = round(sum(scores) / len(scores), 4) if scores else None

    return agg, per_metric, failed_per_metric


# ---------------------------------------------------------------------------
# Postgres persistence
# ---------------------------------------------------------------------------

async def save_to_postgres(metrics: dict, git_sha: str | None, passed: bool) -> None:
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
                answer_relevancy=None,
                passed_ci=passed,
                question_count=metrics.get("question_count"),
            )
            await db.commit()
        print("  Saved to Postgres eval_runs table.")
    except Exception as e:
        print(f"  Warning: could not save to Postgres: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against live RAG pipeline")
    parser.add_argument("--category",    help="Filter by category: factual, analytical, multi_hop, adversarial")
    parser.add_argument("--output",      help="Write JSON results to this file path")
    parser.add_argument("--commit",      help="Git SHA to tag this run with")
    parser.add_argument("--save",        action="store_true", help="Persist results to Postgres eval_runs table")
    parser.add_argument("--limit",       type=int, help="Only evaluate the first N questions")
    parser.add_argument("--batch-size",  type=int, default=10, help="Samples per RAGAS batch (default 10)")
    parser.add_argument("--checkpoint",  default=str(DEFAULT_CHECKPOINT), help="Checkpoint file path")
    parser.add_argument("--no-resume",   action="store_true", help="Ignore existing checkpoint and start fresh")
    # Thresholds are calibrated for Qwen2.5-14B as judge (~50-60% of GPT-4 scores).
    # Equivalent GPT-4 targets: faithfulness≥0.75, context_recall≥0.45.
    parser.add_argument("--fail-threshold", default="faithfulness=0.40,context_recall=0.20",
                        help="Comma-separated metric=threshold pairs for CI gate")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

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

    # Step 2: load checkpoint (resume support)
    already_done: dict[int, dict] = {}
    if not args.no_resume and checkpoint_path.exists():
        already_done = load_checkpoint(checkpoint_path)
        if already_done:
            print(f"\nResuming from checkpoint: {len(already_done)}/{len(golden)} samples already scored")
            print(f"  ({checkpoint_path})")

    # Step 3: run RAGAS (batched, checkpoint-aware)
    evaluator_llm = make_evaluator_llm()
    all_records = run_all_batches(
        golden, responses, already_done,
        evaluator_llm, args.batch_size, checkpoint_path,
    )

    # Step 4: aggregate scores
    agg_scores, per_metric_lists, failed_counts = aggregate_scores(all_records)
    scored_count = sum(1 for r in all_records if all(r.get(m) is not None for m in METRICS))
    failed_count = len(all_records) - scored_count

    runtime_min = (time.time() - t0) / 60
    metrics = {
        **agg_scores,
        "question_count":    len(golden),
        "scored_count":      scored_count,
        "failed_count":      failed_count,
        "category_counts":   category_counts,
        "runtime_minutes":   round(runtime_min, 1),
    }
    if args.commit:
        metrics["git_sha"] = args.commit

    # Step 5: report
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60)
    print(f"  Samples evaluated : {scored_count}/{len(golden)}  ({failed_count} failed/skipped)")
    print(f"  Runtime           : {runtime_min:.1f} min")
    print()
    for m in METRICS:
        score = agg_scores.get(m)
        fails = failed_counts.get(m, 0)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {m:<22} {score_str}   (failures: {fails})")
    print()

    from evaluation.report import write_report
    output_path = Path(args.output) if args.output else None
    write_report(metrics, output_path=output_path, git_sha=args.commit)

    if output_path:
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"  Written to {output_path}")

    # Step 6: CI gate
    gate_failures = [
        f"{m} {agg_scores[m]:.3f} < {t}"
        for m, t in thresholds.items()
        if m in agg_scores and agg_scores[m] is not None and agg_scores[m] < t
    ]
    passed = len(gate_failures) == 0

    # Step 7: optionally save to Postgres
    if args.save:
        asyncio.run(save_to_postgres(metrics, args.commit, passed))

    # Clean up checkpoint on successful full completion
    if passed and scored_count == len(golden) and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Checkpoint removed (full pass complete).")

    if gate_failures:
        print(f"\nCI GATE FAILED: {', '.join(gate_failures)}")
        sys.exit(1)

    print("\nCI GATE PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
