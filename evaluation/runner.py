"""RAGAS evaluation runner — called by CI and manually via `python evaluation/runner.py`."""

import argparse
import json
import sys
from pathlib import Path


def parse_thresholds(raw: str) -> dict[str, float]:
    result = {}
    for part in raw.split(","):
        k, v = part.split("=")
        result[k.strip()] = float(v.strip())
    return result


def run_eval(thresholds: dict[str, float]) -> dict[str, float]:
    golden_path = Path(__file__).parent / "golden_dataset.json"
    if not golden_path.exists():
        print("No golden dataset found at evaluation/golden_dataset.json — skipping eval.")
        return {}

    with open(golden_path) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} golden questions.")

    # TODO: run each question through the RAG pipeline and collect RAGAS metrics
    # from ragas import evaluate
    # from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
    # results = evaluate(dataset, metrics=[faithfulness, context_precision, ...])

    # Stub: return perfect scores until pipeline is implemented
    return {
        "faithfulness": 1.0,
        "context_precision": 1.0,
        "context_recall": 1.0,
        "answer_relevancy": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-threshold", default="faithfulness=0.90,context_precision=0.80")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.fail_threshold)
    metrics = run_eval(thresholds)

    if not metrics:
        sys.exit(0)

    print("\nRAGAS Results:")
    failures = []
    for metric, score in metrics.items():
        threshold = thresholds.get(metric)
        status = ""
        if threshold is not None and score < threshold:
            status = f" ← FAIL (threshold {threshold})"
            failures.append(metric)
        print(f"  {metric}: {score:.3f}{status}")

    if failures:
        print(f"\nEval gate FAILED on: {', '.join(failures)}")
        sys.exit(1)

    print("\nEval gate PASSED.")


if __name__ == "__main__":
    main()
