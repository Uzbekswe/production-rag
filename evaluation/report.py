"""Formats RAGAS evaluation results as a terminal table and optional JSON file."""

import json
from pathlib import Path

THRESHOLDS = {
    "faithfulness":        (0.90, "CI GATE"),
    "context_recall":      (0.75, "target"),
    "factual_correctness": (0.85, "target"),
    "semantic_similarity": (0.80, "target"),
}

METRIC_LABELS = {
    "faithfulness":        "Faithfulness        ",
    "context_recall":      "Context Recall      ",
    "factual_correctness": "Factual Correctness ",
    "semantic_similarity": "Semantic Similarity ",
}


def write_report(
    metrics: dict,
    output_path: Path | None = None,
    git_sha: str | None = None,
) -> list[str]:
    """
    Print a formatted results table to stdout.
    Writes JSON to output_path if provided.
    Returns list of failed metric names (empty = passed).
    """
    width = 62
    print()
    print("=" * width)
    print("  RAGAS EVALUATION RESULTS")
    if git_sha:
        print(f"  commit: {git_sha[:12]}")
    print("=" * width)

    failures = []
    for key, label in METRIC_LABELS.items():
        score = metrics.get(key)
        if score is None:
            print(f"  {label}  N/A")
            continue

        threshold, gate_label = THRESHOLDS.get(key, (None, ""))
        if threshold and score < threshold:
            status = f"  ← FAIL [{gate_label}: {threshold:.2f}]"
            failures.append(key)
        elif threshold:
            status = f"  ✓  [{gate_label}: {threshold:.2f}]"
        else:
            status = ""

        print(f"  {label}  {score:.3f}{status}")

    print("-" * width)
    print(f"  Questions evaluated: {metrics.get('question_count', '?')}")
    category_counts = metrics.get("category_counts", {})
    if category_counts:
        for cat, count in sorted(category_counts.items()):
            print(f"    {cat:<18} {count} questions")

    print("=" * width)

    if failures:
        print(f"  EVAL GATE FAILED on: {', '.join(failures)}")
    else:
        print("  EVAL GATE PASSED")
    print("=" * width)
    print()

    if output_path:
        output_path.write_text(json.dumps(metrics, indent=2))
        print(f"  Results written → {output_path}")
        print()

    return failures
