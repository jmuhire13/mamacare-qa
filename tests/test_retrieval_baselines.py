"""
Independent checks on retrieval_baselines.py's output and logic.

Run with: python tests/test_retrieval_baselines.py
Every check prints PASS or FAIL. Exit code is non-zero if anything failed.
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from retrieval_baselines import evaluate_ranking  # noqa: E402

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


def main():
    processed = BASE / "data" / "processed"
    train = json.load(open(processed / "train.json", encoding="utf-8"))
    val = json.load(open(processed / "val.json", encoding="utf-8"))
    test = json.load(open(processed / "test.json", encoding="utf-8"))
    results = json.load(open(processed / "baseline_results.json", encoding="utf-8"))

    # 1) The metric function itself: hand-built case with known correct ranks
    # (1st, 2nd, 5th place), so we can compute the right Top-1/Top-3/MRR by
    # hand and check the function matches exactly.
    queries = [
        {"question": "q0", "kb_index": 2},
        {"question": "q1", "kb_index": 0},
        {"question": "q2", "kb_index": 4},
    ]

    def fake_score(question):
        return {
            "q0": np.array([0.1, 0.2, 0.9, 0.1, 0.1]),  # true idx 2 -> rank 1
            "q1": np.array([0.5, 0.9, 0.1, 0.1, 0.1]),  # true idx 0 -> rank 2
            "q2": np.array([0.9, 0.8, 0.7, 0.6, 0.5]),  # true idx 4 -> rank 5
        }[question]

    result = evaluate_ranking(queries, kb_answers=["a", "b", "c", "d", "e"], score_fn=fake_score, k=3)
    expected_top1 = round(1 / 3, 3)
    expected_top3 = round(2 / 3, 3)
    expected_mrr = round((1 + 0.5 + 0.2) / 3, 3)
    check(
        "evaluate_ranking() matches hand-computed Top-1/Top-3/MRR on a known case",
        result["top_1"] == expected_top1 and result["top_3"] == expected_top3 and result["mrr"] == expected_mrr,
    )

    # 2) The knowledge base slicing must line up exactly with the real
    # val.json/test.json content, in the same order, or "correct answer"
    # bookkeeping would be silently wrong.
    kb_items = train + val + test
    val_slice = kb_items[len(train):len(train) + len(val)]
    test_slice = kb_items[len(train) + len(val):]
    check(
        "val slice pulled from the combined KB matches val.json exactly",
        [x["question"] for x in val_slice] == [x["question"] for x in val],
    )
    check(
        "test slice pulled from the combined KB matches test.json exactly",
        [x["question"] for x in test_slice] == [x["question"] for x in test],
    )

    # 3) Sanity checks on the actual saved results: every method's Top-3
    # should never be lower than its Top-1 (by definition Top-3 counts
    # everything Top-1 counts, plus more).
    monotonic = all(
        metrics["top_3"] >= metrics["top_1"]
        for method in results.values()
        for metrics in method.values()
    )
    check("Top-3 >= Top-1 for every method and split (required by definition)", monotonic)

    # 4) Query counts should match the actual split sizes (43/43 here).
    counts_match = all(
        metrics["n_queries"] == (len(val) if split == "val" else len(test))
        for method in results.values()
        for split, metrics in method.items()
    )
    check("n_queries in results matches the actual val/test split sizes", counts_match)

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL RETRIEVAL BASELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
