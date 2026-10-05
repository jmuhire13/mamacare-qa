"""
Independent checks on the generation experiments which were tested
and deliberately not shipped. These checks guard the central finding: that
the LoRA generator doesn't outperform simply returning the top-1 retrieved
passage, and that most of its headline score comes from copying.

Run with: python tests/test_generation.py
Every check prints PASS or FAIL. Exit code is non-zero if anything failed.
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


def main():
    lora = json.load(open(PROCESSED / "lora_generation_results.json", encoding="utf-8"))
    retrieval_only = json.load(open(PROCESSED / "retrieval_only_generation_baseline.json", encoding="utf-8"))

    examples = lora["examples"]
    verbatim = sum(1 for e in examples if e["generated_answer"].strip() in [p.strip() for p in e["retrieved_passages"]])

    check("LoRA generation results cover all 43 test questions", len(examples) == 43)
    check(
        f"verbatim-copy rate recomputed from saved outputs is 40/43 ({verbatim}/43 found)",
        verbatim == 40,
    )
    check(
        "retrieval-only baseline (no generation) scores at least as high as the LoRA generator on ROUGE-L",
        retrieval_only["average_rougeL"] >= lora["average_rougeL"],
    )
    check(
        "retrieval-only baseline (no generation) scores at least as high as the LoRA generator on BERTScore F1",
        retrieval_only["average_bertscore_f1"] >= lora["average_bertscore_f1"],
    )

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL GENERATION CHECKS PASSED")


if __name__ == "__main__":
    main()
