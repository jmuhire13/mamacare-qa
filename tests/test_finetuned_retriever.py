"""
Independent checks on the fine-tuned retriever's saved model and results.

Run with: python tests/test_finetuned_retriever.py
Every check prints PASS or FAIL. Exit code is non-zero if anything failed.
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "finetuned-retriever"

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


def main():
    processed = BASE / "data" / "processed"
    val = json.load(open(processed / "val.json", encoding="utf-8"))
    test = json.load(open(processed / "test.json", encoding="utf-8"))
    results = json.load(open(processed / "baseline_results.json", encoding="utf-8"))

    check("finetuned-retriever model folder exists", MODEL_DIR.exists())
    check(
        "model folder contains actual weight files, not just config",
        any(MODEL_DIR.glob("*.safetensors")) or any(MODEL_DIR.glob("*.bin")),
    )

    check("finetuned_embeddings results are present in baseline_results.json", "finetuned_embeddings" in results)

    finetuned = results.get("finetuned_embeddings", {})
    for split_name, split_data in (("val", val), ("test", test)):
        metrics = finetuned.get(split_name, {})
        check(
            f"{split_name}: Top-1/Top-3/MRR are all valid fractions between 0 and 1",
            all(0.0 <= metrics.get(k, -1) <= 1.0 for k in ("top_1", "top_3", "mrr")),
        )
        check(
            f"{split_name}: Top-3 >= Top-1 (required by definition)",
            metrics.get("top_3", -1) >= metrics.get("top_1", -1),
        )
        check(
            f"{split_name}: n_queries matches the actual split size ({len(split_data)})",
            metrics.get("n_queries") == len(split_data),
        )

    # This is a regression flag, not proof that fine-tuning helped: the two
    # models' confidence intervals overlap heavily at this sample size, so
    # it's here to catch a real break (a corrupted or badly undertrained
    # model scoring far below the baseline), not to certify an improvement.
    pretrained_test_top1 = results.get("pretrained_embeddings", {}).get("test", {}).get("top_1")
    finetuned_test_top1 = finetuned.get("test", {}).get("top_1")
    if pretrained_test_top1 is not None and finetuned_test_top1 is not None:
        check(
            f"fine-tuned test Top-1 ({finetuned_test_top1}) is >= pretrained baseline ({pretrained_test_top1})",
            finetuned_test_top1 >= pretrained_test_top1,
        )

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL FINE-TUNED RETRIEVER CHECKS PASSED")


if __name__ == "__main__":
    main()
