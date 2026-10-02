"""
Independent checks on the out-of-domain refusal test results (the corrected
version that measures true accuracy, not just gate pass/fail).

Run with: python tests/test_out_of_domain.py
Every check prints PASS or FAIL. Exit code is non-zero if anything failed.
"""

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

failures = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


def main():
    heldout_questions = json.load(open(BASE / "data" / "out_of_domain_heldout.json", encoding="utf-8"))
    calibration_questions = json.load(open(BASE / "data" / "out_of_domain_calibration.json", encoding="utf-8"))
    tuning_probes_source = (BASE / "src" / "tune_threshold.py").read_text(encoding="utf-8")

    check("held-out out-of-domain set has a real, substantial size (>= 30 questions)", len(heldout_questions) >= 30)

    # The whole point of this test is independence from what the threshold
    # was tuned on - if any held-out question also appears in the tuning
    # script's probe list, or in the calibration half, the test wouldn't be
    # a fair, held-out check.
    overlap_probes = [q for q in heldout_questions if q in tuning_probes_source]
    check("no held-out question also appears in the threshold-tuning probe list", len(overlap_probes) == 0)
    overlap_calibration = set(heldout_questions) & set(calibration_questions)
    check("no held-out question also appears in the calibration half", len(overlap_calibration) == 0)

    results = json.load(open(BASE / "data" / "processed" / "out_of_domain_results.json", encoding="utf-8"))
    in_domain = results["in_domain_test"]
    ood = results["out_of_domain"]
    test = json.load(open(BASE / "data/processed/test.json", encoding="utf-8"))

    check("in_domain_test n matches the actual test split size", in_domain["n"] == len(test))
    check("out_of_domain n matches the actual held-out question set size", ood["n"] == len(heldout_questions))

    # The four in-domain outcome categories must account for every test
    # question exactly once - no double-counting, nothing missed.
    total_categorized = (
        in_domain["true_positive"]
        + in_domain["confidently_wrong"]
        + in_domain["refused_wouldve_been_right"]
        + in_domain["refused_correctly"]
    )
    check("the four in-domain outcome categories sum to the full test set size", total_categorized == in_domain["n"])

    check("true_accuracy is a valid fraction between 0 and 1", 0.0 <= in_domain["true_accuracy"] <= 1.0)
    check(
        "true_accuracy matches true_positive / n",
        abs(in_domain["true_accuracy"] - in_domain["true_positive"] / in_domain["n"]) < 0.001,
    )
    true_acc_ci_low, true_acc_ci_high = in_domain["true_accuracy_95ci"]
    check(
        "true_accuracy 95% confidence interval brackets the point estimate (low <= rate <= high)",
        true_acc_ci_low <= in_domain["true_accuracy"] <= true_acc_ci_high,
    )
    check(
        "true_accuracy 95% confidence interval bounds are both valid fractions between 0 and 1",
        0.0 <= true_acc_ci_low <= true_acc_ci_high <= 1.0,
    )

    check("out-of-domain refusal_rate is a valid fraction between 0 and 1", 0.0 <= ood["refusal_rate"] <= 1.0)
    check(
        "out-of-domain wrongly_answered count matches (n - correctly_refused)",
        len(ood["wrongly_answered"]) == ood["n"] - ood["correctly_refused"],
    )
    ci_low, ci_high = ood["refusal_rate_95ci"]
    check("95% confidence interval brackets the point estimate (low <= rate <= high)", ci_low <= ood["refusal_rate"] <= ci_high)
    check("95% confidence interval bounds are both valid fractions between 0 and 1", 0.0 <= ci_low <= ci_high <= 1.0)

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL OUT-OF-DOMAIN TEST CHECKS PASSED")


if __name__ == "__main__":
    main()
