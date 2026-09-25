"""
Independent checks on data_prep.py's output. These don't just re-run the
pipeline and trust its own printout - they check the actual files it wrote
against rules that must hold if the cleaning and splitting worked correctly.

Run with: python tests/test_data_prep.py
Every check prints PASS or FAIL. Exit code is non-zero if anything failed.
"""

import json
import re
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


def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s)


def is_leakage(question):
    qwords = {
        "what", "why", "how", "when", "where", "who", "which", "is", "are",
        "can", "could", "does", "do", "did", "will", "would", "should",
        "am", "was", "were", "has", "have",
    }
    if "?" in question:
        return False
    first_word = question.strip().lower().split()[0] if question.strip() else ""
    return first_word not in qwords


def main():
    raw = json.load(open(BASE / "data/raw/mother_question_and_answer_pairs_data.json", encoding="utf-8"))
    train = json.load(open(PROCESSED / "train.json", encoding="utf-8"))
    val = json.load(open(PROCESSED / "val.json", encoding="utf-8"))
    test = json.load(open(PROCESSED / "test.json", encoding="utf-8"))
    report = json.load(open(PROCESSED / "cleaning_report.json", encoding="utf-8"))

    check("raw file length matches cleaning_report.json", len(raw) == report["raw_count"])

    check(
        "train/val/test file sizes match cleaning_report.json",
        len(train) == report["train_count"]
        and len(val) == report["val_count"]
        and len(test) == report["test_count"],
    )

    check(
        "train + val + test rows sum to the reported clean count",
        len(train) + len(val) + len(test) == report["clean_count"],
    )

    def key(item):
        return (norm(item["question"]), norm(item["answer"]))

    train_keys, val_keys, test_keys = (
        {key(x) for x in train}, {key(x) for x in val}, {key(x) for x in test}
    )
    check(
        "no question+answer pair appears in more than one split",
        not (train_keys & val_keys) and not (train_keys & test_keys) and not (val_keys & test_keys),
    )

    all_clean = train + val + test
    q_norms = [norm(x["question"]) for x in all_clean]
    check("no duplicate questions remain anywhere in the cleaned data", len(q_norms) == len(set(q_norms)))

    check(
        "no leakage-pattern questions survived into the output",
        not any(is_leakage(x["question"]) for x in all_clean),
    )

    check(
        "no cut-off-pattern answers survived into the output",
        not any(x["answer"].strip().endswith(":") for x in all_clean),
    )

    # Every reported near-duplicate group must land entirely within one split.
    def which_split(question_text):
        for name, split in (("train", train), ("val", val), ("test", test)):
            if any(x["question"] == question_text for x in split):
                return name
        return None

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for link in report["near_duplicate_links"]:
        union(link["question_a"], link["question_b"])

    groups = {}
    for q in parent:
        groups.setdefault(find(q), []).append(q)

    all_groups_intact = all(
        len({which_split(m) for m in members}) == 1 for members in groups.values()
    )
    check(
        f"all {len(groups)} near-duplicate groups stay within a single split",
        all_groups_intact,
    )

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL DATA PREPARATION CHECKS PASSED")


if __name__ == "__main__":
    main()
