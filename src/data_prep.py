"""
Cleans the raw MOTHER maternal-health Q&A dataset and splits it into
train/val/test sets for the retrieval and generation experiments.

Input:  data/raw/mother_question_and_answer_pairs_data.json
Output: data/processed/train.json, val.json, test.json
        data/processed/cleaning_report.json  (exact counts, for the report)

Run with:  python src/data_prep.py
"""

import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAW_PATH = Path("data/raw/mother_question_and_answer_pairs_data.json")
OUT_DIR = Path("data/processed")

SEED = 42
NEAR_DUPLICATE_THRESHOLD = 0.65  # cosine similarity above this = "same topic, keep together"
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# Words a real question normally starts with. Used to catch rows where the
# "question" column is actually a sentence describing the answer, not a
# question someone would type in.
QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "is", "are",
    "can", "could", "does", "do", "did", "will", "would", "should",
    "am", "was", "were", "has", "have",
}


def normalize_for_matching(text):
    """Lowercase, strip punctuation, collapse whitespace. Used to spot
    duplicate questions even when they differ by a comma or capital letter."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_leakage(question):
    """A leakage row is a 'question' that reads like a statement, not a
    question someone would actually ask (no '?' and doesn't start with a
    question word). These got into the dataset from summarised answers."""
    if "?" in question:
        return False
    first_word = question.strip().lower().split()[0] if question.strip() else ""
    return first_word not in QUESTION_WORDS


def is_cut_off(answer):
    """An answer that ends with a colon is almost always introducing a list
    that never made it into the JSON (e.g. '...Here are some ideas to try:')."""
    return answer.strip().endswith(":")


def find_near_duplicate_groups(questions):
    """Some questions are worded differently but ask about the same narrow
    topic (e.g. two different IUD questions). We don't want one to land in
    training and its near-twin to land in the test set, since that would let
    the model "cheat" by memorising something very close to a test question.
    This groups such questions together using simple TF-IDF similarity, so
    the split step can keep each group entirely on one side."""
    vectorizer = TfidfVectorizer().fit_transform(questions)
    similarity = cosine_similarity(vectorizer)

    n = len(questions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    linked_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= NEAR_DUPLICATE_THRESHOLD:
                union(i, j)
                linked_pairs.append((i, j, round(float(similarity[i, j]), 3)))

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values()), linked_pairs


def split_groups(groups, ratios, seed):
    """Shuffles groups (not individual rows) and assigns whole groups to
    train/val/test, so linked near-duplicate questions never get split
    across sets."""
    rng = random.Random(seed)
    groups = groups[:]
    rng.shuffle(groups)

    total_rows = sum(len(g) for g in groups)
    train_target = round(total_rows * ratios[0])
    val_target = round(total_rows * ratios[1])

    train, val, test = [], [], []
    for group in groups:
        if len(train) < train_target:
            train.extend(group)
        elif len(val) < val_target:
            val.extend(group)
        else:
            test.extend(group)
    return train, val, test


def main():
    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    report = {"raw_count": len(raw)}

    # Step 1: drop leakage rows (statements pretending to be questions)
    leakage_idx = {i for i, item in enumerate(raw) if is_leakage(item["question"])}
    report["leakage_removed"] = len(leakage_idx)

    # Step 2: drop cut-off answers
    cutoff_idx = {i for i, item in enumerate(raw) if is_cut_off(item["answer"])}
    report["cutoff_removed"] = len(cutoff_idx)

    # Step 3: drop exact/near-exact duplicate questions, keeping the first copy
    seen_questions = set()
    duplicate_idx = set()
    for i, item in enumerate(raw):
        key = normalize_for_matching(item["question"])
        if key in seen_questions:
            duplicate_idx.add(i)
        else:
            seen_questions.add(key)
    report["duplicate_removed"] = len(duplicate_idx)

    # Rows that survive all three checks
    keep_idx = [
        i for i in range(len(raw))
        if i not in leakage_idx and i not in cutoff_idx and i not in duplicate_idx
    ]
    clean = [raw[i] for i in keep_idx]
    report["clean_count"] = len(clean)

    # Step 4: find near-duplicate topic groups among the surviving rows,
    # so the split step can keep each group on one side.
    questions = [item["question"] for item in clean]
    groups, linked_pairs = find_near_duplicate_groups(questions)
    report["near_duplicate_links"] = [
        {"question_a": questions[a], "question_b": questions[b], "similarity": sim}
        for a, b, sim in linked_pairs
    ]

    # Step 5: split by group, not by row, with a fixed seed for reproducibility
    train_idx, val_idx, test_idx = split_groups(groups, SPLIT_RATIOS, SEED)
    train = [clean[i] for i in train_idx]
    val = [clean[i] for i in val_idx]
    test = [clean[i] for i in test_idx]

    report["train_count"] = len(train)
    report["val_count"] = len(val)
    report["test_count"] = len(test)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        with open(OUT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Raw rows:              {report['raw_count']}")
    print(f"Removed - leakage:     {report['leakage_removed']}")
    print(f"Removed - cut-off:     {report['cutoff_removed']}")
    print(f"Removed - duplicate:   {report['duplicate_removed']}")
    print(f"Clean rows:            {report['clean_count']}")
    print(f"Near-duplicate links:  {len(linked_pairs)}")
    print(f"Train / Val / Test:    {report['train_count']} / {report['val_count']} / {report['test_count']}")


if __name__ == "__main__":
    main()
