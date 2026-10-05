"""
Computes the metrics reported in the write-up that were previously calculated
outside the repository: Recall@5 for the four retrieval methods, 95% Wilson
intervals for their Top-1 accuracy, and a bootstrap 95% interval for the AUROC
of the hybrid refusal gate.

Run with: python src/compute_report_metrics.py
"""

import json
import math
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
from sklearn.metrics.pairwise import cosine_similarity

from retrieval_baselines import simple_tokenize

DATA_DIR = Path("data/processed")
OOD_HELDOUT_PATH = Path("data/out_of_domain_heldout.json")
PRETRAINED_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
BOOTSTRAP_RESAMPLES = 2000
SEED = 42


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def wilson_interval(successes, n, z=1.96):
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return round(center - margin, 3), round(center + margin, 3)


def rank_of_gold(scores, gold_index):
    ranked = np.argsort(-scores)
    return int(np.where(ranked == gold_index)[0][0]) + 1


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    for i, item in enumerate(kb_items):
        item["kb_index"] = i
    kb_answers = [item["answer"] for item in kb_items]
    test_items = kb_items[len(train) + len(val):]
    n = len(test_items)

    tfidf = TfidfVectorizer()
    kb_tfidf = tfidf.fit_transform(kb_answers)
    bm25 = BM25Okapi([simple_tokenize(a) for a in kb_answers])
    pretrained = SentenceTransformer(PRETRAINED_MODEL_NAME)
    kb_pre = pretrained.encode(kb_answers, normalize_embeddings=True)
    finetuned = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    kb_ft = finetuned.encode(kb_answers, normalize_embeddings=True)

    score_fns = {
        "TF-IDF": lambda q: cosine_similarity(tfidf.transform([q]), kb_tfidf)[0],
        "BM25": lambda q: np.array(bm25.get_scores(simple_tokenize(q))),
        "Pretrained embeddings": lambda q: cosine_similarity(
            pretrained.encode([q], normalize_embeddings=True), kb_pre)[0],
        "Fine-tuned embeddings": lambda q: cosine_similarity(
            finetuned.encode([q], normalize_embeddings=True), kb_ft)[0],
    }

    print("Retrieval, test split (43 questions, all 430 answers as candidates)")
    print(f"{'Method':<24}{'Top-1':>8}{'Top-1 95% CI':>20}{'Recall@5':>10}{'Recall@5 count':>16}")
    for name, fn in score_fns.items():
        ranks = [rank_of_gold(fn(item["question"]), item["kb_index"]) for item in test_items]
        top1 = sum(r == 1 for r in ranks)
        top5 = sum(r <= 5 for r in ranks)
        lo, hi = wilson_interval(top1, n)
        print(f"{name:<24}{top1}/{n:>3}{f'[{lo:.3f}, {hi:.3f}]':>20}{top5 / n:>10.3f}{f'{top5}/{n}':>16}")

    correct = {}
    for name in ["Pretrained embeddings", "Fine-tuned embeddings"]:
        fn = score_fns[name]
        correct[name] = {i for i, item in enumerate(test_items)
                         if rank_of_gold(fn(item["question"]), item["kb_index"]) == 1}
    pre_c, ft_c = correct["Pretrained embeddings"], correct["Fine-tuned embeddings"]
    print()
    print(f"Paired comparison, Top-1: fine-tuned fixed {len(ft_c - pre_c)}, broke {len(pre_c - ft_c)}, "
          f"both correct {len(pre_c & ft_c)}, both wrong {n - len(pre_c | ft_c)}")
    fixed, broke = len(ft_c - pre_c), len(pre_c - ft_c)
    if fixed + broke:
        sign_p = binomtest(fixed, fixed + broke, 0.5, alternative="two-sided").pvalue
        print(f"Sign test on the {fixed + broke} disagreeing questions (two-sided exact binomial): p = {sign_p:.4f}")

    bm25_b0 = BM25Okapi([simple_tokenize(a) for a in kb_answers], b=0.0)
    b0_top1 = sum(rank_of_gold(np.array(bm25_b0.get_scores(simple_tokenize(item["question"]))),
                               item["kb_index"]) == 1 for item in test_items)
    print(f"BM25 with length normalization disabled (b = 0): Top-1 {b0_top1}/{n}")

    heldout = json.load(open(OOD_HELDOUT_PATH, encoding="utf-8"))
    gate_scores_test = [float(cosine_similarity(pretrained.encode([item["question"]], normalize_embeddings=True), kb_pre)[0].max())
                        for item in test_items]
    gate_scores_held = [float(cosine_similarity(pretrained.encode([q], normalize_embeddings=True), kb_pre)[0].max())
                        for q in heldout]
    labels = np.array([1] * len(gate_scores_test) + [0] * len(gate_scores_held))
    scores = np.array(gate_scores_test + gate_scores_held)
    auroc = roc_auc_score(labels, scores)
    rng = np.random.RandomState(SEED)
    boot = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.randint(0, len(labels), size=len(labels))
        if len(set(labels[idx])) < 2:
            continue
        boot.append(roc_auc_score(labels[idx], scores[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print()
    print(f"Hybrid gate AUROC (test {len(gate_scores_test)} vs held-out {len(gate_scores_held)}): {auroc:.4f}")
    print(f"Bootstrap 95% interval ({BOOTSTRAP_RESAMPLES} resamples, seed {SEED}): [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
