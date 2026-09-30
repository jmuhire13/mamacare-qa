"""
Phase 0 fix: make the out-of-domain evaluation trustworthy enough to draw
real conclusions from, instead of comparing designs on a set so small that
a one-question difference could just be noise.

Reports, for each of the three refusal-decision designs we've tried:
  - refusal rate on the larger, harder out-of-domain set, WITH a 95% Wilson
    confidence interval (a plain threshold count doesn't tell you how much
    to trust it - the interval does)
  - AUROC: the probability that a randomly chosen in-domain question scores
    higher than a randomly chosen out-of-domain question, using the raw
    similarity score rather than any specific threshold. This lets us
    compare how good each design's underlying signal is, independent of
    where exactly we happen to draw the accept/refuse line.

Run with: python src/evaluate_domain_separation.py
"""

import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/processed")
GATE_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
OOD_QUESTIONS_PATH = Path("data/out_of_domain_test.json")


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a proportion. Wider n = tighter interval -
    this is exactly the fix for 'you can't tell 75% from 80% apart' at n=40."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (round(center - margin, 3), round(center + margin, 3))


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_questions = [item["question"] for item in kb_items]
    kb_answers = [item["answer"] for item in kb_items]

    with open(OOD_QUESTIONS_PATH, encoding="utf-8") as f:
        ood_questions = json.load(f)

    print(f"Out-of-domain test set size: {len(ood_questions)} (was 40)")
    print(f"In-domain test set size: {len(test)}\n")

    gate_model = SentenceTransformer(GATE_MODEL_NAME)
    retrieval_model = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))

    gate_kb_emb = gate_model.encode(kb_answers, normalize_embeddings=True)
    retrieval_kb_emb = retrieval_model.encode(kb_answers, normalize_embeddings=True)
    retrieval_kb_q_emb = retrieval_model.encode(kb_questions, normalize_embeddings=True)

    test_offset = len(train) + len(val)

    # Design A: single fine-tuned signal (answer-similarity only)
    def score_A(text, exclude_index=None):
        emb = retrieval_model.encode([text], normalize_embeddings=True)
        return float(cosine_similarity(emb, retrieval_kb_emb)[0].max())

    # Design B: fine-tuned, two-signal (min of answer-sim, question-sim)
    def score_B(text, exclude_index=None):
        emb = retrieval_model.encode([text], normalize_embeddings=True)
        answer_sim = float(cosine_similarity(emb, retrieval_kb_emb)[0].max())
        q_sims = cosine_similarity(emb, retrieval_kb_q_emb)[0]
        if exclude_index is not None:
            q_sims = np.delete(q_sims, exclude_index)
        return min(answer_sim, float(q_sims.max()))

    # Design C: hybrid gate (pretrained model, answer-similarity only)
    def score_C(text, exclude_index=None):
        emb = gate_model.encode([text], normalize_embeddings=True)
        return float(cosine_similarity(emb, gate_kb_emb)[0].max())

    designs = {
        "A: single fine-tuned signal": (score_A, 0.34),
        "B: fine-tuned, two-signal": (score_B, 0.32),
        "C: hybrid (pretrained gate)": (score_C, 0.34),
    }

    for name, (score_fn, threshold) in designs.items():
        in_domain_scores = [
            score_fn(item["question"], exclude_index=test_offset + i) for i, item in enumerate(test)
        ]
        ood_scores = [score_fn(q) for q in ood_questions]

        refused = sum(1 for s in ood_scores if s < threshold)
        ci_low, ci_high = wilson_interval(refused, len(ood_scores))

        labels = [1] * len(in_domain_scores) + [0] * len(ood_scores)
        scores = in_domain_scores + ood_scores
        auroc = roc_auc_score(labels, scores)

        print(f"--- {name} (threshold={threshold}) ---")
        print(f"  Out-of-domain refusal: {refused}/{len(ood_scores)} "
              f"({refused/len(ood_scores):.1%}), 95% CI [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"  AUROC (threshold-independent separation quality): {auroc:.3f}")
        print()


if __name__ == "__main__":
    main()
