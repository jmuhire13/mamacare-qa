"""
Recalibrates each design's threshold using ONLY calibration data (val, the
original 18 hand-written probes, and the new 58-question calibration half),
then reports the final refusal rate, with a 95% confidence interval,
against the 59-question held-out half, which none of that calibration ever
sees. This is the fair, final number for the report.

Run with: python src/finalize_thresholds.py
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
OOD_CALIBRATION_PATH = Path("data/out_of_domain_calibration.json")
OOD_HELDOUT_PATH = Path("data/out_of_domain_heldout.json")
RESULTS_DIR = Path("results")


ORIGINAL_18_PROBES = [
    "What is the capital of France?",
    "How do I fix a flat tire on my car?",
    "What's the weather going to be like tomorrow?",
    "Can you recommend a good recipe for lasagna?",
    "Who won the World Cup in 2018?",
    "How do I reset my email password?",
    "What is the square root of 144?",
    "What time does the stock market open?",
    "How do I train my dog to sit?",
    "What programming language should I learn first?",
    "Can you explain how photosynthesis works?",
    "What's a good movie to watch this weekend?",
    "What can i do to play volley ball",
    "How do I improve my basketball shooting technique?",
    "What's the best way to practice my golf swing?",
    "How can I get better at playing chess?",
    "What equipment do I need to start rock climbing?",
    "How do I train for a marathon?",
]


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (round(center - margin, 3), round(center + margin, 3))


def bootstrap_auroc_interval(labels, scores, resamples=2000, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    boot = []
    for _ in range(resamples):
        idx = rng.randint(0, len(labels), size=len(labels))
        if len(set(labels[idx])) < 2:
            continue
        boot.append(roc_auc_score(labels[idx], scores[idx]))
    return tuple(np.round(np.percentile(boot, [2.5, 97.5]), 4))


def best_threshold(in_domain_scores, off_topic_scores):
    best_t, best_score = None, -1
    for candidate in [i / 200 for i in range(0, 201)]:
        accepted = sum(1 for s in in_domain_scores if s >= candidate)
        rejected = sum(1 for s in off_topic_scores if s < candidate)
        combined = accepted + rejected
        if combined > best_score:
            best_score = combined
            best_t = candidate
    return best_t


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_questions = [item["question"] for item in kb_items]
    kb_answers = [item["answer"] for item in kb_items]

    ood_calibration = json.load(open(OOD_CALIBRATION_PATH, encoding="utf-8"))
    ood_heldout = json.load(open(OOD_HELDOUT_PATH, encoding="utf-8"))

    gate_model = SentenceTransformer(GATE_MODEL_NAME)
    retrieval_model = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))

    gate_kb_emb = gate_model.encode(kb_answers, normalize_embeddings=True)
    retrieval_kb_emb = retrieval_model.encode(kb_answers, normalize_embeddings=True)
    retrieval_kb_q_emb = retrieval_model.encode(kb_questions, normalize_embeddings=True)

    val_offset = len(train)
    test_offset = len(train) + len(val)

    def score_A(text, exclude_index=None):
        emb = retrieval_model.encode([text], normalize_embeddings=True)
        return float(cosine_similarity(emb, retrieval_kb_emb)[0].max())

    def score_B(text, exclude_index=None):
        emb = retrieval_model.encode([text], normalize_embeddings=True)
        answer_sim = float(cosine_similarity(emb, retrieval_kb_emb)[0].max())
        q_sims = cosine_similarity(emb, retrieval_kb_q_emb)[0]
        if exclude_index is not None:
            q_sims = np.delete(q_sims, exclude_index)
        return min(answer_sim, float(q_sims.max()))

    def score_C(text, exclude_index=None):
        emb = gate_model.encode([text], normalize_embeddings=True)
        return float(cosine_similarity(emb, gate_kb_emb)[0].max())

    designs = {"A: single fine-tuned signal": score_A, "B: fine-tuned, two-signal": score_B,
               "C: hybrid (pretrained gate)": score_C}

    calibration_off_topic_texts = ORIGINAL_18_PROBES + ood_calibration

    saved_scores = {}
    for name, score_fn in designs.items():
        # Calibrate: val (in-domain) + 18 probes + 58 calibration OOD questions.
        val_scores = [score_fn(item["question"], exclude_index=val_offset + i) for i, item in enumerate(val)]
        calib_off_topic_scores = [score_fn(q) for q in calibration_off_topic_texts]
        threshold = best_threshold(val_scores, calib_off_topic_scores)

        # Final, honest test: the untouched 59-question held-out half, plus
        # the real in-domain TEST split (also never used for calibration).
        test_scores = [score_fn(item["question"], exclude_index=test_offset + i) for i, item in enumerate(test)]
        heldout_scores = [score_fn(q) for q in ood_heldout]

        in_domain_ok = sum(1 for s in test_scores if s >= threshold)
        heldout_refused = sum(1 for s in heldout_scores if s < threshold)
        ci_low, ci_high = wilson_interval(heldout_refused, len(heldout_scores))

        labels = [1] * len(test_scores) + [0] * len(heldout_scores)
        scores = test_scores + heldout_scores
        auroc = roc_auc_score(labels, scores)
        saved_scores[name] = {
            "threshold": threshold,
            "test_scores": [float(s) for s in test_scores],
            "heldout_scores": [float(s) for s in heldout_scores],
        }

        print(f"--- {name} ---")
        print(f"  Recalibrated threshold: {threshold}")
        print(f"  In-domain TEST accept rate: {in_domain_ok}/{len(test_scores)} ({in_domain_ok/len(test_scores):.1%})")
        print(f"  Held-out OOD refusal: {heldout_refused}/{len(heldout_scores)} "
              f"({heldout_refused/len(heldout_scores):.1%}), 95% CI [{ci_low:.1%}, {ci_high:.1%}]")
        auroc_low, auroc_high = bootstrap_auroc_interval(labels, scores)
        print(f"  AUROC (test vs held-out): {auroc:.3f}, bootstrap 95% CI [{auroc_low:.4f}, {auroc_high:.4f}]")
        print()

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "design_scores.json", "w", encoding="utf-8") as f:
        json.dump(saved_scores, f, indent=2)
    print(f"Saved per-question scores to {RESULTS_DIR / 'design_scores.json'}")


if __name__ == "__main__":
    main()
