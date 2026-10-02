"""
Experiment 7: the final out-of-domain refusal test, reported with a 95%
confidence interval so the number can actually be trusted.

Evaluated against data/out_of_domain_heldout.json (59 questions) only -
never against the calibration half, which was used to pick the threshold
in tune_threshold.py. Mixing the two would mean testing the threshold
against data it was chosen to fit, which proves nothing.

Also reports the real in-domain accuracy breakdown for the TEST split
(never used for calibration either): true positive (confident AND correct)
vs confidently wrong (confident, wrong answer given) vs missed refusals -
not just whether the confidence gate was passed, which is a different and
much less useful question.

Uses the hybrid design: the PRETRAINED model gates whether to attempt an
answer, the FINE-TUNED model decides which answer to give, matching app.py.

Run with: python src/out_of_domain_test.py
"""

import json
import math
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/processed")
GATE_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
OOD_HELDOUT_PATH = Path("data/out_of_domain_heldout.json")
RESULTS_PATH = Path("data/processed/out_of_domain_results.json")


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a proportion - makes explicit how much
    to trust a rate measured on a limited number of examples."""
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
    kb_answers = [item["answer"] for item in kb_items]

    with open(OOD_HELDOUT_PATH, encoding="utf-8") as f:
        ood_questions = json.load(f)

    threshold = json.load(open(DATA_DIR / "refusal_threshold.json", encoding="utf-8"))["threshold"]

    gate_model = SentenceTransformer(GATE_MODEL_NAME)
    retrieval_model = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    gate_kb_embeddings = gate_model.encode(kb_answers, normalize_embeddings=True)
    retrieval_kb_embeddings = retrieval_model.encode(kb_answers, normalize_embeddings=True)

    def gate_passes(question):
        emb = gate_model.encode([question], normalize_embeddings=True)
        return float(cosine_similarity(emb, gate_kb_embeddings)[0].max()) >= threshold

    def retrieve_answer(question):
        emb = retrieval_model.encode([question], normalize_embeddings=True)
        sims = cosine_similarity(emb, retrieval_kb_embeddings)[0]
        return kb_answers[int(sims.argmax())]

    true_positive = 0
    confidently_wrong = 0
    refused_wouldve_been_right = 0
    refused_correctly = 0
    for item in test:
        passes = gate_passes(item["question"])
        answer = retrieve_answer(item["question"])
        is_correct = answer == item["answer"]
        if passes and is_correct:
            true_positive += 1
        elif passes and not is_correct:
            confidently_wrong += 1
        elif not passes and is_correct:
            refused_wouldve_been_right += 1
        else:
            refused_correctly += 1

    ood_refused = sum(1 for q in ood_questions if not gate_passes(q))
    ood_wrongly_answered = [q for q in ood_questions if gate_passes(q)]
    ci_low, ci_high = wilson_interval(ood_refused, len(ood_questions))
    true_acc_ci_low, true_acc_ci_high = wilson_interval(true_positive, len(test))

    results = {
        "threshold": threshold,
        "in_domain_test": {
            "n": len(test),
            "true_positive": true_positive,
            "confidently_wrong": confidently_wrong,
            "refused_wouldve_been_right": refused_wouldve_been_right,
            "refused_correctly": refused_correctly,
            "true_accuracy": round(true_positive / len(test), 3),
            "true_accuracy_95ci": [true_acc_ci_low, true_acc_ci_high],
        },
        "out_of_domain": {
            "n": len(ood_questions),
            "correctly_refused": ood_refused,
            "refusal_rate": round(ood_refused / len(ood_questions), 3),
            "refusal_rate_95ci": [ci_low, ci_high],
            "wrongly_answered": ood_wrongly_answered,
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Threshold: {threshold}\n")
    print("IN-DOMAIN TEST SPLIT (43 questions), full breakdown:")
    print(f"  True positive (confident AND correct):     {true_positive}/{len(test)} "
          f"({results['in_domain_test']['true_accuracy']:.1%}), 95% CI "
          f"[{true_acc_ci_low:.1%}, {true_acc_ci_high:.1%}]")
    print(f"  Confidently WRONG (confident, wrong answer): {confidently_wrong}/{len(test)}")
    print(f"  Refused, would've been correct (missed):    {refused_wouldve_been_right}/{len(test)}")
    print(f"  Refused, would've been wrong anyway:        {refused_correctly}/{len(test)}")
    print()
    print(f"OUT-OF-DOMAIN, held-out set ({len(ood_questions)} fresh questions, never used to calibrate):")
    print(f"  Correctly refused: {ood_refused}/{len(ood_questions)} "
          f"({results['out_of_domain']['refusal_rate']:.1%}), 95% CI [{ci_low:.1%}, {ci_high:.1%}]")
    if ood_wrongly_answered:
        print("  Wrongly answered:")
        for q in ood_wrongly_answered:
            print(f"    - {q}")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
