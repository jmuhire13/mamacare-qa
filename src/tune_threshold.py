"""
Picks the gate threshold: above it, the app attempts an answer; below it,
the app refuses rather than guess.

Uses the PRETRAINED model (never fine-tuned), not the fine-tuned retriever,
to make this accept/refuse decision. This is a deliberate choice based on a
direct comparison between the two models' confidence scores: fine-tuning
made retrieval more accurate but also made its confidence scores less
reliable at telling real maternal-health questions apart from other
health-adjacent topics (hair loss, sports injuries, etc.). The pretrained
model is a better judge of
"is this even in-domain," even though the fine-tuned model is better at
picking the exact right answer once that gate says yes. app.py uses the
pretrained model for this gate decision and the fine-tuned model separately
for the actual answer, matching this calibration.

How the threshold is chosen: for each validation question, compute its best
similarity to any answer using the pretrained model (should be ABOVE the
threshold: the app must be willing to attempt real in-domain questions),
and the same for each off-topic question (should be BELOW the threshold -
the app must refuse them). We pick the value that best separates the two
groups.

The off-topic calibration set is deliberately larger than it first was: the
original 18 hand-written probes were mostly easy (unrelated topics like
"capital of France"), and a proper held-out test later showed the threshold
they produced only reached ~50% refusal on harder, health-adjacent
questions (hair loss, seasonal allergies, etc.).
Calibrating against `data/out_of_domain_calibration.json` (58 harder
questions) as well fixes this. Confirmed by checking the resulting
threshold against a completely separate 59-question held-out set that this
script never touches (see out_of_domain_test.py).

Run with: python src/tune_threshold.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

GATE_MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path("data/processed")
THRESHOLD_PATH = Path("data/processed/refusal_threshold.json")
OOD_CALIBRATION_PATH = Path("data/out_of_domain_calibration.json")

# Deliberately unrelated to maternal health, covering different topics and
# phrasing styles, so the threshold isn't tuned to just one kind of
# off-topic question.
OFF_TOPIC_PROBES = [
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
    # Added after a real failure found by testing the live app: everyday
    # hobby/sport questions can accidentally share words with unrelated
    # medical answers (e.g. "play volleyball" vs "playing a key role in...").
    # Kept in the user's exact original wording, not a cleaned-up paraphrase -
    # a rephrased version scored differently and would have hidden this case.
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


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_answers = [item["answer"] for item in kb_items]

    model = SentenceTransformer(GATE_MODEL_NAME)
    kb_embeddings = model.encode(kb_answers, normalize_embeddings=True)

    def best_similarity(question):
        q_embedding = model.encode([question], normalize_embeddings=True)
        return float(cosine_similarity(q_embedding, kb_embeddings)[0].max())

    ood_calibration = json.load(open(OOD_CALIBRATION_PATH, encoding="utf-8"))

    in_domain_scores = [best_similarity(item["question"]) for item in val]
    off_topic_scores = [best_similarity(q) for q in OFF_TOPIC_PROBES + ood_calibration]

    print("In-domain (val) best-match similarity: min={:.3f} max={:.3f} mean={:.3f}".format(
        min(in_domain_scores), max(in_domain_scores), sum(in_domain_scores) / len(in_domain_scores)
    ))
    print("Off-topic probe best-match similarity: min={:.3f} max={:.3f} mean={:.3f}".format(
        min(off_topic_scores), max(off_topic_scores), sum(off_topic_scores) / len(off_topic_scores)
    ))

    # Scan candidate thresholds and pick the one that correctly keeps the
    # most in-domain questions answerable while rejecting the most
    # off-topic probes.
    best_threshold = None
    best_score = -1
    for candidate in [i / 200 for i in range(0, 201)]:
        accepted_in_domain = sum(1 for s in in_domain_scores if s >= candidate)
        rejected_off_topic = sum(1 for s in off_topic_scores if s < candidate)
        combined = accepted_in_domain + rejected_off_topic
        if combined > best_score:
            best_score = combined
            best_threshold = candidate

    accepted_in_domain = sum(1 for s in in_domain_scores if s >= best_threshold)
    rejected_off_topic = sum(1 for s in off_topic_scores if s < best_threshold)

    print(f"\nChosen threshold: {best_threshold}")
    print(f"  In-domain val questions correctly answerable: {accepted_in_domain}/{len(in_domain_scores)}")
    print(f"  Off-topic probes correctly refused: {rejected_off_topic}/{len(off_topic_scores)}")

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": best_threshold,
            "in_domain_accept_rate": round(accepted_in_domain / len(in_domain_scores), 3),
            "off_topic_reject_rate": round(rejected_off_topic / len(off_topic_scores), 3),
            "gate_model": GATE_MODEL_NAME,
            "note": "accept rate here means 'gate did not refuse', not 'answer given was correct'. See "
                    "out_of_domain_test.py for the true correctness breakdown",
        }, f, indent=2)
    print(f"Saved to {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
