"""
The web app: a user types a maternal-health question, the app finds the
closest matching reviewed answer out of all 430, and returns it - or
refuses if nothing in the knowledge base is a close enough match.

Uses two different models for two different jobs, based on a direct,
measured comparison (see docs/WORK_LOG.md): the ORIGINAL pretrained model
(never fine-tuned) decides WHETHER to answer at all, and the FINE-TUNED
model decides WHICH answer to give once that gate says yes. Fine-tuning
made retrieval more accurate but also made confidence scores less reliable
for telling real maternal-health questions apart from other health-adjacent
topics - the pretrained model turned out to be a better judge of "is this
even in-domain," even though it's worse at picking the exact right answer
once it agrees to attempt one. Using each model for the job it is better
at measurably beats using either model alone for both jobs.

Run with: python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import json
from pathlib import Path

import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

GATE_MODEL_NAME = "all-MiniLM-L6-v2"  # pretrained, decides whether to answer at all
RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")  # fine-tuned, decides which answer
DATA_DIR = Path("data/processed")
THRESHOLD_PATH = Path("data/processed/refusal_threshold.json")

REFUSAL_MESSAGE = (
    "I don't have a confident answer to that in my maternal health knowledge base. "
    "Please ask a question about pregnancy, childbirth, or newborn care, or consult "
    "a qualified health worker."
)

DISCLAIMER = (
    "This assistant answers from a reviewed set of maternal health Q&A. "
    "It is not a substitute for medical advice - always consult a qualified "
    "health worker for your specific situation."
)


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_knowledge_base():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    return [item["answer"] for item in kb_items]


def load_threshold():
    with open(THRESHOLD_PATH, encoding="utf-8") as f:
        return json.load(f)["threshold"]


print("Loading gate model (pretrained), retrieval model (fine-tuned), and knowledge base...")
gate_model = SentenceTransformer(GATE_MODEL_NAME)
retrieval_model = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
kb_answers = load_knowledge_base()
gate_kb_embeddings = gate_model.encode(kb_answers, normalize_embeddings=True)
retrieval_kb_embeddings = retrieval_model.encode(kb_answers, normalize_embeddings=True)
threshold = load_threshold()
print(f"Ready. Knowledge base: {len(kb_answers)} answers. Refusal threshold: {threshold}")


def answer_question(question):
    if not question or not question.strip():
        return "Please type a question."

    # Step 1: should we even attempt an answer? Decided by the PRETRAINED
    # model, which is the better judge of "is this in-domain at all."
    gate_embedding = gate_model.encode([question], normalize_embeddings=True)
    gate_score = float(cosine_similarity(gate_embedding, gate_kb_embeddings)[0].max())
    if gate_score < threshold:
        return REFUSAL_MESSAGE

    # Step 2: which answer? Decided by the FINE-TUNED model, which is more
    # accurate at picking the exact right answer once we've agreed to try.
    retrieval_embedding = retrieval_model.encode([question], normalize_embeddings=True)
    retrieval_similarities = cosine_similarity(retrieval_embedding, retrieval_kb_embeddings)[0]
    best_index = retrieval_similarities.argmax()
    return kb_answers[best_index]


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Your question", placeholder="e.g. Why do I feel tired during pregnancy?"),
    outputs=gr.Textbox(label="Answer"),
    title="Maternal Health Q&A Assistant",
    description=DISCLAIMER,
)

if __name__ == "__main__":
    demo.launch()
