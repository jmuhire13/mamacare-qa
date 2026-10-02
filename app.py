"""
The web app: a user types a maternal-health question, the app finds the
closest matching reviewed answer out of all 430, and returns it - or
refuses if nothing in the knowledge base is a close enough match.

Uses two different models for two different jobs, based on a direct,
measured comparison: the ORIGINAL pretrained model
(never fine-tuned) decides WHETHER to answer at all, and the FINE-TUNED
model decides WHICH answer to give once that gate says yes. Fine-tuning
made retrieval more accurate but also made confidence scores less reliable
for telling real maternal-health questions apart from other health-adjacent
topics - the pretrained model turned out to be a better judge of "is this
even in-domain," even though it's worse at picking the exact right answer
once it agrees to attempt one. Using each model for the job it is better
at measurably beats using either model alone for both jobs.

Run with: streamlit run app.py
Then open the local URL it prints (usually http://localhost:8501).
"""

import json
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

GATE_MODEL_NAME = "all-MiniLM-L6-v2"  # pretrained, decides whether to answer at all

# Uses the local model folder when it exists (after running finetune_retriever.py
# yourself), and falls back to the Hugging Face Hub copy otherwise - the deployed
# app has no local models/ folder, only the Hub repo.
LOCAL_RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
HUB_RETRIEVAL_MODEL_ID = "jmuhire13/mamacare-qa-retriever"
RETRIEVAL_MODEL_SOURCE = (
    str(LOCAL_RETRIEVAL_MODEL_DIR) if LOCAL_RETRIEVAL_MODEL_DIR.exists() else HUB_RETRIEVAL_MODEL_ID
)

DATA_DIR = Path("data/processed")
THRESHOLD_PATH = Path("data/processed/refusal_threshold.json")

REFUSAL_MESSAGE = (
    "I don't have a confident answer to that in my maternal health knowledge base. "
    "Please ask a question about pregnancy, childbirth, or newborn care, or consult "
    "a qualified health worker."
)

DISCLAIMER = (
    "This assistant answers from a reviewed set of maternal health Q&A. "
    "It is not a substitute for medical advice. Always consult a qualified "
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
    return [item["question"] for item in kb_items], [item["answer"] for item in kb_items]


def load_threshold():
    with open(THRESHOLD_PATH, encoding="utf-8") as f:
        return json.load(f)["threshold"]


# Streamlit re-runs this whole script on every interaction, so without
# caching, both models would reload from scratch on every single question.
@st.cache_resource(show_spinner="Loading models and knowledge base...")
def load_resources():
    gate_model = SentenceTransformer(GATE_MODEL_NAME)
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_SOURCE)
    kb_questions, kb_answers = load_knowledge_base()
    gate_kb_embeddings = gate_model.encode(kb_answers, normalize_embeddings=True)
    retrieval_kb_embeddings = retrieval_model.encode(kb_answers, normalize_embeddings=True)
    threshold = load_threshold()
    return {
        "gate_model": gate_model,
        "retrieval_model": retrieval_model,
        "kb_questions": kb_questions,
        "kb_answers": kb_answers,
        "gate_kb_embeddings": gate_kb_embeddings,
        "retrieval_kb_embeddings": retrieval_kb_embeddings,
        "threshold": threshold,
    }


def answer_question(question, resources):
    if not question or not question.strip():
        return "Please type a question."

    # Step 1: should we even attempt an answer? Decided by the PRETRAINED
    # model, which is the better judge of "is this in-domain at all."
    gate_embedding = resources["gate_model"].encode([question], normalize_embeddings=True)
    gate_score = float(cosine_similarity(gate_embedding, resources["gate_kb_embeddings"])[0].max())
    if gate_score < resources["threshold"]:
        return REFUSAL_MESSAGE

    # Step 2: which answer? Decided by the FINE-TUNED model, which is more
    # accurate at picking the exact right answer once we've agreed to try.
    retrieval_embedding = resources["retrieval_model"].encode([question], normalize_embeddings=True)
    retrieval_similarities = cosine_similarity(retrieval_embedding, resources["retrieval_kb_embeddings"])[0]
    best_index = retrieval_similarities.argmax()

    # Shown so the user can judge the match themselves - at a measured 4/43
    # confidently-wrong rate, surfacing the matched question is the cheapest
    # way to make a mismatch visible instead of invisible.
    matched_question = resources["kb_questions"][best_index]
    return f'Matched to this question in the knowledge base:\n"{matched_question}"\n\n{resources["kb_answers"][best_index]}'


st.set_page_config(page_title="Maternal Health Q&A Assistant")
st.title("Maternal Health Q&A Assistant")
st.caption(DISCLAIMER)

resources = load_resources()

question = st.text_input("Your question", placeholder="e.g. Why do I feel tired during pregnancy?")
if question:
    st.text(answer_question(question, resources))
