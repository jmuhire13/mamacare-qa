"""
Experiment: does cross-encoder reranking fix the fine-tuned retriever's
wrong-answer cases?

Why this might help: the bi-encoder (our fine-tuned retriever) squeezes the
question and each answer into separate single vectors, compared only after
the fact: it never lets the question and answer "look at" each other
directly. A cross-encoder instead feeds [question] [SEP] [answer] into a
transformer TOGETHER, so every word of the question can attend to every
word of the answer. This is slower (it has to do this once per candidate,
not once per whole knowledge base), so the normal pattern is: use the
fast bi-encoder to narrow 430 answers down to a top-10 shortlist, then use
the slower but more precise cross-encoder to re-sort just those 10.

Diagnostic first: for the 7 known wrong-retrieval test cases, is the true
answer within the bi-encoder's top-10? If not, reranking can't fix it -
reranking can only reorder candidates that are already in the shortlist.

Run with: python src/rerank_experiment.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/processed")
RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SHORTLIST_SIZE = 10


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_answers = [item["answer"] for item in kb_items]

    bi_encoder = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    kb_embeddings = bi_encoder.encode(kb_answers, normalize_embeddings=True)

    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    bi_encoder_correct = 0
    reranked_correct = 0
    true_answer_missing_from_shortlist = 0
    fixed_by_reranking = []
    broken_by_reranking = []

    for item in test:
        q_embedding = bi_encoder.encode([item["question"]], normalize_embeddings=True)
        sims = cosine_similarity(q_embedding, kb_embeddings)[0]
        shortlist_indices = np.argsort(-sims)[:SHORTLIST_SIZE]

        bi_encoder_top1 = kb_answers[shortlist_indices[0]]
        if bi_encoder_top1 == item["answer"]:
            bi_encoder_correct += 1

        true_in_shortlist = item["answer"] in [kb_answers[i] for i in shortlist_indices]
        if not true_in_shortlist:
            true_answer_missing_from_shortlist += 1

        # Cross-encoder scores each (question, candidate answer) pair
        # jointly, then we take whichever candidate it scores highest.
        pairs = [(item["question"], kb_answers[i]) for i in shortlist_indices]
        rerank_scores = reranker.predict(pairs)
        best_local_index = int(np.argmax(rerank_scores))
        reranked_top1 = kb_answers[shortlist_indices[best_local_index]]

        if reranked_top1 == item["answer"]:
            reranked_correct += 1

        if bi_encoder_top1 != item["answer"] and reranked_top1 == item["answer"]:
            fixed_by_reranking.append(item["question"])
        if bi_encoder_top1 == item["answer"] and reranked_top1 != item["answer"]:
            broken_by_reranking.append(item["question"])

    n = len(test)
    print(f"Bi-encoder alone Top-1 accuracy:        {bi_encoder_correct}/{n} ({bi_encoder_correct/n:.1%})")
    print(f"Bi-encoder + cross-encoder rerank Top-1: {reranked_correct}/{n} ({reranked_correct/n:.1%})")
    print(f"True answer missing from top-{SHORTLIST_SIZE} shortlist entirely: {true_answer_missing_from_shortlist}/{n}")
    print(f"\nCases FIXED by reranking ({len(fixed_by_reranking)}):")
    for q in fixed_by_reranking:
        print(f"  - {q}")
    print(f"\nCases BROKEN by reranking ({len(broken_by_reranking)}):")
    for q in broken_by_reranking:
        print(f"  - {q}")


if __name__ == "__main__":
    main()
