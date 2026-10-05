"""
Retrains the retriever with a different seed, saves to a
throwaway folder (never touches models/finetuned-retriever), and evaluates
it the same way evaluate_finetuned_retriever.py does. Measures how much the
result depends on the random seed. The shipped model is not overwritten.

Run with: python src/check_seed_variance.py <seed>
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "data" / "processed"
BASE_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5

SEED = int(sys.argv[1])
OUTPUT_DIR = Path("models/seed_checks") / f"seed_{SEED}"


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def evaluate_ranking(query_items, score_fn, k=3):
    reciprocal_ranks = []
    top1_hits = 0
    topk_hits = 0
    for item in query_items:
        scores = score_fn(item["question"])
        ranked_indices = np.argsort(-scores)
        true_index = item["kb_index"]
        rank = int(np.where(ranked_indices == true_index)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank)
        if rank == 1:
            top1_hits += 1
        if rank <= k:
            topk_hits += 1
    n = len(query_items)
    return {
        "top_1": round(top1_hits / n, 3),
        f"top_{k}": round(topk_hits / n, 3),
        "mrr": round(float(np.mean(reciprocal_ranks)), 3),
        "n_queries": n,
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    model = SentenceTransformer(BASE_MODEL_NAME)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        shuffled = train[:]
        random.shuffle(shuffled)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(shuffled), BATCH_SIZE):
            batch = shuffled[i:i + BATCH_SIZE]
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]
            question_features = model.tokenize(questions)
            answer_features = model.tokenize(answers)
            optimizer.zero_grad()
            loss = loss_fn([question_features, answer_features], labels=None)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"seed {SEED} epoch {epoch}/{EPOCHS}, average loss: {total_loss / n_batches:.4f}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))

    kb_items = train + val + test
    for i, item in enumerate(kb_items):
        item["kb_index"] = i
    kb_answers = [item["answer"] for item in kb_items]
    val_items = kb_items[len(train):len(train) + len(val)]
    test_items = kb_items[len(train) + len(val):]

    kb_embeddings = model.encode(kb_answers, normalize_embeddings=True)

    def embedding_score(question):
        q_embedding = model.encode([question], normalize_embeddings=True)
        return cosine_similarity(q_embedding, kb_embeddings)[0]

    val_result = evaluate_ranking(val_items, embedding_score)
    test_result = evaluate_ranking(test_items, embedding_score)

    print(f"RESULT seed={SEED} val={val_result}", flush=True)
    print(f"RESULT seed={SEED} test={test_result}", flush=True)


if __name__ == "__main__":
    main()
