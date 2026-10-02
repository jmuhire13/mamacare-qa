"""
Experiments 1-3: baseline retrieval methods, no training involved.

The task: given a question, search all 430 reviewed answers and rank them
by similarity to the question text. "Correct" means the answer actually
paired with that question comes out on top.

Knowledge base = all 430 answers (train + val + test combined). This isn't
leakage: the deployed app has to be able to return any of the 430 reviewed
answers, so a question's own true answer must always be reachable. What's
being measured is whether text similarity alone can pick the right answer
out of all 430 candidates - not whether the model generalises to entirely
new questions, which these numbers don't test.

Run with: python src/retrieval_baselines.py
"""

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/processed")
RESULTS_PATH = Path("data/processed/baseline_results.json")

PRETRAINED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def simple_tokenize(text):
    """Lowercase, keep only letters/numbers/spaces, split on whitespace.
    Used for BM25, which expects a list of word tokens rather than raw text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def evaluate_ranking(query_items, kb_answers, score_fn, k=3):
    """Runs score_fn(query_question) -> array of scores (one per KB answer)
    for every query, then checks where the true answer ranks.

    Returns Top-1 accuracy, Top-k accuracy, and Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    top1_hits = 0
    topk_hits = 0

    for item in query_items:
        scores = score_fn(item["question"])
        ranked_indices = np.argsort(-scores)  # highest score first
        true_index = item["kb_index"]

        rank = int(np.where(ranked_indices == true_index)[0][0]) + 1  # 1-based rank
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
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    # The knowledge base is all 430 answers. We tag every item with its
    # position in this combined list (kb_index) so we know, for any query,
    # which KB row is the actually-correct answer.
    kb_items = train + val + test
    for i, item in enumerate(kb_items):
        item["kb_index"] = i
    kb_answers = [item["answer"] for item in kb_items]

    # train/val/test occupy fixed, known slices of kb_items in that order,
    # so we can pull the val/test query sets back out by position rather
    # than re-matching them by content.
    val_items = kb_items[len(train):len(train) + len(val)]
    test_items = kb_items[len(train) + len(val):]

    results = {}

    # --- Experiment 1: TF-IDF ---
    tfidf = TfidfVectorizer()
    kb_tfidf = tfidf.fit_transform(kb_answers)

    def tfidf_score(question):
        q_vec = tfidf.transform([question])
        return cosine_similarity(q_vec, kb_tfidf)[0]

    results["tfidf"] = {
        "val": evaluate_ranking(val_items, kb_answers, tfidf_score),
        "test": evaluate_ranking(test_items, kb_answers, tfidf_score),
    }

    # --- Experiment 2: BM25 ---
    tokenized_kb = [simple_tokenize(a) for a in kb_answers]
    bm25 = BM25Okapi(tokenized_kb)

    def bm25_score(question):
        return np.array(bm25.get_scores(simple_tokenize(question)))

    results["bm25"] = {
        "val": evaluate_ranking(val_items, kb_answers, bm25_score),
        "test": evaluate_ranking(test_items, kb_answers, bm25_score),
    }

    # --- Experiment 3: pretrained sentence embeddings, no fine-tuning ---
    embedder = SentenceTransformer(PRETRAINED_MODEL_NAME)
    kb_embeddings = embedder.encode(kb_answers, normalize_embeddings=True)

    def embedding_score(question):
        q_embedding = embedder.encode([question], normalize_embeddings=True)
        return cosine_similarity(q_embedding, kb_embeddings)[0]

    results["pretrained_embeddings"] = {
        "val": evaluate_ranking(val_items, kb_answers, embedding_score),
        "test": evaluate_ranking(test_items, kb_answers, embedding_score),
    }

    # Merge with whatever's already in the results file instead of overwriting
    # it outright - evaluate_finetuned_retriever.py adds its own
    # "finetuned_embeddings" entry to this same file, and a blind overwrite
    # here would destroy that if this script runs afterward.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            existing_results = json.load(f)
    existing_results.update(results)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=2)

    print(f"{'Method':<22} {'Split':<6} {'Top-1':<8} {'Top-3':<8} {'MRR':<8}")
    for method, splits in results.items():
        for split_name, metrics in splits.items():
            print(f"{method:<22} {split_name:<6} {metrics['top_1']:<8} {metrics['top_3']:<8} {metrics['mrr']:<8}")


if __name__ == "__main__":
    main()
