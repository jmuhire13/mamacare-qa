"""
Experiment 4 evaluation: score the fine-tuned retriever (from
finetune_retriever.py) the exact same way the three baselines were scored,
so the comparison between them is fair.

Run with: python src/evaluate_finetuned_retriever.py
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from retrieval_baselines import load_split, evaluate_ranking

MODEL_DIR = Path("models/finetuned-retriever")
RESULTS_PATH = Path("data/processed/baseline_results.json")


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    kb_items = train + val + test
    for i, item in enumerate(kb_items):
        item["kb_index"] = i
    kb_answers = [item["answer"] for item in kb_items]
    val_items = kb_items[len(train):len(train) + len(val)]
    test_items = kb_items[len(train) + len(val):]

    model = SentenceTransformer(str(MODEL_DIR))
    kb_embeddings = model.encode(kb_answers, normalize_embeddings=True)

    def embedding_score(question):
        q_embedding = model.encode([question], normalize_embeddings=True)
        return cosine_similarity(q_embedding, kb_embeddings)[0]

    result = {
        "val": evaluate_ranking(val_items, kb_answers, embedding_score),
        "test": evaluate_ranking(test_items, kb_answers, embedding_score),
    }

    with open(RESULTS_PATH, encoding="utf-8") as f:
        all_results = json.load(f)
    all_results["finetuned_embeddings"] = result
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"{'Split':<6} {'Top-1':<8} {'Top-3':<8} {'MRR':<8}")
    for split_name, metrics in result.items():
        print(f"{split_name:<6} {metrics['top_1']:<8} {metrics['top_3']:<8} {metrics['mrr']:<8}")


if __name__ == "__main__":
    main()
