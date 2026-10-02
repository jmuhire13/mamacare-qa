"""
Scores the retriever's own top-1 passage as if it were the final answer, with
no generation step at all, using the exact same ROUGE-L/BERTScore pipeline as
generate_zeroshot.py and evaluate_generator.py. This is the comparison point
Experiments 5 and 6 were missing: without it, there was no way to tell
whether the generator added anything over simply returning what retrieval
already found.

Run with: python src/evaluate_retrieval_only_baseline.py
"""

import json
from pathlib import Path

from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from generation_utils import load_split, retrieve_top_k

RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
RESULTS_PATH = Path("data/processed/retrieval_only_generation_baseline.json")


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_answers = [item["answer"] for item in kb_items]

    retriever = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    kb_embeddings = retriever.encode(kb_answers, normalize_embeddings=True)

    results = []
    for item in test:
        top1_answer = retrieve_top_k(item["question"], retriever, kb_embeddings, kb_answers, 1)[0]
        results.append({
            "question": item["question"],
            "reference_answer": item["answer"],
            "generated_answer": top1_answer,
        })

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(r["reference_answer"], r["generated_answer"])["rougeL"].fmeasure for r in results
    ]
    references = [r["reference_answer"] for r in results]
    candidates = [r["generated_answer"] for r in results]
    _, _, bert_f1 = bert_score(candidates, references, lang="en", verbose=False)

    for r, rouge, bert in zip(results, rouge_scores, bert_f1.tolist()):
        r["rougeL"] = round(rouge, 3)
        r["bertscore_f1"] = round(bert, 3)

    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bert = sum(bert_f1.tolist()) / len(bert_f1)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"average_rougeL": round(avg_rouge, 3), "average_bertscore_f1": round(avg_bert, 3),
                   "examples": results}, f, indent=2)

    print(f"Retrieval-only (top-1 passage, no generation) - Average ROUGE-L: {avg_rouge:.3f}")
    print(f"Retrieval-only (top-1 passage, no generation) - Average BERTScore F1: {avg_bert:.3f}")
    print(f"(LoRA fine-tuned generator was ROUGE-L 0.831, BERTScore F1 0.972)")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
