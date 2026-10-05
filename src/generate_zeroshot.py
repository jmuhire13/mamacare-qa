"""
Zero-shot generation: can a small instruction-following language model write a
good answer just by reading the top-3 retrieved passages, with no training of
its own?

For each test question: retrieve the top-3 answers from the knowledge base
(using our fine-tuned retriever), then ask Qwen2.5-0.5B-Instruct to answer
the question using ONLY those passages, in its own words. This tests
whether a pretrained generator, with no fine-tuning, already does a
reasonable job when it's given the right source material. The baseline
for the LoRA fine-tuned model in finetune_generator.py.

Evaluated with ROUGE-L (word-overlap with the reference answer) and
BERTScore (meaning-based similarity), used together rather than either
alone: ROUGE-L catches exact wording matches, BERTScore catches correct
answers phrased differently, and each covers the other's blind spot.

Run with: python src/generate_zeroshot.py
"""

import json
from pathlib import Path

import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils import load_split, retrieve_top_k, generate_answer

RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_PATH = Path("data/processed/zeroshot_generation_results.json")
TOP_K_PASSAGES = 3


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_answers = [item["answer"] for item in kb_items]

    print("Loading retriever...")
    retriever = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    kb_embeddings = retriever.encode(kb_answers, normalize_embeddings=True)

    print("Loading generator (this downloads ~1GB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME, dtype=torch.float32)

    results = []
    for i, item in enumerate(test):
        passages = retrieve_top_k(item["question"], retriever, kb_embeddings, kb_answers, TOP_K_PASSAGES)
        generated = generate_answer(item["question"], passages, tokenizer, model)
        results.append({
            "question": item["question"],
            "reference_answer": item["answer"],
            "generated_answer": generated,
            "retrieved_passages": passages,
        })
        print(f"[{i+1}/{len(test)}] done")

    # ROUGE-L: word-overlap based similarity to the reference answer.
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [
        scorer.score(r["reference_answer"], r["generated_answer"])["rougeL"].fmeasure for r in results
    ]

    # BERTScore: meaning-based similarity, computed all at once (it's more
    # efficient in a batch than one pair at a time).
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

    print(f"\nAverage ROUGE-L: {avg_rouge:.3f}")
    print(f"Average BERTScore F1: {avg_bert:.3f}")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
