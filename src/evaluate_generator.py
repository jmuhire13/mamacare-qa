"""
Evaluates the LoRA-fine-tuned generator on the TEST split,
never seen during training, using the same pipeline and metrics as the
zero-shot baseline (generate_zeroshot.py), for a fair
side-by-side comparison.

The training loss dropped very low (0.005) after only 3 epochs on 344
examples, which needs checking, because a model can reach that loss either by learning
the task or by memorizing the training answers in a way that does not transfer
to new questions. This script checks which of the two happened.

Run with: python src/evaluate_generator.py
"""

import json
from pathlib import Path

import torch
from bert_score import score as bert_score
from peft import PeftModel
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils import load_split, retrieve_top_k, generate_answer

RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER_DIR = Path("models/finetuned-generator")
RESULTS_PATH = Path("data/processed/lora_generation_results.json")
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

    print("Loading base generator + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(LORA_ADAPTER_DIR))
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(LORA_ADAPTER_DIR))
    model.eval()

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

    print(f"\nLoRA fine-tuned, average ROUGE-L: {avg_rouge:.3f}")
    print(f"LoRA fine-tuned, average BERTScore F1: {avg_bert:.3f}")
    print(f"(Zero-shot baseline was ROUGE-L 0.437, BERTScore F1 0.909)")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
