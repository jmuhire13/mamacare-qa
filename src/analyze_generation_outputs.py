"""
Diagnostics for the generation experiments, computed from the saved outputs
and the fine-tuned retriever: where the correct answer ranks among the 430
candidates for the training questions and the test questions, which passage a
generated answer copies, and how each low-scoring test case relates to the
retrieved passages.

Run with: python src/analyze_generation_outputs.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/processed")
RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
LOW_SCORE_THRESHOLD = 0.3


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    for i, item in enumerate(kb_items):
        item["kb_index"] = i
    kb_answers = [item["answer"] for item in kb_items]

    retriever = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    kb_emb = retriever.encode(kb_answers, normalize_embeddings=True)

    def ranks_for(questions, gold_indices):
        emb = retriever.encode(questions, normalize_embeddings=True)
        sims = cosine_similarity(emb, kb_emb)
        out = []
        for i, gold in enumerate(gold_indices):
            ranked = np.argsort(-sims[i])
            out.append(int(np.where(ranked == gold)[0][0]) + 1)
        return np.array(out)

    train_ranks = ranks_for([x["question"] for x in train], [x["kb_index"] for x in train])
    print(f"Training questions: gold answer in top 3: {(train_ranks <= 3).sum()}/{len(train_ranks)}; "
          f"at rank 1: {(train_ranks == 1).sum()}/{len(train_ranks)}")

    lora = json.load(open(DATA_DIR / "lora_generation_results.json", encoding="utf-8"))["examples"]
    test_gold = [x["kb_index"] for x in kb_items[len(train) + len(val):]]
    test_ranks = ranks_for([e["question"] for e in lora], test_gold)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    exact = 0
    near = 0
    copy_position = Counter()
    for e in lora:
        gen = e["generated_answer"].strip()
        passages = [p.strip() for p in e["retrieved_passages"]]
        if gen in passages:
            exact += 1
            copy_position[passages.index(gen) + 1] += 1
        nearest = max(scorer.score(p, gen)["rougeL"].fmeasure for p in passages)
        if nearest >= 0.95:
            near += 1
    print(f"Exact verbatim copies of a retrieved passage: {exact}/{len(lora)}")
    print(f"Near-verbatim copies (ROUGE-L of 0.95 or more against the nearest passage): {near}/{len(lora)}")
    print(f"Passage position copied in exact copies: {dict(sorted(copy_position.items()))}")

    print()
    print("Low-scoring test cases (LoRA ROUGE-L below 0.3):")
    groups = Counter()
    for e, rank in zip(lora, test_ranks):
        if e["rougeL"] >= LOW_SCORE_THRESHOLD:
            continue
        gen = e["generated_answer"].strip()
        passages = [p.strip() for p in e["retrieved_passages"]]
        matched = passages.index(gen) + 1 if gen in passages else None
        if rank > 3:
            group = "gold outside top 3"
        elif rank > 1:
            group = "gold in top 3, not first"
        else:
            group = "gold first, different passage copied"
        groups[group] += 1
        print(f"  rank of gold={rank}, copied position={matched}, ROUGE-L={e['rougeL']}: {e['question'][:70]}")
    print("Groups:", dict(groups))

    second_or_third = [i for i, r in enumerate(test_ranks) if r in (2, 3)]
    copied_first = sum(1 for i in second_or_third
                       if lora[i]["generated_answer"].strip()
                       == [p.strip() for p in lora[i]["retrieved_passages"]][0])
    print(f"Test cases where gold ranks 2nd or 3rd: {len(second_or_third)}; "
          f"model copied the first passage in {copied_first} of them")


if __name__ == "__main__":
    main()
