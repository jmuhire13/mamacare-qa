"""
Retrains the retriever with the two hand-written-loop settings that
finetune_retriever.py genuinely omits: learning-rate warmup and gradient
clipping. (Weight decay is not omitted: torch.optim.AdamW defaults to a
weight decay of 0.01 whether or not the hand-written loop sets it, which is
the same default the library's fit method uses, so there is nothing to
ablate there.)

Gradient clipping is added with torch.nn.utils.clip_grad_norm_(max_norm=1),
matching fit()'s default max_grad_norm. Warmup is added with a linear
warmup-then-decay schedule (transformers.get_linear_schedule_with_warmup),
matching fit()'s default scheduler shape (WarmupLinear).

The number of warmup steps needs a choice. fit()'s own default is 10000
warmup steps, but this training run only has 176 total optimizer steps (344
training pairs, batch size 16, 8 epochs), so that literal default would keep
the learning rate well below its target value for the entire run. Two
variants are trained so both readings are on record:

  literal_default: warmup_steps = 10000, fit()'s own number, unsuited to a
    run this short.
  scaled: warmup_steps = round(0.1 * total_steps), a conventional warmup
    fraction (about 10 percent of training) sized to this run.

Saves each model to a throwaway folder under models/ablation_checks/ and
never touches models/finetuned-retriever/. Evaluated the same way
evaluate_finetuned_retriever.py evaluates the shipped model.

Run with: python src/ablation_training_settings.py literal_default
          python src/ablation_training_settings.py scaled
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from sklearn.metrics.pairwise import cosine_similarity
from transformers import get_linear_schedule_with_warmup

DATA_DIR = Path("data/processed")
BASE_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
SEED = 42
MAX_GRAD_NORM = 1.0

VARIANT = sys.argv[1]
assert VARIANT in ("literal_default", "scaled"), "variant must be literal_default or scaled"
OUTPUT_DIR = Path("models/ablation_checks") / VARIANT


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

    batches_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = batches_per_epoch * EPOCHS
    warmup_steps = 10000 if VARIANT == "literal_default" else round(0.1 * total_steps)
    print(f"variant={VARIANT} total_steps={total_steps} warmup_steps={warmup_steps}", flush=True)

    model = SentenceTransformer(BASE_MODEL_NAME)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        last_lr = scheduler.get_last_lr()[0]
        print(f"variant={VARIANT} epoch {epoch}/{EPOCHS}, average loss: {total_loss / n_batches:.4f}, lr: {last_lr:.2e}", flush=True)

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

    print(f"RESULT variant={VARIANT} val={val_result}", flush=True)
    print(f"RESULT variant={VARIANT} test={test_result}", flush=True)

    Path("results").mkdir(exist_ok=True)
    summary_path = Path("results") / f"ablation_{VARIANT}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "variant": VARIANT,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "max_grad_norm": MAX_GRAD_NORM,
            "val": val_result,
            "test": test_result,
        }, f, indent=2)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
