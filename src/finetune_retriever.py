"""
Experiment 4: fine-tune a sentence-embedding model on our own train pairs,
instead of using it exactly as downloaded (that was Experiment 3).

How the fine-tuning works: for every (question, answer) pair in the training
set, we tell the model "these two belong together, pull their embeddings
closer." Every other answer in the same training batch acts as an automatic
"these do NOT belong together, push them apart" example - this is called
MultipleNegativesRankingLoss, and it needs no manually written wrong
answers, just batches of real pairs.

This trains with a plain PyTorch loop instead of the library's usual
convenience method (model.fit()), because on this machine the newer
sentence-transformers version routes .fit() through the `datasets` library,
and loading torch + scikit-learn + datasets together crashes Python outright
(a native-library conflict, confirmed by isolating each import separately -
see docs/WORK_LOG.md). Writing the training step by hand avoids the `datasets`
import entirely while doing exactly the same computation.

Output: a fine-tuned model saved to models/finetuned-retriever/ (gitignored;
this gets pushed to Hugging Face Hub later instead of committed to GitHub).

Run with: python src/finetune_retriever.py
"""

import json
import random
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, losses

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/finetuned-retriever")
BASE_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
SEED = 42


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train = load_split("train")

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

            # Turn raw text into the tokenized tensors the model expects.
            question_features = model.tokenize(questions)
            answer_features = model.tokenize(answers)

            optimizer.zero_grad()
            loss = loss_fn([question_features, answer_features], labels=None)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch}/{EPOCHS} - average loss: {total_loss / n_batches:.4f}")

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_DIR))
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
