"""
Fine-tunes the generator with LoRA on the training data, to fix the grounding
failure of the zero-shot model. The zero-shot model sometimes ignores the
retrieved passages and writes generic content instead. Training on real (question, retrieved passages, correct answer)
triples should teach the model to actually use what it's given.

LoRA (Low-Rank Adaptation): instead of updating all ~500 million of the
model's parameters, LoRA freezes the original model and adds a small set
of new, trainable parameters alongside it. Only those get updated during
training. This uses far less memory and is much cheaper to train, at a
small cost in flexibility compared to fine-tuning everything.

Trains with a plain PyTorch loop (not the Hugging Face Trainer class),
for the same reason finetune_retriever.py does: Trainer routes through the
`datasets` library on this machine, which conflicts with torch+scikit-learn
and crashes.

Run with: python src/finetune_generator.py
"""

import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation_utils import load_split, retrieve_top_k, PROMPT_TEMPLATE

RETRIEVAL_MODEL_DIR = Path("models/finetuned-retriever")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = Path("models/finetuned-generator")
TOP_K_PASSAGES = 3

EPOCHS = 3
LEARNING_RATE = 1e-4
SEED = 42


def build_training_example(question, passages, answer, tokenizer):
    passages_text = "\n".join(f"- {p}" for p in passages)
    prompt = PROMPT_TEMPLATE.format(passages=passages_text, question=question)

    prompt_only_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_only_messages, tokenize=False, add_generation_prompt=True
    )

    full_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Only the assistant's answer should count toward the training loss -
    # the model shouldn't be "graded" on predicting the prompt it was given.
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    kb_items = train + val + test
    kb_answers = [item["answer"] for item in kb_items]

    print("Loading retriever to build training passages...")
    retriever = SentenceTransformer(str(RETRIEVAL_MODEL_DIR))
    kb_embeddings = retriever.encode(kb_answers, normalize_embeddings=True)

    print("Loading base generator model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building training examples (retrieving top-3 passages for each)...")
    examples = []
    for item in train:
        passages = retrieve_top_k(item["question"], retriever, kb_embeddings, kb_answers, TOP_K_PASSAGES)
        input_ids, labels = build_training_example(item["question"], passages, item["answer"], tokenizer)
        examples.append((input_ids, labels))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        shuffled = examples[:]
        random.shuffle(shuffled)

        total_loss = 0.0
        for input_ids, labels in shuffled:
            input_tensor = torch.tensor([input_ids])
            label_tensor = torch.tensor([labels])

            optimizer.zero_grad()
            outputs = model(input_ids=input_tensor, labels=label_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch}/{EPOCHS}, average loss: {total_loss / len(shuffled):.4f}")

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
