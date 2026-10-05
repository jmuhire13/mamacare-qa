"""
Addresses a shortcut-learning problem in the generator: the generator
learned to copy a retrieved passage verbatim instead of genuinely writing
an answer, because every training target was character-for-character
identical to one of the inputs. The easiest possible pattern to learn.

The fix: before fine-tuning, rewrite each of the 344 training answers into
a paraphrase (same facts, different wording) using the base model's own
zero-shot ability. The zero-shot test already showed it paraphrases well without
any training (40/43 test answers were genuine paraphrases, not copies).
Training on these paraphrased targets removes the "just copy the input"
shortcut, since the target text no longer matches any retrieved passage
exactly.

Disclosure: this generates machine-written
text, but only as TRAINING TARGETS for the generator, not as the primary
dataset. The real, human-authored question/answer pairs remain the
primary data throughout, and this is a clearly labeled, secondary
transformation of them for one specific training purpose. The evaluation
ground truth (the actual TEST set references) is never touched or
paraphrased. Only the 344 TRAIN answers are rewritten, and only for
building generator training targets.

Each paraphrase is checked against its original with BERTScore and
dropped (falling back to the original answer) if it drifts too far in
meaning. A cheap safety net against the paraphrasing step accidentally
introducing factual drift into training data for a health application.

Run with: python src/paraphrase_training_answers.py
"""

import json
from pathlib import Path

import torch
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("data/processed")
GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH = Path("data/processed/train_paraphrased.json")
FAITHFULNESS_THRESHOLD = 0.85  # BERTScore F1 below this -> fall back to the original

PARAPHRASE_PROMPT = """Rewrite the following answer in different words, keeping exactly the same facts and meaning. Do not add or remove any information. Keep it roughly the same length.

Answer: {answer}

Rewritten:"""


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def paraphrase(answer, tokenizer, model):
    messages = [{"role": "user", "content": PARAPHRASE_PROMPT.format(answer=answer)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    train = load_split("train")

    print("Loading base generator model...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME, dtype=torch.float32)

    paraphrased_items = []
    for i, item in enumerate(train):
        rewritten = paraphrase(item["answer"], tokenizer, model)
        paraphrased_items.append({
            "question": item["question"],
            "original_answer": item["answer"],
            "paraphrased_answer": rewritten,
        })
        print(f"[{i+1}/{len(train)}] done")

    # Faithfulness check: compare each paraphrase to its original.
    originals = [item["original_answer"] for item in paraphrased_items]
    paraphrases = [item["paraphrased_answer"] for item in paraphrased_items]
    _, _, bert_f1 = bert_score(paraphrases, originals, lang="en", verbose=False)

    fallback_count = 0
    for item, score in zip(paraphrased_items, bert_f1.tolist()):
        item["faithfulness_bertscore"] = round(score, 3)
        if score < FAITHFULNESS_THRESHOLD:
            item["training_target"] = item["original_answer"]
            item["used_fallback"] = True
            fallback_count += 1
        else:
            item["training_target"] = item["paraphrased_answer"]
            item["used_fallback"] = False

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(paraphrased_items, f, indent=2)

    print(f"\n{len(paraphrased_items) - fallback_count}/{len(paraphrased_items)} paraphrases passed the faithfulness check")
    print(f"{fallback_count}/{len(paraphrased_items)} fell back to the original answer (paraphrase drifted too far)")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
