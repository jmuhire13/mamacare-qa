"""
Shared helpers for the generation experiments (generate_zeroshot.py,
finetune_generator.py, evaluate_generator.py). These three scripts each
had their own copy of the same retrieval-and-prompting logic; pulled out
here once so there's a single place to read or change it, instead of three
copies that could silently drift apart.
"""

import json
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("data/processed")

PROMPT_TEMPLATE = """Answer the question using ONLY the information in the passages below. Be concise and stay factual to the passages - do not add anything not supported by them.

Passages:
{passages}

Question: {question}

Answer:"""


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_k(question, retriever, kb_embeddings, kb_answers, k):
    q_embedding = retriever.encode([question], normalize_embeddings=True)
    sims = cosine_similarity(q_embedding, kb_embeddings)[0]
    top_indices = sims.argsort()[-k:][::-1]
    return [kb_answers[i] for i in top_indices]


def generate_answer(question, passages, tokenizer, model):
    passages_text = "\n".join(f"- {p}" for p in passages)
    prompt = PROMPT_TEMPLATE.format(passages=passages_text, question=question)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()
