"""
Draws the three figures used in the report from the saved results, so that
every plotted value can be traced to a file in results/.

Figure 1: Top-1 accuracy of the four retrieval methods, with 95% Wilson
intervals, from baseline_results.json (counts recovered from the Top-1 values
over the 43 test questions).

Figure 2: ROC curves for the three refusal designs, from design_scores.json
(written by finalize_thresholds.py), scored on the 43 in-domain test questions
against the 59 held-out off-topic questions.

Figure 3: ROUGE-L for each of the 43 test questions under the three generation
systems, from the three generation result files.

Run with: python src/make_figures.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

RESULTS_DIR = Path("results")
N_TEST = 43


def wilson_interval(successes, n, z=1.96):
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return center - margin, center + margin


def figure_1_retrieval():
    baseline = json.load(open(RESULTS_DIR / "baseline_results.json", encoding="utf-8"))
    rows = [
        ("TF-IDF", baseline["tfidf"]["test"]["top_1"]),
        ("BM25", baseline["bm25"]["test"]["top_1"]),
        ("Pretrained\nembeddings", baseline["pretrained_embeddings"]["test"]["top_1"]),
        ("Fine-tuned\nembeddings", baseline["finetuned_embeddings"]["test"]["top_1"]),
    ]
    names = [name for name, _ in rows]
    counts = [round(value * N_TEST) for _, value in rows]
    rates = [c / N_TEST for c in counts]
    intervals = [wilson_interval(c, N_TEST) for c in counts]
    lower = [r - lo for r, (lo, _) in zip(rates, intervals)]
    upper = [hi - r for r, (_, hi) in zip(rates, intervals)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names))
    ax.bar(x, rates, color="#4C72B0", width=0.55)
    ax.errorbar(x, rates, yerr=[lower, upper], fmt="none", ecolor="black", capsize=5)
    for xi, c, r in zip(x, counts, rates):
        ax.text(xi, r + 0.02, f"{c}/{N_TEST}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Top-1 accuracy on the 43 test questions (95% Wilson intervals)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure_1_retrieval_top1.png", dpi=200)
    plt.close(fig)


def figure_2_refusal_designs():
    scores = json.load(open(RESULTS_DIR / "design_scores.json", encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, entry in scores.items():
        y_true = [1] * len(entry["test_scores"]) + [0] * len(entry["heldout_scores"])
        y_score = entry["test_scores"] + entry["heldout_scores"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate (off-topic questions answered)")
    ax.set_ylabel("True positive rate (in-domain questions answered)")
    ax.set_title("ROC curves for the three refusal designs")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure_2_refusal_roc.png", dpi=200)
    plt.close(fig)


def figure_3_generation():
    systems = [
        ("Retrieval only", "retrieval_only_generation_baseline.json"),
        ("Zero-shot", "zeroshot_generation_results.json"),
        ("LoRA", "lora_generation_results.json"),
    ]
    data = []
    for _, filename in systems:
        examples = json.load(open(RESULTS_DIR / filename, encoding="utf-8"))["examples"]
        data.append([e["rougeL"] for e in examples])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.RandomState(42)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.6, s=18)
        ax.hlines(np.mean(values), i - 0.25, i + 0.25, colors="black")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([name for name, _ in systems])
    ax.set_ylabel("ROUGE-L per test question")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("ROUGE-L for the 43 test questions (black line: mean)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure_3_generation_rougeL.png", dpi=200)
    plt.close(fig)


def main():
    figure_1_retrieval()
    figure_2_refusal_designs()
    figure_3_generation()
    print("Saved figure_1_retrieval_top1.png, figure_2_refusal_roc.png, figure_3_generation_rougeL.png to results/")


if __name__ == "__main__":
    main()
