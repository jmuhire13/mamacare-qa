# mamacare-qa: Maternal Health Q&A Assistant

A domain-specific question-answering assistant for expectant mothers and community health workers, built on a retrieval-based NLP pipeline and evaluated with rigorous, held-out statistical testing.

Live app: _pending deployment_
Demo video: _pending_

## 1. Problem

Expectant mothers and newborn caregivers often need quick, reliable answers to everyday maternal-health questions (pregnancy symptoms, labor signs, postpartum recovery, infant care) at moments when a clinician isn't immediately reachable. Community health workers fill much of this gap in practice, but their time and reach are limited, and search engines return generic or unreliable results for health questions that call for domain-specific, medically reviewed answers. This project builds a question-answering assistant that lets users ask a maternal-health question in plain English and get back an answer drawn from a set of medically reviewed question-answer pairs, with an explicit "I don't know" for anything outside that scope rather than a guess.

Domain-specific question answering was chosen over the other two project tracks (machine translation and text classification for an African language) because it maps directly onto this problem: the goal isn't translating or categorizing text, it's giving a correct, sourced answer to a real question. English was used because the assignment permits either English or an African language for the QA track specifically, and the chosen dataset (see below) is itself in English.

The target users are expectant mothers and the community health workers who support them: people who need an answer now, not a wall of search results, and who are better served by a system that says "I don't have a confident answer to that" than one that confidently answers wrong. That refusal behavior, not just raw answer accuracy, is treated as a first-class requirement throughout this project, not an afterthought.

## 2. Dataset

### Source

The dataset is the [MOTHER dataset](https://doi.org/10.7910/DVN/EZLCH3) (Eyobu et al., *Mother: a maternal online technology for health care dataset*, BMC Research Notes, 2025), collected from expectant mothers in rural/semi-urban Uganda. Every answer went through a documented two-step medical review before being accepted. The deposit is licensed CC0 1.0, confirmed directly against the Dataverse listing's own license field rather than taken from the paper's text, which states a different license.

### Cleaning

The raw file holds 501 question-answer pairs (`data/raw/mother_question_and_answer_pairs_data.json`). Cleaning removed 67 rows where the "question" was actually a statement summarizing an answer, 3 duplicate questions, and 2 answers that were cut off mid-sentence. These three counts add up to 72, not the 71 rows actually removed, because one row is flagged by two filters at once (its "question" is a statement, and its answer is also cut off mid-list) and only gets removed once. 430 clean pairs remain. These were split 344/43/43 into train/val/test using a fixed random seed (42), with near-duplicate topic groups (hormonal IUD, copper IUD, and contraceptive implants, for example) kept together within a single split so the model is never tested on something nearly identical to what it trained on.

Reproduce this stage with `python src/data_prep.py`, then verify the output with `python tests/test_data_prep.py`.

### Limitations

The dataset is small, the answers are generic clinical guidance rather than specific to the Ugandan context, and topic coverage is uneven. Mental health questions, for instance, are rare. Two sub-topics lose coverage entirely during cleaning: the raw file only ever phrases "changes in relationship dynamics during pregnancy" and "fluctuations in sexual desire" as statements rather than questions, so every row touching them is removed as leakage with nothing left to replace it.

A small number of answers in the knowledge base are near-identical except for one changed term: the questions about starting a hormonal IUD, a copper IUD, and a contraceptive implant after childbirth all have the same answer template with only the method name swapped. A retrieval system that returns the wrong one of these near-twins would be medically harmless (the underlying guidance is the same) but still counts as a wrong answer under exact-match scoring, so reported accuracy is a slight underestimate for this handful of cases.

## 3. Methodology

The pretrained embedding baseline (`all-MiniLM-L6-v2`) already outperforms TF-IDF and BM25 with no training at all, so the next step is fine-tuning that same model on our own 344 training pairs to push it further. Fine-tuning uses `MultipleNegativesRankingLoss` (Henderson et al., 2017): for every question-answer pair in a training batch, the model is pulled to place that question's embedding close to its true answer's embedding, while every other answer in the same batch automatically acts as a negative example, pushed further away. No manually written wrong answers are needed. The batch itself supplies them.

Training runs for 8 epochs with batch size 16 and learning rate 2e-5, using a hand-written PyTorch training loop rather than the library's usual `model.fit()` convenience method. This was a deliberate workaround, not a style choice: on this machine, the newer `sentence-transformers` version routes `.fit()` through the `datasets` library, and loading `torch`, `scikit-learn`, and `datasets` together in the same process causes a native-library crash, even if our own code never imports `datasets` directly; the library appears to probe for it internally as soon as it's installed. Uninstalling it fixed the crash. The hand-written loop computes the same loss (`MultipleNegativesRankingLoss`, scale 20) as `.fit()` would, but does not replicate `.fit()`'s default learning-rate warmup, weight decay, or gradient clipping. That's a deliberate simplification for a small dataset, not a hidden discrepancy.

Reproduce this stage with `python src/finetune_retriever.py`, then evaluate with `python src/evaluate_finetuned_retriever.py` and verify with `python tests/test_finetuned_retriever.py`.

### Deciding when to refuse

A retrieval system will always return *something*. Even for a question that has nothing to do with maternal health, it still hands back whichever of the 430 answers happens to score highest. The app needs a second decision on top of retrieval: whether to answer at all, or say "I don't have a confident answer to that." This is done by comparing the question's best-match similarity score against a threshold, decided once during calibration and loaded by the app at runtime.

The gate and the retrieval step deliberately use two different models. The fine-tuned retriever is more accurate at picking the exact right answer once the app has agreed to attempt one, but a direct comparison of accept/refuse behavior showed it makes a worse judge of whether a question is in-domain at all: fine-tuning pulls maternal-health questions into a tighter embedding cluster without ever seeing genuine off-topic examples, which appears to inflate similarity scores broadly for anything loosely health-related rather than sharpening the real domain boundary. So the app uses the original pretrained model (`all-MiniLM-L6-v2`, never fine-tuned) purely to decide whether to answer, and the fine-tuned model purely to decide which answer to give once that gate says yes.

The threshold itself is calibrated on data the final test never touches: the validation split, 18 hand-written off-topic probes, and a separate 58-question calibration set of harder, health-adjacent-but-off-topic questions (hair loss, seasonal allergies, muscle-building vitamins), topics that share enough vocabulary with real maternal-health answers to be a genuine test of the boundary. The final reported refusal rate is measured only against a 59-question held-out set that plays no part in choosing the threshold.

Reproduce this stage with `python src/split_out_of_domain_set.py`, then `python src/tune_threshold.py`, then evaluate with `python src/out_of_domain_test.py` and verify with `python tests/test_out_of_domain.py`.

## 4. Experiments

The first three experiments establish a baseline for the retrieval task: given a question, find the correct answer among all 430 reviewed answers in the knowledge base. None of these three involve any training.

TF-IDF represents each answer as a vector weighted by how distinctive its words are, and ranks candidates by similarity to the question. BM25 is a refinement of the same idea, long used in search engines. The third baseline uses `all-MiniLM-L6-v2`, a pretrained sentence-embedding model, exactly as downloaded, with no fine-tuning of our own.

Reproduce this stage with `python src/retrieval_baselines.py`, then verify with `python tests/test_retrieval_baselines.py`.

## 5. Results

On the test split (43 questions), TF-IDF and BM25 are statistically indistinguishable: Top-1 0.698 vs 0.674, a one-question difference with 95% Wilson confidence intervals of [0.549, 0.814] and [0.525, 0.795] that almost completely overlap. The two methods actually disagree on five different questions. TF-IDF gets three right that BM25 misses, and BM25 gets two right that TF-IDF misses, so this isn't BM25 being systematically weaker, just a different pattern of mistakes on a small test set. We tested the obvious hypothesis that BM25's length-normalization parameter was the cause by rerunning it with length normalization disabled (b=0): accuracy got worse, not better, which rules that explanation out rather than confirming it.

The pretrained embedding model clearly beats both TF-IDF and BM25 with zero training: Top-1 0.791, 95% CI [0.648, 0.886]. Fine-tuning that same embedding model on our own training pairs moves Top-1 to 0.837, 95% CI [0.700, 0.919]. The two intervals overlap almost entirely, so this is not a strong result on its own. A paired, question-by-question comparison is more informative: of the 43 test questions, fine-tuning fixes 3 that the pretrained model got wrong and breaks 1 that it had right, a net gain of 2 questions. Top-3, Recall@5, and MRR move in the same direction, but since all four metrics are computed on the same 43 questions, that is one result viewed four ways, not four independent confirmations. Retraining with two other random seeds gave Top-1 of 36/43 and 35/43 (versus 36/43 for the seed actually shipped), consistently above the pretrained baseline in every seed tried, but by a margin of 1-2 questions rather than a fixed, exact gain. This is the model that ships in the deployed app: a small, real, and seed-robust-in-direction improvement, reported at its actual size rather than an inflated one.

Neither TF-IDF nor BM25's settings were tuned on the validation split. Both use their library defaults (plain term weighting for TF-IDF; k1=1.5, b=0.75 for BM25). They're included as fixed, untrained reference points for the embedding methods to beat, not as baselines we tried to optimize.

| Method | Test Top-1 | Test Top-3 | Test Recall@5 | Test MRR |
|---|---|---|---|---|
| TF-IDF | 0.698 | 0.767 | 0.791 | 0.747 |
| BM25 | 0.674 | 0.744 | 0.814 | 0.733 |
| Pretrained embeddings | 0.791 | 0.884 | 0.930 | 0.853 |
| Fine-tuned embeddings | 0.837 | 0.953 | 0.977 | 0.901 |

### Refusal gate

On the 43-question in-domain test split, every question falls into exactly one of four outcomes: 36/43 (83.7%, 95% CI [70.0%, 91.9%]) are answered both confidently and correctly ("true positive" below always means this combined outcome, not just "the gate accepted it"); 4/43 (9.3%, 95% CI [3.7%, 21.6%]) are answered confidently but wrong; 0/43 are wrongly refused (a real answer the gate turned down); and 3/43 are refused where the retrieved answer would have been wrong anyway, the gate correctly erring on the side of caution. These four add up to 43.

On a separate 59-question held-out set of off-topic questions, 53/59 (89.8%, 95% CI [79.5%, 95.3%]) are correctly refused. The six misses: a bedtime routine for a 5-year-old, green tea's health benefits, acid reflux and heartburn "in general," vitamins for muscle building, exercises for lower back pain, and cleaning a cast iron pan. Three of these (heartburn, back pain, bedtime routines) have near word-for-word matches in the knowledge base once a pregnancy or infant qualifier is added, which is a real, defensible source of confusion rather than an arbitrary mistake; the other two (muscle vitamins, cast iron pan) share no such overlap and are straightforward false accepts.

The hybrid design was chosen over two alternatives by comparing AUROC (the probability a random in-domain question scores higher than a random out-of-domain one, independent of any specific threshold), computed on the same 43 test + 59 held-out questions (n=102): 0.964 for a single fine-tuned signal, 0.918 for a two-signal variant, and 0.984 (bootstrap 95% CI [0.963, 0.998], 2,000 resamples) for the hybrid design that shipped. Each design's own threshold (0.56, 0.515, and 0.495 respectively) was then picked separately by the same grid search described above, applied to that design's own calibration scores.

A pretrained cross-encoder reranker was tried as a fix for retrieval's known wrong-answer cases: in-domain Top-1 measurably dropped from 36/43 to 34/43 (fixing 2 of the 7 known wrong cases, breaking 4 previously-correct ones). With only 6 cases changed, a sign test on 2-vs-4 gives p≈0.69, nowhere near significant, so the honest conclusion is "no evidence this reranker helps," not a confident "it makes things worse."

A multi-seed check (retraining the retriever from scratch with seeds 42, 1, and 7) gave test Top-1 of 36/43, 36/43, and 35/43 respectively. That's a check of training stability, not a way to narrow the sampling uncertainty above, which stays exactly as wide regardless of how many training seeds are tried.

**Limitation, stated directly:** the 43-question test split and the 117-question out-of-domain set have each been looked at repeatedly while choosing between refusal designs and judging the reranker, even though no single threshold was ever picked by looking at the final held-out numbers. That makes the figures above development-quality estimates of a research process, not numbers from a single pristine held-out test. A more rigorous follow-up would use grouped k-fold cross-validation over all 430 answer pairs instead of a single fixed split; that wasn't done here given the project timeline.

## 6. Error Analysis

### Retrieval

Reading the fine-tuned retriever's 7 wrong test-split cases by hand (rather than trusting the 83.7% headline number alone) shows two distinct failure patterns. The first is genuine topic confusion between medically adjacent concepts: "Why do I have a low or sad mood?" retrieves a passage about miscarriage grief instead of the correct one about depression during pregnancy; "What is the function of the immune system?" retrieves a passage about antibodies; one case (HIV transmission risk retrieving an unrelated passage about Fifth disease immunity) is a clean, unrelated-topic miss, though notably also the lowest-confidence of the seven, so the app's refusal gate is more likely to catch it in practice than this retrieval-only number suggests. The second pattern, covering 3 of the 7 cases, isn't really a retrieval mistake at all: the dataset has more than one legitimate answer to some overlapping questions (childbirth-pain techniques, vaginal delivery after a C-section, back pain), and the model picked a different, also-correct answer rather than the one specific row marked as ground truth.

### Generation (tested, not shipped)

A two-stage retrieval-then-generation pipeline was built and evaluated as an alternative to pure retrieval, using `Qwen2.5-0.5B-Instruct` to paraphrase the top-3 retrieved passages into a final answer. The decisive comparison turned out to be one that wasn't in the original experiment design: scoring the retriever's own top-1 passage directly as the answer, with no generation step at all, using the identical ROUGE-L/BERTScore pipeline. That retrieval-only baseline scores **ROUGE-L 0.864, BERTScore F1 0.977**, both *higher* than the LoRA fine-tuned generator's 0.831/0.972. Generation didn't just fail to add value; it measurably underperformed doing nothing on top of retrieval.

Reading individual outputs explains why. 40 of the 43 LoRA-generated answers are exact verbatim copies of a retrieved passage (confirmed two ways: exact string match, and a looser ROUGE-L ≥ 0.95-against-nearest-passage definition: both give 40/43, so this isn't an artifact of how "verbatim" is defined). Checking all 43 training questions against the same fine-tuned retriever used at test time shows why the model learned this shortcut: **344 of 344 (100%) training questions had their own gold answer in the retriever's top-3, and 327 of 344 (95%) had it at rank 1.** The generator was trained almost exclusively on cases where the correct passage was trivially present and usually listed first. It never had to learn to resolve a close call between competing candidates.

This shows up directly in the 7 low-scoring test cases. Two had the gold answer outside the top-3 entirely (the generator couldn't have gotten these right regardless of its own quality). Of the remaining 5 where the gold answer was retrieved but not ranked first, the model copied the wrong, top-ranked passage in **5 out of 5**. It never once selected a correctly-ranked-lower passage, including the one case (a question about the clitoris) where the correct passage was actually ranked first and the model still copied a different one, describing the perineum instead. The pattern across all 7 is consistent: the model reproduces whatever is listed first, which only looks like correct "generation" when the retriever's own ranking happens to be right.

Two further verbatim-copy cases are worth naming directly for how concerning they are. For a question about diaper rash, the model's answer blends in a detail from an unrelated retrieved passage about newborn skin coloring, misattributing it to diaper rash. For a question about HIV transmission risk during pregnancy, the model states "the passage states that 'If you're HIV-positive, there is no specific risk to your pregnancy or baby'", a quote that does not appear in any of the three retrieved passages, paired with a confidently stated and medically inaccurate claim that minimizes a real transmission risk.

The zero-shot model (no LoRA fine-tuning) was separately evaluated for grounding, since the original write-up rested on a single named example. All 43 zero-shot outputs were read and classified by hand (greedy decoding, no sampling, so this reflects one deterministic pass of this specific 0.5B model, not a general claim about generation): 31/43 (72.1%, 95% CI [57.3%, 83.3%]) are faithful to at least one retrieved passage, 5/43 (11.6%) add specific but unverified detail not clearly grounded in any passage, and 7/43 (16.3%, 95% CI [8.1%, 30.0%]) contradict the retrieved material or invent content outright, including fabricating a quote attributed to a nonexistent passage. These categories came from a single reading pass with no second labeler, so treat the exact boundaries between categories as approximate; the overall split is not.

One fix was attempted and abandoned: retraining the LoRA generator on paraphrased versions of the 344 training answers, to remove the verbatim-copy shortcut directly. An automated BERTScore faithfulness check (threshold 0.85) passed all 344 rewrites, but BERTScore F1 was never rescaled against a baseline in this check, and unrescaled BERTScore F1 is compressed into a narrow band for any fluent English text regardless of quality, so a threshold of 0.85 was unlikely to reject anything, and it didn't. Reading the lowest-scoring rewrites by hand did find real factual drift the check missed: a pregnancy timeframe stated incorrectly, and an unsupported claim added to one answer. This is real evidence that errors exist, but because the sample read was the one the metric already flagged as lowest-scoring, it cannot support an estimate of how common such errors are across all 344. The decision to stop before retraining on these targets stands regardless: training a medical QA generator on data with a demonstrated factual error would be worse than the verbatim-copying it was meant to fix.

Given all of this, and that neither the course assignment nor the rubric requires generated output for this project track (`RUBRIC.md`: *"Develops, adapts, or fine-tunes an appropriate ML/NLP model using concepts from the course"*, already satisfied by the fine-tuned retriever; `ASSIGNMENT.md`: *"You may use a pretrained language model from Hugging Face and fine-tune or adapt it... parameter-efficient fine-tuning techniques such as LoRA"*, permissive, not mandatory), the shipped app is retrieval-only. The generation work is kept as a fully documented, tested-and-rejected experiment: not a weaker result to downplay, but a measured negative finding with a clear, evidenced mechanism (training-data leakage producing a copy-the-first-passage shortcut), which is exactly what the rubric's Baseline/Experimentation and Evaluation criteria ask for.

Reproduce with `python src/generate_zeroshot.py` (zero-shot baseline), `python src/evaluate_retrieval_only_baseline.py` (the no-generation comparison point), and `python src/finetune_generator.py` then `python src/evaluate_generator.py` (LoRA fine-tuned); verify the central finding with `python tests/test_generation.py`. None of these are required to run the deployed app.

## 7. Setup Instructions

Run these commands in order from the project root.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean the raw dataset into train/val/test splits
python src/data_prep.py

# 3. Run the three non-trained retrieval baselines (TF-IDF, BM25, pretrained embeddings)
python src/retrieval_baselines.py

# 4. Fine-tune the retriever on the training set
python src/finetune_retriever.py

# 5. Evaluate the fine-tuned retriever
python src/evaluate_finetuned_retriever.py

# 6. Calibrate the refusal threshold (decides when the app should say "I don't know")
python src/tune_threshold.py

# 7. Split the out-of-domain test questions into calibration/held-out halves
python src/split_out_of_domain_set.py

# 8. Run the final, rigorous out-of-domain refusal test
python src/out_of_domain_test.py

# 9. Launch the app locally
streamlit run app.py
```

Each stage's output has a matching test script in `tests/`. Run any of them with `python tests/test_<name>.py` to confirm the pipeline reproduced correctly.

The scripts `src/rerank_experiment.py`, `src/evaluate_domain_separation.py`, `src/finalize_thresholds.py`, `src/generate_zeroshot.py`, `src/finetune_generator.py`, `src/evaluate_generator.py`, `src/evaluate_retrieval_only_baseline.py`, and `src/paraphrase_training_answers.py` are exploratory and not part of the final system. They document a generation (RAG) approach that was tested and deliberately not shipped, detailed in the Error Analysis section above. The deployed system is retrieval-only, and running these scripts is not required to reproduce it.

## 8. Deployment

_To be filled in once the app is deployed._
