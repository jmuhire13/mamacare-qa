# mamacare-qa: Maternal Health Q&A Assistant

mamacare-qa answers English questions about maternal and newborn health. It returns the stored answer closest to the question, and it declines to answer when no stored answer is close enough. The system chooses only from 430 medically reviewed question-answer pairs, so every answer it gives is one of those pairs.

Live app: [https://mamacare-app-4dakesxahaqvamub4rjcbk.streamlit.app/](https://mamacare-app-4dakesxahaqvamub4rjcbk.streamlit.app/)

Demo video: to be added

## Why this design

Expectant mothers and newborn caregivers often need a quick answer when a clinician is not available. A search engine returns generic results for health questions, and a language model can state a wrong answer with confidence. This project treats refusal as a required behavior. An answer that says the assistant does not know is preferable to a confident wrong answer, so the system measures both how often it answers correctly and how often it refuses off-topic questions.

## Data

The dataset is the MOTHER dataset (Eyobu et al., 2025), with answers that the source article states were provided and validated by professional medical personnel. The question-answer data are deposited in the Harvard Dataverse at [https://doi.org/10.7910/DVN/EZLCH3](https://doi.org/10.7910/DVN/EZLCH3), whose record lists the rights as CC0 1.0. The article is published under CC BY 4.0.

The released file contains 501 pairs. The article reports 503, and the reason for the difference was not determined. Cleaning removed 67 rows whose question contains no question mark and does not begin with a question word, 2 rows whose answer ends with a colon, and 3 duplicate questions. One row meets two rules, so 71 distinct rows are removed and 430 remain. The rules were not checked for false removals.

Questions with a TF-IDF cosine similarity of at least 0.65 were grouped, which produced 25 links forming 19 groups of near-duplicate questions. Each group was assigned to one split as a whole. The 430 pairs were split 344, 43, and 43 into training, validation, and test sets, using a random seed of 42.

Some limits follow from the data. The dataset is small, and two sub-topics, changes in relationship dynamics during pregnancy and fluctuations in sexual desire, appear in the raw file only as statements, so the cleaning rules removed them. The answers for starting a hormonal IUD, a copper IUD, or a contraceptive implant after childbirth differ only in the method name, so a retrieval result that returns one of them for another is scored as wrong.

## Method

Three baseline retrieval methods were compared over all 430 answers: TF-IDF (scikit-learn defaults), BM25 (rank-bm25 with k1 = 1.5 and b = 0.75), and the pretrained `all-MiniLM-L6-v2` sentence-embedding model without training.

The fine-tuned retriever starts from `all-MiniLM-L6-v2` and is trained on the 344 training pairs with `MultipleNegativesRankingLoss` from sentence-transformers 6.1.0, which treats the other answers in each batch as negative examples. Training uses batches of 16, 8 epochs, a learning rate of 2e-5, AdamW, and seed 42. The training loop is written by hand. Loading the `datasets` package together with torch and scikit-learn crashed Python on the development machine, and the library's `fit()` method loads that package. The hand-written loop has no learning-rate warmup, weight decay, or gradient clipping, which the default `fit()` applies.

The refusal gate decides whether to answer. It uses the pretrained `all-MiniLM-L6-v2` model, not the fine-tuned one. The gate takes the highest cosine similarity between the question and any stored answer, and it answers only when that value is at least the threshold. The threshold is 0.495. It was chosen by a grid search over 201 values from 0 to 1, using the validation questions and 76 off-topic calibration questions. The calibration questions are 18 hand-written probes and 58 questions from a set of 117 out-of-domain questions, which was split into calibration and held-out halves of 58 and 59. The held-out questions were used only to report the final refusal rate.

The generation alternative uses Qwen2.5-0.5B-Instruct with the top three passages from the fine-tuned retriever in the prompt, greedy decoding, and at most 150 new tokens. A second version adds LoRA adapters (rank 8, alpha 16, dropout 0.05) to the q, k, v, and o projections and trains them for 3 epochs at a learning rate of 1e-4. The adapters add 1,081,344 trainable parameters, about 0.22 percent of the model.

## Results

On the 43 test questions, with all 430 answers as candidates:

| Method | Top-1 (95% Wilson CI) | Top-3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| TF-IDF | 30/43 = 0.698 [0.549, 0.814] | 0.767 | 0.791 | 0.747 |
| BM25 | 29/43 = 0.674 [0.525, 0.795] | 0.744 | 0.814 | 0.733 |
| Pretrained embeddings | 34/43 = 0.791 [0.648, 0.886] | 0.884 | 0.930 | 0.853 |
| Fine-tuned embeddings | 36/43 = 0.837 [0.700, 0.919] | 0.953 | 0.977 | 0.901 |

Fine-tuning fixed three questions that the pretrained model got wrong and broke one that it got right. The two-sided exact sign test on those four disagreeing questions gives p = 0.625, so the difference is not distinguishable from chance on this test set. Retraining with seeds 1 and 7 gave 36/43 and 35/43 on the test set, so the improvement over the pretrained model held in every seed tried, but its size changes by a question or two between seeds. Disabling BM25's length normalization (b = 0) lowered its Top-1 to 25/43.

For the refusal gate on the 43 in-domain test questions, 36 were answered correctly (83.7%, 95% CI 70.0% to 91.9%), 4 were answered wrongly, none were refused when a correct answer existed, and 3 were refused where the retrieved answer would have been wrong. The four wrong answers all passed the gate. Each retrieved a question on the same topic as the true one, for example "How can you recognize intestinal cramps?" for a question about sharp stomach pains. On the 59 held-out off-topic questions, the gate refused 53 (89.8%, 95% CI 79.5% to 95.3%).

Three gate designs were compared by AUROC on the test and held-out questions together. The single fine-tuned signal scored 0.964 (bootstrap 95% interval 0.927 to 0.991), the two-signal design scored 0.918 (0.861 to 0.963), and the shipped hybrid design with the pretrained gate scored 0.984 (0.963 to 0.998). These intervals use 2,000 bootstrap resamples with seed 42. The test and held-out questions were used both to compare designs and to report results, so these numbers are not from a fully unseen set.

A cross-encoder reranker, `cross-encoder/ms-marco-MiniLM-L-6-v2`, was tested on the top ten candidates from the fine-tuned retriever. It reduced in-domain Top-1 from 36/43 to 34/43, fixing two questions and breaking four. The sign test on those six questions gives p = 0.69. The shipped application does not use the reranker.

Generation results on the same 43 test questions:

| System | ROUGE-L | BERTScore F1 |
| --- | --- | --- |
| Retrieval only (top-1 passage as the answer) | 0.864 | 0.977 |
| Qwen2.5-0.5B-Instruct, zero-shot, top three passages | 0.437 | 0.909 |
| Qwen2.5-0.5B-Instruct with LoRA, top three passages | 0.831 | 0.972 |

The LoRA model scored below the retrieval-only baseline on both metrics. It copied a retrieved passage word for word in 40 of 43 answers, and 39 of those 40 copies were the first passage in the prompt. The fine-tuned retriever ranks the gold answer in the top three for all 344 training questions and first for 327. Training on prompts built this way taught the model to copy the first passage, which is correct when the retriever ranks well and wrong otherwise.

The zero-shot answers were read by hand and labeled once: 31 of 43 are faithful to at least one retrieved passage (72.1%), 5 add specific detail that no passage supports, and 7 contradict the passages or invent content. These labels are not in the repository, because they were assigned by hand and no script produces them.

## Deployment

The app runs on Streamlit Community Cloud. The cloud service builds it from the `main` branch of this repository and installs `requirements.txt`. The fine-tuned retriever is not committed to git, so the app loads `jmuhire13/mamacare-qa-retriever` from the Hugging Face Hub at startup. The Hub download does not pin a revision, so a change to that Hub repository changes the deployed model. The app shows the matched knowledge-base question above each answer, so a user can see which stored question the system matched.

The deployed app is retrieval-only and does not generate text. The generation experiments are documented here but are not part of the app.

## Reproducing the results

Run these commands from the project root, in this order. The order matters because each step reads files that an earlier step creates. The data preparation step creates the splits that every later step uses. The fine-tuned retriever must exist before the threshold, generation, and reranker steps, and the held-out split must exist before the threshold is calibrated and tested.

```bash
# 1. Install the pinned dependencies
pip install -r requirements.txt

# 2. Clean the raw data and create the train, validation, and test splits
python src/data_prep.py
python tests/test_data_prep.py

# 3. Score the three baseline retrievers on the validation and test splits
python src/retrieval_baselines.py
python tests/test_retrieval_baselines.py

# 4. Fine-tune the retriever on the training split, then score it
python src/finetune_retriever.py
python src/evaluate_finetuned_retriever.py
python tests/test_finetuned_retriever.py

# 5. Split the 117 out-of-domain questions into calibration and held-out halves
python src/split_out_of_domain_set.py

# 6. Choose the refusal threshold on validation and calibration data only
python src/tune_threshold.py

# 7. Measure the refusal rate on the held-out questions and the in-domain accuracy on the test split
python src/out_of_domain_test.py
python tests/test_out_of_domain.py

# 8. Compute the paired comparison, Recall@5, confidence intervals, sign test, and AUROC interval
python src/compute_report_metrics.py

# 9. Compare the three refusal designs by AUROC and held-out refusal rate
python src/finalize_thresholds.py

# 10. Test the cross-encoder reranker on the retrieval errors (not used by the app)
python src/rerank_experiment.py

# 11. Repeat the fine-tuning with other seeds to check training stability (run once per seed)
python src/check_seed_variance.py 1
python src/check_seed_variance.py 7

# 12. Generation experiments (not used by the app)
python src/generate_zeroshot.py
python src/evaluate_retrieval_only_baseline.py
python src/finetune_generator.py
python src/evaluate_generator.py
python tests/test_generation.py
python src/analyze_generation_outputs.py

# 13. Draw the three report figures from the saved results (written to results/)
python src/make_figures.py

# 14. Run the app locally
streamlit run app.py
```

Step 11 retrains the retriever from scratch for each seed and saves the models under `models/seed_checks/`, which is excluded from git. The shipped model in `models/finetuned-retriever/` is not changed by this step. Step 12 needs the fine-tuned retriever from step 4, and the analysis in its last command needs the LoRA results that the evaluation writes.

Each test script prints the checks it runs and their results.

## Repository layout

```text
app.py                      Streamlit application
requirements.txt            Pinned dependencies
data/raw/                   Raw MOTHER files
data/processed/             Splits and results (only the four split and threshold files are committed)
src/                        Data preparation, retrieval, refusal, generation, and reporting scripts
tests/                      Check scripts for each step of the pipeline
```

## Limitations

The test split has 43 questions and the held-out split has 59, so the confidence intervals are wide. The test and held-out questions were looked at during development, including while comparing refusal designs. The threshold was chosen on validation and calibration data only. The faithfulness labels came from one reader and one pass. The stored answers reflect one dataset, and this project did not assess their clinical accuracy. The system is a research prototype, and it is not a substitute for medical advice. The app states this on its page.
