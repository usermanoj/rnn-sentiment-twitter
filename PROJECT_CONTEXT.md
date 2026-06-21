# RNN Twitter Sentiment Analysis - Project Context

## Programme And Assignment

This project is for Week 31 of the Advanced Certificate Programme in Applied Artificial Intelligence and Machine Learning. The graded task is to perform sentiment analysis on Twitter posts using a Recurrent Neural Network (RNN), with the goal of classifying tweets as positive, negative, or neutral.

The assignment requires:

- Loading the Twitter dataset into a structured format.
- Cleaning tweet text and handling data quality issues.
- Conducting exploratory data analysis (EDA).
- Creating numerical text features and token sequences.
- Building an LSTM or GRU model with an embedding layer.
- Training, evaluating, and improving the model.
- Documenting the process and preparing presentation-ready findings.

## Source Files

The original files in the workspace are:

- `IITM_Pravartak_Week 31_Graded Mini Project.pdf`
- `Week 31_Task to be performed.pdf`
- `twitter_training.csv`

The CSV has no header row. It should be loaded with the following column names:

```text
tweet_id, entity, sentiment, tweet_text
```

## Initial Dataset Profile

Read-only profiling of `twitter_training.csv` found:

```text
Rows: 74,682
Columns: 4
Missing tweet_text values: 686
Blank or missing tweet_text rows: 858
Exact duplicate rows: 2,700
Unique tweet IDs: 12,447
Unique entities/topics: 32
Unique sentiment labels: 4
```

Original sentiment distribution:

```text
Negative:   22,542
Positive:   20,832
Neutral:    18,318
Irrelevant: 12,990
```

## Important Modeling Decision

The assignment objective and rubric describe a three-class task: positive, negative, and neutral sentiment classification. The dataset, however, includes a fourth label: `Irrelevant`.

Recommended decision for the main graded model:

1. Document the presence of `Irrelevant`.
2. Exclude `Irrelevant` from the main classifier.
3. Build a three-class model for `Negative`, `Neutral`, and `Positive`.

Rationale: `Irrelevant` is not equivalent to neutral sentiment. Merging it into `Neutral` would introduce label noise and weaken alignment with the assignment goal.

After dropping blank or missing tweet text, exact duplicate rows, and excluding `Irrelevant`, the expected modeling dataset is:

```text
Rows: 58,980
Negative: 21,652
Positive: 19,677
Neutral:  17,651
```

## Phase 2 Status - Data Loading And Validation

Phase 2 has been implemented on branch `phase/02-data-loading-validation`.

Completed work:

- Loaded `twitter_training.csv` with explicit no-header column names.
- Validated schema, data types, missing values, blank tweet text, duplicate rows, sentiment labels, text lengths, and entity distribution.
- Confirmed the raw data contains the extra label `Irrelevant`.
- Generated audit tables under `outputs/tables`.
- Generated a short validation summary under `outputs/reports/phase2_validation_summary.md`.

No data cleaning has been applied in Phase 2. Cleaning, label filtering, and text normalization are intentionally deferred to Phase 3.

## Phase 3 Status - Data Cleaning

Phase 3 has been implemented on branch `phase/03-data-cleaning`.

Completed work:

- Dropped blank or missing tweet text rows.
- Removed exact duplicate rows after blank-text removal.
- Excluded `Irrelevant` to align with the three-class assignment objective.
- Cleaned tweet text by lowercasing and removing URLs, mentions, hashtags, special characters, and extra whitespace.
- Tokenized cleaned text.
- Removed stop words while preserving negation words.
- Applied stemming, using NLTK Porter stemming when NLTK is installed and a deterministic fallback suffix stemmer otherwise.
- Created two cleaned text fields:
  - `model_text` for RNN sequence modeling.
  - `processed_text` for rubric-visible stop-word removal, stemming, top-word analysis, and word clouds.
- Saved the cleaned dataset under `outputs/data/twitter_training_cleaned_phase3.csv`.
- Generated Phase 3 audit tables under `outputs/tables`.
- Generated a short cleaning summary under `outputs/reports/phase3_cleaning_summary.md`.

Phase 3 row counts from the verified run:

```text
Raw rows: 74,682
Removed blank or missing tweet text rows: 858
Removed exact duplicate rows after blank-text removal: 2,340
Removed Irrelevant rows: 12,504
Removed rows empty after text cleaning: 139
Final cleaned rows: 58,841
```

## Phase 4 Status - Text Preprocessing

Phase 4 has been implemented on branch `phase/04-text-preprocessing`.

Completed work:

- Loaded the cleaned Phase 3 dataset.
- Validated the remaining three assignment labels: `Negative`, `Neutral`, and `Positive`.
- Created `analysis_text`, using `processed_text` where available and falling back to `model_text` when stop-word removal and stemming left a row empty.
- Created token fields in the notebook for `model_text` and `analysis_text`.
- Saved a compact preprocessed dataset under `outputs/data/twitter_text_preprocessed_phase4.csv`.
- Created label mapping metadata for later modeling:
  - `Negative` -> `0`
  - `Neutral` -> `1`
  - `Positive` -> `2`
- Generated token length summaries and top-token frequency tables under `outputs/tables`.
- Generated a duplicate-text audit to identify repeated cleaned model inputs and conflicting cleaned-text labels before modeling.
- Generated a short preprocessing summary under `outputs/reports/phase4_text_preprocessing_summary.md`.

Phase 4 verified run:

```text
Rows preprocessed: 58,841
Empty processed_text rows handled with model_text fallback: 1,151
Empty model_text rows: 0
Empty analysis_text rows: 0
Model token count p95: 47
Model token count p99: 56
Recommended max sequence length for later padding: 60
Duplicate model_text + sentiment rows to audit before modeling: 3,851
Cleaned model_text values with multiple sentiment labels: 123
Rows in conflicting model_text groups: 1,725
```

The duplicate-text audit is intentionally not applied as a row-removal step in Phase 4. It should guide Phase 6 train/test splitting and optional text-label deduplication so repeated cleaned text does not leak across splits.

## Phase 5 Status - Exploratory Data Analysis

Phase 5 has been implemented on branch `phase/05-eda`.

Completed work:

- Loaded the Phase 4 preprocessed dataset.
- Produced basic statistics for the cleaned three-class dataset.
- Generated sentiment distribution and entity distribution tables.
- Generated tweet length summaries by sentiment.
- Generated top-token frequency tables by sentiment.
- Generated SVG visualizations under `outputs/figures`:
  - `phase5_sentiment_distribution.svg`
  - `phase5_top_entities.svg`
  - `phase5_tweet_length_by_sentiment.svg`
  - `phase5_wordcloud_negative.svg`
  - `phase5_wordcloud_positive.svg`
  - `phase5_top_tokens_by_sentiment.svg`
- Generated Phase 5 EDA tables under `outputs/tables`.
- Generated written EDA insights under `outputs/reports/phase5_eda_insights.md`.

Phase 5 verified run:

```text
Rows analyzed: 58,841
Entities: 32
Sentiments: 3
Negative: 21,605 (36.72%)
Positive: 19,644 (33.38%)
Neutral: 17,592 (29.90%)
Most frequent entity: TomClancysGhostRecon with 2,254 rows (3.83%)
Model token count p95: 47
Model token count p99: 56
Recommended initial max sequence length: 60
```

EDA carry-forward decision:

- Use stratified train, validation, and test splits because class balance is moderate but not equal.
- Report macro F1 as well as accuracy during evaluation.
- Build the model vocabulary from the training split only to avoid leakage.
- Treat 60 tokens as the initial maximum sequence length candidate.
- Use the Phase 4 duplicate-text audit before final modeling to reduce leakage risk.

## Phase 6 Status - Feature Engineering

Phase 6 has been implemented on branch `phase/06-feature-engineering`.

Completed work:

- Loaded the Phase 4 preprocessed dataset.
- Created leakage-aware train, validation, and test splits by assigning each unique `model_text` value to exactly one split.
- Preserved the Phase 4 label mapping:
  - `Negative` -> `0`
  - `Neutral` -> `1`
  - `Positive` -> `2`
- Built the RNN vocabulary from training rows only.
- Reserved special token IDs for later PyTorch embedding use:
  - `<PAD>` -> `0`
  - `<OOV>` -> `1`
- Converted each tweet into a padded integer sequence with maximum length 60.
- Saved compressed NumPy arrays for train, validation, and test splits under `outputs/data/phase6_sequences.npz`.
- Saved a row-level feature audit under `outputs/data/twitter_features_phase6.csv`.
- Saved training-derived vocabulary metadata under `outputs/data/phase6_vocabulary.json`.
- Saved TF-IDF vocabulary metadata for an optional baseline under `outputs/data/phase6_tfidf_vocabulary.csv`.
- Generated Phase 6 audit tables under `outputs/tables`.
- Generated a short feature engineering summary under `outputs/reports/phase6_feature_engineering_summary.md`.

Phase 6 verified run:

```text
Rows engineered: 58,841
Train rows: 41,189
Validation rows: 8,826
Test rows: 8,826
model_text values appearing in multiple splits: 0
Conflicting-label model_text groups kept in one split: 123
Rows in conflicting-label model_text groups: 1,725
Vocabulary size including special tokens: 17,924
Max sequence length: 60
Rows truncated at max sequence length: 213
TF-IDF metadata features: 5,000
```

Phase 6 intentionally did not train a model. The next phase should use `outputs/data/phase6_sequences.npz` to create PyTorch `Dataset` and `DataLoader` objects, then build an embedding-based LSTM or GRU with `input_dim=17,924` and `padding_idx=0`.

## Phase 7 Status - Baseline RNN Modeling

Phase 7 has been implemented on branch `phase/07-rnn-modeling`.

Completed work:

- Merged the completed Phase 6 artifacts into the Phase 7 branch.
- Loaded `outputs/data/phase6_sequences.npz` and `outputs/data/phase6_feature_config.json`.
- Created PyTorch `TensorDataset` and `DataLoader` objects for the train and validation splits.
- Built a compact embedding plus GRU sentiment classifier.
- Used class-weighted cross entropy to account for moderate class imbalance.
- Trained the model on the training split for three epochs.
- Monitored validation loss, accuracy, and macro F1 after every epoch.
- Saved the best validation checkpoint under `outputs/models/phase7_baseline_gru_state.pt`.
- Saved checkpoint metadata under `outputs/models/phase7_baseline_gru_metadata.json`.
- Saved Phase 7 history, validation metrics, confusion matrix, and model config under `outputs/tables`.
- Saved a learning curve under `outputs/figures/phase7_learning_curve.svg`.
- Generated a short modeling summary under `outputs/reports/phase7_rnn_modeling_summary.md`.

Phase 7 verified run:

```text
Runtime: PyTorch 2.10.0+cpu
Device: CPU
Train rows: 41,189
Validation rows: 8,826
Held-out test rows reserved for Phase 8: 8,826
Vocabulary size: 17,924
Model: Embedding(17,924, 64) + GRU(hidden=64) + dropout + linear classifier
Trainable parameters: 1,172,291
Epochs trained: 3
Best validation epoch: 3
Best validation loss: 0.8735
Best validation accuracy: 0.6313
Best validation macro F1: 0.6185
Held-out test split used: no
```

Phase 7 intentionally used only the train and validation splits. The next phase should load the saved checkpoint and evaluate it once on the held-out test split, reporting accuracy, precision, recall, macro F1, weighted F1, and a confusion matrix.

## Phase 8 Status - Held-Out Evaluation

Phase 8 has been implemented on branch `phase/08-evaluation`.

Completed work:

- Merged the completed Phase 7 baseline modeling artifacts into the Phase 8 branch.
- Loaded `outputs/models/phase7_baseline_gru_state.pt` and checkpoint metadata.
- Recreated the saved embedding plus GRU architecture from Phase 7 metadata.
- Loaded `X_test`, `y_test`, and `row_id_test` from `outputs/data/phase6_sequences.npz`.
- Verified every evaluated row belongs to the Phase 6 `test` split.
- Evaluated the saved Phase 7 checkpoint exactly once on the held-out test split.
- Computed test loss, accuracy, precision, recall, macro F1, weighted F1, and a confusion matrix.
- Saved full test predictions under `outputs/data/phase8_test_predictions.csv`.
- Saved Phase 8 metrics, confusion matrices, prediction distribution, and sample predictions under `outputs/tables`.
- Saved a confusion matrix figure under `outputs/figures/phase8_test_confusion_matrix.svg`.
- Generated a short evaluation summary under `outputs/reports/phase8_evaluation_summary.md`.

Phase 8 verified run:

```text
Evaluated checkpoint: outputs/models/phase7_baseline_gru_state.pt
Test rows: 8,826
Test weighted cross-entropy loss: 0.8627
Test accuracy: 0.6387
Test macro precision: 0.6467
Test macro recall: 0.6273
Test macro F1: 0.6262
Test weighted F1: 0.6313
Negative F1: 0.7012
Neutral F1: 0.5503
Positive F1: 0.6271
Most common error pattern: Neutral predicted as Negative (757 rows)
Model retraining or tuning in this phase: no
```

Phase 8 intentionally did not retrain, tune, or select a model. Phase 9 should treat these held-out test metrics as the baseline result and perform model-improvement experiments using training and validation data before making any further test-set comparisons.

## Rubric Alignment

The final notebook and report should visibly cover every rubric criterion:

| Criterion | Marks | Evidence To Include |
|---|---:|---|
| Loading the Dataset | 5 | Explicit column names, shape, dtypes, sample rows, no load errors. |
| Data Cleaning | 10 | Missing values, duplicates, URL/mention/hashtag/special-character cleaning, tokenization, lowercasing, stop-word removal, stemming or lemmatization. |
| Feature Engineering | 15 | Numerical representation, token sequences, padding, vocabulary, embedding layer; optional TF-IDF baseline for comparison. |
| Basic Statistics | 5 | Dataset summary, missing values, duplicates, sentiment counts, tweet length statistics. |
| Visualisations | 10 | Sentiment distribution, word frequency, word clouds, tweet length versus sentiment. |
| Insights | 5 | Clear written trends and modeling implications from EDA. |
| Model Architecture | 10 | LSTM or GRU model with embedding layer. |
| Model Implementation | 10 | Stratified split, training loop, validation, dropout and/or normalization. |
| Evaluation | 10 | Accuracy, precision, recall, F1-score, confusion matrix, classification report, learning curves. |
| Model Improvement | 10 | Hyperparameter comparison, grid search, cross-validation, or transfer-learning discussion/attempt. |
| Documentation | 5 | Clear notebook markdown, comments, visuals, and code snippets. |
| Presentation | 5 | Slide-ready summary and sample tweet predictions. |

## Proposed Technical Approach

Use PyTorch for the RNN implementation. TensorFlow/Keras is not currently available in the local Python environments checked during planning, while PyTorch is available in the system Python. PyTorch satisfies the assignment requirement because the rubric asks for an LSTM or GRU based RNN with embeddings, not a specific framework.

Recommended model sequence:

1. Data loading and validation.
2. Cleaning and preprocessing.
3. EDA and visualizations.
4. TF-IDF baseline for comparison.
5. Tokenization, vocabulary building, sequence padding.
6. Embedding plus BiLSTM or GRU model.
7. Evaluation on held-out test data.
8. Small hyperparameter search and final model selection.
9. Sample tweet prediction demo.

## Project Structure

```text
.
|-- PROJECT_CONTEXT.md
|-- requirements.txt
|-- twitter_training.csv
|-- notebooks/
|   `-- RNN_Sentiment_Analysis_Twitter.ipynb
|-- src/
|   `-- __init__.py
|-- outputs/
|   |-- data/
|   |-- figures/
|   |-- models/
|   |-- reports/
|   `-- tables/
```

## Submission Notes

The first PDF requests the completed activity to be saved as a PDF and uploaded. It also notes that output should be provided as a Jupyter Notebook or Python script with code and comments.

Recommended final deliverables:

- A completed Jupyter Notebook.
- A Python script export of the notebook.
- A PDF report generated from the notebook or written from the notebook results.
- Presentation-ready summary content.
