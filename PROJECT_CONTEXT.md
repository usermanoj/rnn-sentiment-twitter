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
