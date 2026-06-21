# Final Project Report: RNN Twitter Sentiment Analysis

Generated on: 2026-06-21 15:02

## Executive Summary

This project built a recurrent neural network sentiment classifier for the Twitter
entity sentiment dataset. The final baseline model is a GRU classifier trained on
preprocessed token sequences and evaluated once on the held-out test split. The
baseline reached 63.9%
test accuracy and a 0.6262
test macro F1 score. Phase 9 then explored three validation-only improvement
experiments and identified `gru_hidden96_dropout02` as the strongest
candidate, improving validation macro F1 from
0.6185 to
0.6874.

## Problem Framing

The project goal was to classify tweets into Negative, Neutral, and Positive
sentiment classes. The modeling workflow was designed to keep a clean validation
and test protocol: model choices were made on training/validation evidence, and
the held-out test split was used only for the Phase 8 baseline evaluation.

## Data Preparation

- The cleaned working dataset contains 58,841 rows across three sentiment
  labels.
- Text cleaning removed unusable text, standardized model-facing fields, and kept
  duplicate/conflict audits as traceable CSV artifacts.
- Preprocessing produced separate analysis tokens and model tokens so EDA could
  stay interpretable while the model consumed compact normalized sequences.
- The feature pipeline retained a final vocabulary of 17,924 tokens,
  including `<PAD>` and `<OOV>`.
- Group-aware splitting produced 41,189 training rows,
  8,826 validation rows, and 8,826
  held-out test rows.

## Modeling Approach

The Phase 7 baseline used a GRU sequence classifier with an embedding layer,
class-weighted cross-entropy, dropout, and early checkpoint selection by
validation macro F1. This was a conservative baseline for a short-text
multi-class sentiment task: simple enough to interpret, but still able to use
token order.

## Baseline Evaluation

The Phase 8 held-out test evaluation used the saved Phase 7 checkpoint without
retraining. Key test metrics were:

- Accuracy: 0.6387
- Macro F1: 0.6262
- Weighted F1: 0.6313
- Negative F1: 0.7012
- Neutral F1: 0.5503
- Positive F1: 0.6271

Neutral was the weakest class. The largest error mode was Neutral being predicted
as Negative, which makes sense for short tweets where factual or ambiguous
mentions can share vocabulary with complaints.

## Improvement Experiments

Phase 9 tested three validation-only candidates. The best candidate,
`gru_hidden96_dropout02`, used a larger GRU hidden state with slightly
lower dropout. It reached validation accuracy
0.6926 and validation macro F1
0.6874, a
+0.0689 macro
F1 lift over the Phase 7 validation baseline. The held-out test split was not
used in Phase 9.

## Limitations

- The reported held-out test result is for the Phase 7 baseline checkpoint. The
  Phase 9 best model remains a validation-selected candidate unless the project
  requires one final test evaluation of that selected candidate.
- Neutral sentiment remains harder than Negative or Positive, likely because
  neutral tweets are often brief, context-dependent, or entity-specific.
- The RNN uses learned embeddings from the project corpus rather than pretrained
  contextual representations, so sarcasm and world knowledge remain challenging.

## Recommended Next Step

If another modeling phase is allowed, run exactly one final held-out test
evaluation for the selected Phase 9 checkpoint and report it as the final model
comparison. If no more test use is allowed, submit the Phase 7 held-out test
baseline and the Phase 9 validation improvement as separate, clearly labeled
results.

## Final Artifact Index

- Main notebook: `notebooks/RNN_Sentiment_Analysis_Twitter.ipynb`
- Final presentation deck: `outputs/reports/rnn_twitter_sentiment_final_presentation.pptx`
- Final key metrics: `outputs/tables/phase10_final_key_metrics.csv`
- Submission checklist: `outputs/reports/phase10_final_submission_checklist.md`
- Phase 8 held-out metrics: `outputs/tables/phase8_test_metrics.csv`
- Phase 9 experiment results: `outputs/tables/phase9_experiment_results.csv`
- Baseline confusion matrix: `outputs/figures/phase8_test_confusion_matrix.svg`
- Validation improvement figure: `outputs/figures/phase9_validation_comparison.svg`
- Baseline checkpoint: `outputs/models/phase7_baseline_gru_state.pt`
- Best validation checkpoint: `outputs/models/phase9_best_validation_model_state.pt`
