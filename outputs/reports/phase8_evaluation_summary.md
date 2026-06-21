# Phase 8 Held-Out Test Evaluation Summary

- Evaluated checkpoint: C:\Projects\RNN-Graded-Mini-Project\outputs\models\phase7_baseline_gru_state.pt
- Full test predictions: C:\Projects\RNN-Graded-Mini-Project\outputs\data\phase8_test_predictions.csv
- Test metrics: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase8_test_metrics.csv
- Test confusion matrix: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase8_test_confusion_matrix.csv
- Normalized test confusion matrix: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase8_test_confusion_matrix_normalized.csv
- Test prediction samples: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase8_test_prediction_samples.csv
- Confusion matrix figure: C:\Projects\RNN-Graded-Mini-Project\outputs\figures\phase8_test_confusion_matrix.svg
- Test rows evaluated: 8,826
- Test weighted cross-entropy loss: 0.8627
- Test accuracy: 0.6387
- Test macro F1: 0.6262
- Test weighted F1: 0.6313
- Negative F1: 0.7012
- Neutral F1: 0.5503
- Positive F1: 0.6271
- Most common error pattern: Neutral predicted as Negative (757 rows)
- Model retraining or tuning in this phase: no.

## Carry-Forward Notes

- Phase 9 should use these held-out test results as the baseline for improvement comparisons.
- Neutral is the weakest class for this baseline and should be watched during improvement experiments.
- Future experiments should tune on validation data first and only compare against this test result after selecting a final candidate.
