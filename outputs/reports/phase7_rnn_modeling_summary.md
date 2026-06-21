# Phase 7 RNN Modeling Summary

- Source sequence arrays: C:\Projects\RNN-Graded-Mini-Project\outputs\data\phase6_sequences.npz
- Model checkpoint: C:\Projects\RNN-Graded-Mini-Project\outputs\models\phase7_baseline_gru_state.pt
- Model metadata: C:\Projects\RNN-Graded-Mini-Project\outputs\models\phase7_baseline_gru_metadata.json
- Training history: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase7_training_history.csv
- Validation metrics: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase7_validation_metrics.csv
- Validation confusion matrix: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase7_validation_confusion_matrix.csv
- Learning curve: C:\Projects\RNN-Graded-Mini-Project\outputs\figures\phase7_learning_curve.svg
- Model: embedding + GRU baseline with dropout.
- Vocabulary size: 17,924
- Embedding dimension: 64
- Hidden dimension: 64
- Train rows: 41,189
- Validation rows: 8,826
- Epochs trained: 3
- Best epoch by validation macro F1: 3
- Best validation loss: 0.8735
- Best validation accuracy: 0.6313
- Best validation macro F1: 0.6185
- Total training time: 11.3 seconds
- Held-out test split used: no.

## Carry-Forward Notes

- Phase 8 should load the saved checkpoint and evaluate once on the held-out test split.
- Report macro F1 alongside accuracy because the classes are moderately imbalanced.
- Phase 9 can compare this compact GRU baseline against larger GRU/LSTM variants or tuned dropout/hidden dimensions.
