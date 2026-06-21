# Phase 9 Model Improvement Summary

- Source sequence arrays: C:\Projects\RNN-Graded-Mini-Project\outputs\data\phase6_sequences.npz
- Baseline validation metrics: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase7_validation_metrics.csv
- Held-out test metrics used as reference only: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase8_test_metrics.csv
- Best validation checkpoint: C:\Projects\RNN-Graded-Mini-Project\outputs\models\phase9_best_validation_model_state.pt
- Best validation metadata: C:\Projects\RNN-Graded-Mini-Project\outputs\models\phase9_best_validation_model_metadata.json
- Experiment results: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase9_experiment_results.csv
- Training history: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase9_training_history.csv
- Best validation metrics: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase9_best_validation_metrics.csv
- Best validation confusion matrix: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase9_best_validation_confusion_matrix.csv
- Comparison figure: C:\Projects\RNN-Graded-Mini-Project\outputs\figures\phase9_validation_comparison.svg
- Best learning curve: C:\Projects\RNN-Graded-Mini-Project\outputs\figures\phase9_best_learning_curve.svg
- Candidate experiments run: 3
- Best experiment by validation macro F1: gru_hidden96_dropout02
- Phase 7 baseline validation macro F1: 0.6185
- Best Phase 9 validation macro F1: 0.6874
- Validation macro F1 delta: +0.0689
- Best Phase 9 neutral F1: 0.6309
- Phase 8 held-out test accuracy reference: 0.6387
- Phase 8 held-out test macro F1 reference: 0.6262
- Total experiment time: 82.7 seconds
- Held-out test split used in this phase: no.

## Experiment Notes

- gru_hidden96_dropout02: validation macro F1 0.6874, neutral F1 0.6309, delta +0.0689.
- gru_hidden96_neutral_boost: validation macro F1 0.6670, neutral F1 0.6133, delta +0.0485.
- lstm_hidden64_dropout03: validation macro F1 0.6585, neutral F1 0.5871, delta +0.0400.

## Carry-Forward Notes

- Phase 10 can present the Phase 7/8 baseline and Phase 9 validation improvement as separate, clearly labeled results.
- Do not replace the Phase 8 held-out test result unless a final improved model is selected from validation evidence first.
- The best Phase 9 checkpoint can be used for one final test-set comparison only if the project requires an improved final-model test result.
