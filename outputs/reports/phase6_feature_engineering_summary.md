# Phase 6 Feature Engineering Summary

- Source preprocessed dataset: C:\Projects\RNN-Graded-Mini-Project\outputs\data\twitter_text_preprocessed_phase4.csv
- Feature dataset: C:\Projects\RNN-Graded-Mini-Project\outputs\data\twitter_features_phase6.csv
- Sequence arrays: C:\Projects\RNN-Graded-Mini-Project\outputs\data\phase6_sequences.npz
- Vocabulary metadata: C:\Projects\RNN-Graded-Mini-Project\outputs\data\phase6_vocabulary.json
- TF-IDF vocabulary metadata: C:\Projects\RNN-Graded-Mini-Project\outputs\data\phase6_tfidf_vocabulary.csv
- Rows engineered: 58,841
- Split rows: train 41,189, validation 8,826, test 8,826
- Split strategy: group-preserving split on model_text to reduce duplicate-text leakage.
- model_text values appearing in multiple splits: 0
- Conflicting-label model_text groups kept in a single split: 123
- Rows in conflicting-label model_text groups: 1,725
- Label mapping: Negative -> 0, Neutral -> 1, Positive -> 2
- Max sequence length: 60
- Rows truncated at max sequence length: 213
- Vocabulary built from training split only: 17,924 tokens including <PAD> and <OOV>.
- Embedding carry-forward: input_dim=17,924, padding_idx=0.
- TF-IDF carry-forward: 5,000 training-derived features saved for an optional baseline.
- Model training run in this phase: no.

## Artifacts

- phase6_split_distribution.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_split_distribution.csv
- phase6_group_split_audit.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_group_split_audit.csv
- phase6_vocabulary_summary.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_vocabulary_summary.csv
- phase6_sequence_length_summary.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_sequence_length_summary.csv
- phase6_oov_summary_by_split.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_oov_summary_by_split.csv
- phase6_tfidf_top_terms_by_sentiment.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_tfidf_top_terms_by_sentiment.csv
- phase6_sample_sequences.csv: C:\Projects\RNN-Graded-Mini-Project\outputs\tables\phase6_sample_sequences.csv

## Carry-Forward Notes

- Use phase6_sequences.npz for PyTorch Dataset objects in Phase 7.
- Keep PAD_ID as 0 and pass padding_idx=0 to the embedding layer.
- Use the validation split for model selection and the test split only for final evaluation.
- The vocabulary and TF-IDF metadata were fit on training rows only, so later modeling should reuse these artifacts instead of refitting on validation or test data.
