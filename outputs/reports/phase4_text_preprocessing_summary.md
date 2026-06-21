# Phase 4 Text Preprocessing Summary

- Source cleaned dataset: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\data\twitter_training_cleaned_phase3.csv
- Preprocessed dataset: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\data\twitter_text_preprocessed_phase4.csv
- Rows preprocessed: 58,841
- Empty processed_text rows handled with model_text fallback: 1,151
- Empty model token rows after preprocessing: 0
- Empty analysis token rows after preprocessing: 0
- Model token count p95: 47.00
- Model token count p99: 56.00
- Recommended max sequence length for later padding: 60
- Duplicate model_text and sentiment rows to audit before modeling: 3,851
- Cleaned model_text values with multiple sentiment labels: 123
- Rows in conflicting model_text groups: 1,725

## Label Mapping For Later Modeling

- Negative: 0
- Neutral: 1
- Positive: 2

## Notes

The Phase 4 token frequency tables are descriptive preprocessing artifacts for EDA. They are not the final model vocabulary. The model vocabulary must be built from the training split only during feature engineering to avoid data leakage.

The duplicate-text audit is intentionally not applied as a row-removal step in this phase. It should guide Phase 6 train/test splitting and optional text-label deduplication so repeated cleaned text does not leak across splits.
