# Phase 3 Data Cleaning Summary

- Raw rows loaded: 74,682
- Removed blank or missing tweet text rows: 858
- Removed exact duplicate rows after blank-text removal: 2,340
- Removed rows with Irrelevant label: 12,504
- Removed rows that became empty after text cleaning: 139
- Final cleaned rows: 58,841
- Stemming method used in this run: fallback_suffix_stemmer
- Cleaned dataset: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\data\twitter_training_cleaned_phase3.csv

## Final Cleaned Sentiment Distribution

- Negative: 21,605 (36.72%)
- Positive: 19,644 (33.38%)
- Neutral: 17,592 (29.90%)

## Cleaning Notes

The original raw CSV was not modified. The cleaned dataset excludes Irrelevant because the graded assignment defines a three-class sentiment task. The notebook keeps both model_text and processed_text so later RNN work can use a sequence-friendly text representation while EDA and rubric documentation can show stop-word removal and stemming.
