# Phase 5 EDA Insights

- The cleaned dataset contains 58,841 tweets across 32 entities and three sentiment labels.
- Sentiment distribution is moderately imbalanced: Negative 21,605, Positive 19,644, and Neutral 17,592.
- Negative is the largest class, so later modeling should use stratified splits and report macro F1 in addition to accuracy.
- The most frequent entity is TomClancysGhostRecon with 2,254 rows (3.83%). Entity counts are fairly spread across the 32 topics.
- Model token lengths are short enough for sequence modeling: p95 is 47 tokens and p99 is 56 tokens, supporting a later padding length of about 60.
- Top negative analysis tokens include: not, game, fuck, play, com, like, no, shit.
- Top positive analysis tokens include: game, play, not, com, love, good, real, like.
- Top neutral analysis tokens include: com, not, johnson, game, play, amazon, out, facebook.
- The duplicate-text audit from Phase 4 remains important for model splitting, because repeated cleaned text can inflate performance if similar examples appear in both train and test sets.

## Generated Figures

- sentiment_distribution: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\figures\phase5_sentiment_distribution.svg
- top_entities: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\figures\phase5_top_entities.svg
- length_by_sentiment: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\figures\phase5_tweet_length_by_sentiment.svg
- negative_word_cloud: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\figures\phase5_wordcloud_negative.svg
- positive_word_cloud: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\figures\phase5_wordcloud_positive.svg
- top_tokens_by_sentiment: C:\Users\Admin\OneDrive\Imp-Continuous-Backup-GDrive\Study\Technology\AI\IITM\W31-CNN-4-23-Feb-01-Mar-26\RNN-Graded-Mini-Project\outputs\figures\phase5_top_tokens_by_sentiment.svg

## Modeling Implications

- Use stratified splitting to preserve sentiment proportions.
- Build the model vocabulary only on the training split to avoid leakage.
- Treat 60 tokens as the initial max sequence length candidate for RNN padding.
- Consider deduplicating repeated model_text and sentiment pairs before final modeling experiments if validation leakage appears likely.
