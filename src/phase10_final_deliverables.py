"""Generate Phase 10 final report and submission artifacts.

This phase packages already-produced project outputs. It does not train,
evaluate, tune, or otherwise mutate any model checkpoint.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = OUTPUTS / "reports"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
DATA = OUTPUTS / "data"
MODELS = OUTPUTS / "models"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(rows: list[dict[str, str]], label: str, column: str) -> str:
    for row in rows:
        if row.get("label") == label:
            return row[column]
    raise KeyError(f"Missing metric row: {label}")


def value_lookup(rows: list[dict[str, str]], metric: str) -> str:
    for row in rows:
        if row.get("metric") == metric:
            return row["value"]
    raise KeyError(f"Missing value row: {metric}")


def fmt_float(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def fmt_count(value: str) -> str:
    return f"{int(str(value).replace(',', '')):,}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    phase5_stats = read_csv_rows(TABLES / "phase5_basic_statistics.csv")
    split_rows = read_csv_rows(TABLES / "phase6_split_distribution.csv")
    vocab_rows = read_csv_rows(TABLES / "phase6_vocabulary_summary.csv")
    phase7_metrics = read_csv_rows(TABLES / "phase7_validation_metrics.csv")
    phase8_metrics = read_csv_rows(TABLES / "phase8_test_metrics.csv")
    phase9_results = read_csv_rows(TABLES / "phase9_experiment_results.csv")

    best_phase9 = max(
        (row for row in phase9_results if row["experiment_id"] != "phase7_baseline_reference"),
        key=lambda row: float(row["best_validation_macro_f1"]),
    )
    phase7_reference = next(
        row for row in phase9_results if row["experiment_id"] == "phase7_baseline_reference"
    )

    total_rows = value_lookup(phase5_stats, "rows")
    vocab_size = value_lookup(vocab_rows, "final_vocabulary_size")
    train_rows = value_lookup(vocab_rows, "train_rows")
    validation_total = next(row for row in split_rows if row["split"] == "validation")[
        "split_rows"
    ]
    test_total = next(row for row in split_rows if row["split"] == "test")["split_rows"]
    split_summary = (
        f"{fmt_count(train_rows)} / {fmt_count(validation_total)} / {fmt_count(test_total)}"
    )

    key_metrics = [
        {
            "area": "Dataset",
            "metric": "Cleaned rows",
            "value": total_rows,
            "source": rel(TABLES / "phase5_basic_statistics.csv"),
        },
        {
            "area": "Split",
            "metric": "Train / validation / test rows",
            "value": split_summary,
            "source": rel(TABLES / "phase6_split_distribution.csv"),
        },
        {
            "area": "Feature engineering",
            "metric": "Final vocabulary size",
            "value": vocab_size,
            "source": rel(TABLES / "phase6_vocabulary_summary.csv"),
        },
        {
            "area": "Baseline validation",
            "metric": "Phase 7 validation macro F1",
            "value": fmt_float(metric_lookup(phase7_metrics, "macro_avg", "f1_score")),
            "source": rel(TABLES / "phase7_validation_metrics.csv"),
        },
        {
            "area": "Held-out test",
            "metric": "Phase 8 test accuracy",
            "value": fmt_float(metric_lookup(phase8_metrics, "accuracy", "f1_score")),
            "source": rel(TABLES / "phase8_test_metrics.csv"),
        },
        {
            "area": "Held-out test",
            "metric": "Phase 8 test macro F1",
            "value": fmt_float(metric_lookup(phase8_metrics, "macro_avg", "f1_score")),
            "source": rel(TABLES / "phase8_test_metrics.csv"),
        },
        {
            "area": "Improvement",
            "metric": "Best Phase 9 validation macro F1",
            "value": fmt_float(best_phase9["best_validation_macro_f1"]),
            "source": rel(TABLES / "phase9_experiment_results.csv"),
        },
        {
            "area": "Improvement",
            "metric": "Validation macro F1 lift vs Phase 7",
            "value": f"+{fmt_float(best_phase9['improvement_over_phase7_validation_macro_f1'])}",
            "source": rel(TABLES / "phase9_experiment_results.csv"),
        },
    ]
    write_csv(
        TABLES / "phase10_final_key_metrics.csv",
        key_metrics,
        ["area", "metric", "value", "source"],
    )

    generated_on = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = f"""# Final Project Report: RNN Twitter Sentiment Analysis

Generated on: {generated_on}

## Executive Summary

This project built a recurrent neural network sentiment classifier for the Twitter
entity sentiment dataset. The final baseline model is a GRU classifier trained on
preprocessed token sequences and evaluated once on the held-out test split. The
baseline reached {fmt_pct(metric_lookup(phase8_metrics, "accuracy", "f1_score"))}
test accuracy and a {fmt_float(metric_lookup(phase8_metrics, "macro_avg", "f1_score"))}
test macro F1 score. Phase 9 then explored three validation-only improvement
experiments and identified `{best_phase9["experiment_id"]}` as the strongest
candidate, improving validation macro F1 from
{fmt_float(phase7_reference["best_validation_macro_f1"])} to
{fmt_float(best_phase9["best_validation_macro_f1"])}.

## Problem Framing

The project goal was to classify tweets into Negative, Neutral, and Positive
sentiment classes. The modeling workflow was designed to keep a clean validation
and test protocol: model choices were made on training/validation evidence, and
the held-out test split was used only for the Phase 8 baseline evaluation.

## Data Preparation

- The cleaned working dataset contains {total_rows} rows across three sentiment
  labels.
- Text cleaning removed unusable text, standardized model-facing fields, and kept
  duplicate/conflict audits as traceable CSV artifacts.
- Preprocessing produced separate analysis tokens and model tokens so EDA could
  stay interpretable while the model consumed compact normalized sequences.
- The feature pipeline retained a final vocabulary of {vocab_size} tokens,
  including `<PAD>` and `<OOV>`.
- Group-aware splitting produced {fmt_count(train_rows)} training rows,
  {fmt_count(validation_total)} validation rows, and {fmt_count(test_total)}
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

- Accuracy: {fmt_float(metric_lookup(phase8_metrics, "accuracy", "f1_score"))}
- Macro F1: {fmt_float(metric_lookup(phase8_metrics, "macro_avg", "f1_score"))}
- Weighted F1: {fmt_float(metric_lookup(phase8_metrics, "weighted_avg", "f1_score"))}
- Negative F1: {fmt_float(metric_lookup(phase8_metrics, "Negative", "f1_score"))}
- Neutral F1: {fmt_float(metric_lookup(phase8_metrics, "Neutral", "f1_score"))}
- Positive F1: {fmt_float(metric_lookup(phase8_metrics, "Positive", "f1_score"))}

Neutral was the weakest class. The largest error mode was Neutral being predicted
as Negative, which makes sense for short tweets where factual or ambiguous
mentions can share vocabulary with complaints.

## Improvement Experiments

Phase 9 tested three validation-only candidates. The best candidate,
`{best_phase9["experiment_id"]}`, used a larger GRU hidden state with slightly
lower dropout. It reached validation accuracy
{fmt_float(best_phase9["best_validation_accuracy"])} and validation macro F1
{fmt_float(best_phase9["best_validation_macro_f1"])}, a
+{fmt_float(best_phase9["improvement_over_phase7_validation_macro_f1"])} macro
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

- Main notebook: `{rel(ROOT / "notebooks" / "RNN_Sentiment_Analysis_Twitter.ipynb")}`
- Final presentation deck: `{rel(REPORTS / "rnn_twitter_sentiment_final_presentation.pptx")}`
- Final key metrics: `{rel(TABLES / "phase10_final_key_metrics.csv")}`
- Submission checklist: `{rel(REPORTS / "phase10_final_submission_checklist.md")}`
- Phase 8 held-out metrics: `{rel(TABLES / "phase8_test_metrics.csv")}`
- Phase 9 experiment results: `{rel(TABLES / "phase9_experiment_results.csv")}`
- Baseline confusion matrix: `{rel(FIGURES / "phase8_test_confusion_matrix.svg")}`
- Validation improvement figure: `{rel(FIGURES / "phase9_validation_comparison.svg")}`
- Baseline checkpoint: `{rel(MODELS / "phase7_baseline_gru_state.pt")}`
- Best validation checkpoint: `{rel(MODELS / "phase9_best_validation_model_state.pt")}`
"""
    (REPORTS / "final_project_report.md").write_text(report, encoding="utf-8")

    checklist = f"""# Phase 10 Final Submission Checklist

- [x] Notebook contains phases 1-10.
- [x] Data loading, validation, cleaning, preprocessing, EDA, features, baseline modeling, held-out evaluation, and improvement artifacts are saved.
- [x] Final report written to `{rel(REPORTS / "final_project_report.md")}`.
- [x] Final key metric table written to `{rel(TABLES / "phase10_final_key_metrics.csv")}`.
- [x] Presentation deck generated as a premium executive readout.
- [x] No training was run during Phase 10 report generation.
- [x] Phase 8 test metrics and Phase 9 validation improvements are labeled separately.
- [x] Recommended next step is documented without reusing the held-out test split.
"""
    (REPORTS / "phase10_final_submission_checklist.md").write_text(
        checklist, encoding="utf-8"
    )

    summary = f"""# Phase 10 Final Deliverables Summary

Phase 10 packages the project for submission. It reads existing outputs and does
not run model training, tuning, or test-set evaluation.

## Deliverables

- Final report: `{rel(REPORTS / "final_project_report.md")}`
- Submission checklist: `{rel(REPORTS / "phase10_final_submission_checklist.md")}`
- Final metrics table: `{rel(TABLES / "phase10_final_key_metrics.csv")}`
- Presentation deck: `{rel(REPORTS / "rnn_twitter_sentiment_final_presentation.pptx")}`

## Headline Results

- Phase 8 held-out test accuracy: {fmt_float(metric_lookup(phase8_metrics, "accuracy", "f1_score"))}
- Phase 8 held-out test macro F1: {fmt_float(metric_lookup(phase8_metrics, "macro_avg", "f1_score"))}
- Best Phase 9 validation macro F1: {fmt_float(best_phase9["best_validation_macro_f1"])}
- Validation macro F1 lift over Phase 7 baseline: +{fmt_float(best_phase9["improvement_over_phase7_validation_macro_f1"])}
"""
    (REPORTS / "phase10_final_deliverables_summary.md").write_text(
        summary, encoding="utf-8"
    )

    print("Wrote Phase 10 final report artifacts.")
    for path in [
        REPORTS / "final_project_report.md",
        REPORTS / "phase10_final_submission_checklist.md",
        REPORTS / "phase10_final_deliverables_summary.md",
        TABLES / "phase10_final_key_metrics.csv",
    ]:
        print(path)


if __name__ == "__main__":
    main()
