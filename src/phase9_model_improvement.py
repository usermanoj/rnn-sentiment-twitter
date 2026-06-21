"""Phase 9 model-improvement experiments.

This script compares a small set of validation-driven RNN variants against the
Phase 7 baseline. It intentionally does not load or evaluate the held-out test
split; Phase 8 remains the held-out baseline result.
"""

from __future__ import annotations

import copy
import html
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

for directory in [FIGURES_DIR, MODELS_DIR, TABLES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PHASE6_SEQUENCES_PATH = DATA_OUTPUT_DIR / "phase6_sequences.npz"
PHASE6_CONFIG_PATH = DATA_OUTPUT_DIR / "phase6_feature_config.json"
PHASE7_VALIDATION_METRICS_PATH = TABLES_DIR / "phase7_validation_metrics.csv"
PHASE8_TEST_METRICS_PATH = TABLES_DIR / "phase8_test_metrics.csv"

PHASE9_BEST_CHECKPOINT_PATH = MODELS_DIR / "phase9_best_validation_model_state.pt"
PHASE9_BEST_METADATA_PATH = MODELS_DIR / "phase9_best_validation_model_metadata.json"
PHASE9_EXPERIMENT_RESULTS_PATH = TABLES_DIR / "phase9_experiment_results.csv"
PHASE9_TRAINING_HISTORY_PATH = TABLES_DIR / "phase9_training_history.csv"
PHASE9_BEST_VALIDATION_METRICS_PATH = TABLES_DIR / "phase9_best_validation_metrics.csv"
PHASE9_BEST_CONFUSION_MATRIX_PATH = TABLES_DIR / "phase9_best_validation_confusion_matrix.csv"
PHASE9_MODEL_CONFIGS_PATH = TABLES_DIR / "phase9_model_configs.csv"
PHASE9_COMPARISON_FIGURE_PATH = FIGURES_DIR / "phase9_validation_comparison.svg"
PHASE9_BEST_LEARNING_CURVE_PATH = FIGURES_DIR / "phase9_best_learning_curve.svg"
PHASE9_SUMMARY_PATH = REPORTS_DIR / "phase9_model_improvement_summary.md"

PHASE9_RANDOM_SEED = 42


class RecurrentSentimentClassifier(nn.Module):
    """Embedding plus GRU/LSTM classifier used for Phase 9 candidates."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        padding_idx: int,
        recurrent_type: str = "GRU",
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.recurrent_type = recurrent_type.upper()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_rate)
        if self.recurrent_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        elif self.recurrent_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent_type: {recurrent_type}")
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.recurrent(packed)
        hidden_state = hidden[0] if self.recurrent_type == "LSTM" else hidden
        final_hidden = self.dropout(hidden_state[-1])
        return self.classifier(final_hidden)


def build_confusion_matrix(
    true_labels: np.ndarray, predicted_labels: np.ndarray, num_classes: int
) -> np.ndarray:
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[int(true_label), int(predicted_label)] += 1
    return confusion_matrix


def classification_metrics_from_confusion(
    confusion_matrix: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    rows = []
    total = confusion_matrix.sum()
    accuracy = float(np.trace(confusion_matrix) / total) if total else 0.0
    for index, class_name in enumerate(class_names):
        true_positive = float(confusion_matrix[index, index])
        false_positive = float(confusion_matrix[:, index].sum() - true_positive)
        false_negative = float(confusion_matrix[index, :].sum() - true_positive)
        support = float(confusion_matrix[index, :].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": int(support),
            }
        )
    metrics = pd.DataFrame(rows)
    summary_rows = pd.DataFrame(
        [
            {
                "label": "accuracy",
                "precision": accuracy,
                "recall": accuracy,
                "f1_score": accuracy,
                "support": int(total),
            },
            {
                "label": "macro_avg",
                "precision": metrics["precision"].mean(),
                "recall": metrics["recall"].mean(),
                "f1_score": metrics["f1_score"].mean(),
                "support": int(total),
            },
            {
                "label": "weighted_avg",
                "precision": np.average(metrics["precision"], weights=metrics["support"]),
                "recall": np.average(metrics["recall"], weights=metrics["support"]),
                "f1_score": np.average(metrics["f1_score"], weights=metrics["support"]),
                "support": int(total),
            },
        ]
    )
    return pd.concat([metrics, summary_rows], ignore_index=True)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 1.0,
) -> tuple[float, float, float, np.ndarray, pd.DataFrame]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    all_true_labels = []
    all_predictions = []

    for input_ids, lengths, labels in data_loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_true_labels.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())

    true_labels = np.concatenate(all_true_labels)
    predicted_labels = np.concatenate(all_predictions)
    confusion_matrix = build_confusion_matrix(true_labels, predicted_labels, num_classes)
    metrics_table = classification_metrics_from_confusion(confusion_matrix, class_names)
    average_loss = total_loss / max(total_examples, 1)
    accuracy = float(metrics_table.loc[metrics_table["label"].eq("accuracy"), "f1_score"].iat[0])
    macro_f1 = float(metrics_table.loc[metrics_table["label"].eq("macro_avg"), "f1_score"].iat[0])
    return average_loss, accuracy, macro_f1, confusion_matrix, metrics_table


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def svg_text(text: object) -> str:
    return html.escape(str(text), quote=True)


def save_bar_comparison_svg(results: pd.DataFrame, path: Path, width: int = 960, height: int = 540) -> None:
    plot_data = results[["experiment_id", "best_validation_macro_f1", "neutral_f1"]].copy()
    plot_data["experiment_id"] = plot_data["experiment_id"].str.replace("_", " ")
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 75, 145
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(float(plot_data[["best_validation_macro_f1", "neutral_f1"]].to_numpy().max()), 0.7)
    group_width = plot_width / len(plot_data)
    bar_width = min(56, group_width * 0.28)
    colors = {"best_validation_macro_f1": "#2f6f9f", "neutral_f1": "#8a6f2a"}
    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#222">Phase 9 Validation Improvement Comparison</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
    ]
    for tick in np.linspace(0, max_value, 6):
        y = height - margin_bottom - tick / max_value * plot_height
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{tick:.2f}</text>')
    for index, row in plot_data.reset_index(drop=True).iterrows():
        center_x = margin_left + index * group_width + group_width / 2
        for offset, metric in [(-bar_width / 1.8, "best_validation_macro_f1"), (bar_width / 1.8, "neutral_f1")]:
            value = float(row[metric])
            bar_height = value / max_value * plot_height
            x = center_x + offset - bar_width / 2
            y = height - margin_bottom - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{colors[metric]}"/>')
            parts.append(f'<text x="{x + bar_width/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#222">{value:.3f}</text>')
        parts.append(f'<text x="{center_x:.1f}" y="{height-margin_bottom+24}" text-anchor="middle" font-family="Arial" font-size="11" fill="#222">{svg_text(row["experiment_id"])}</text>')
    legend_x = width - 245
    parts.extend(
        [
            f'<rect x="{legend_x}" y="54" width="14" height="14" fill="{colors["best_validation_macro_f1"]}"/>',
            f'<text x="{legend_x+20}" y="66" font-family="Arial" font-size="12" fill="#222">Validation macro F1</text>',
            f'<rect x="{legend_x}" y="76" width="14" height="14" fill="{colors["neutral_f1"]}"/>',
            f'<text x="{legend_x+20}" y="88" font-family="Arial" font-size="12" fill="#222">Neutral F1</text>',
        ]
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">\n'
        + "\n".join(parts)
        + "\n</svg>\n",
        encoding="utf-8",
    )


def save_best_learning_curve_svg(
    history: pd.DataFrame, best_experiment_id: str, path: Path, width: int = 900, height: int = 520
) -> None:
    history = history[history["experiment_id"].eq(best_experiment_id)].copy()
    margin_left, margin_right, margin_top, margin_bottom = 80, 150, 70, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    epochs = history["epoch"].to_numpy(dtype=float)
    series = {
        "train_loss": (history["train_loss"].to_numpy(dtype=float), "#b4494b"),
        "validation_loss": (history["validation_loss"].to_numpy(dtype=float), "#2f6f9f"),
        "validation_macro_f1": (history["validation_macro_f1"].to_numpy(dtype=float), "#3f8f61"),
    }
    y_values = np.concatenate([values for values, _ in series.values()])
    y_min = max(0.0, float(y_values.min()) - 0.05)
    y_max = min(1.5, float(y_values.max()) + 0.05)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_for_epoch(epoch: float) -> float:
        if len(epochs) == 1:
            return margin_left + plot_width / 2
        return margin_left + (epoch - epochs.min()) / (epochs.max() - epochs.min()) * plot_width

    def y_for_value(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="36" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="#222">Best Phase 9 Learning Curve</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = y_for_value(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{tick:.2f}</text>')
    for metric_index, (label, (values, color)) in enumerate(series.items()):
        points = " ".join([f"{x_for_epoch(epoch):.1f},{y_for_value(value):.1f}" for epoch, value in zip(epochs, values)])
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for epoch, value in zip(epochs, values):
            parts.append(f'<circle cx="{x_for_epoch(epoch):.1f}" cy="{y_for_value(value):.1f}" r="4" fill="{color}"/>')
        legend_y = margin_top + 8 + metric_index * 24
        parts.append(f'<rect x="{width-margin_right+25}" y="{legend_y-10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width-margin_right+46}" y="{legend_y+2}" font-family="Arial" font-size="13" fill="#222">{svg_text(label)}</text>')
    for epoch in epochs:
        x = x_for_epoch(epoch)
        parts.append(f'<text x="{x:.1f}" y="{height-margin_bottom+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">{int(epoch)}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">\n'
        + "\n".join(parts)
        + "\n</svg>\n",
        encoding="utf-8",
    )


def main() -> dict[str, object]:
    random.seed(PHASE9_RANDOM_SEED)
    np.random.seed(PHASE9_RANDOM_SEED)
    torch.manual_seed(PHASE9_RANDOM_SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Improvement experiment device: {device}")

    with np.load(PHASE6_SEQUENCES_PATH) as sequence_data:
        x_train = sequence_data["X_train"].astype(np.int64)
        y_train = sequence_data["y_train"].astype(np.int64)
        x_validation = sequence_data["X_validation"].astype(np.int64)
        y_validation = sequence_data["y_validation"].astype(np.int64)

    phase6_config = json.loads(PHASE6_CONFIG_PATH.read_text(encoding="utf-8"))
    label_to_id = {label: int(label_id) for label, label_id in phase6_config["label_to_id"].items()}
    id_to_label = {label_id: label for label, label_id in label_to_id.items()}
    class_names = [id_to_label[index] for index in sorted(id_to_label)]
    vocab_size = int(phase6_config["embedding_input_dim_for_phase7"])
    padding_idx = int(phase6_config["embedding_padding_idx_for_phase7"])
    train_lengths = np.maximum((x_train != padding_idx).sum(axis=1), 1).astype(np.int64)
    validation_lengths = np.maximum((x_validation != padding_idx).sum(axis=1), 1).astype(np.int64)
    num_classes = len(class_names)

    baseline_validation_metrics = pd.read_csv(PHASE7_VALIDATION_METRICS_PATH)
    baseline_validation_macro_f1 = float(
        baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("macro_avg"), "f1_score"].iat[0]
    )
    baseline_validation_accuracy = float(
        baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("accuracy"), "f1_score"].iat[0]
    )
    baseline_neutral_f1 = float(
        baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("Neutral"), "f1_score"].iat[0]
    )
    phase8_test_metrics = pd.read_csv(PHASE8_TEST_METRICS_PATH)
    phase8_test_macro_f1 = float(
        phase8_test_metrics.loc[phase8_test_metrics["label"].eq("macro_avg"), "f1_score"].iat[0]
    )
    baseline_test_accuracy = float(
        phase8_test_metrics.loc[phase8_test_metrics["label"].eq("accuracy"), "f1_score"].iat[0]
    )

    print(f"Train rows: {len(y_train):,}; validation rows: {len(y_validation):,}")
    print(f"Phase 7 validation baseline macro F1: {baseline_validation_macro_f1:.4f}")
    print(f"Phase 8 held-out test baseline macro F1, reference only: {phase8_test_macro_f1:.4f}")

    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(train_lengths), torch.from_numpy(y_train)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(x_validation), torch.from_numpy(validation_lengths), torch.from_numpy(y_validation)
    )
    base_class_counts = np.bincount(y_train, minlength=num_classes)
    base_class_weights = len(y_train) / (num_classes * np.maximum(base_class_counts, 1))

    experiment_configs = [
        {
            "experiment_id": "gru_hidden96_dropout02",
            "recurrent_type": "GRU",
            "embedding_dim": 64,
            "hidden_dim": 96,
            "dropout_rate": 0.20,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 4,
            "early_stopping_patience": 2,
            "gradient_clip_norm": 1.0,
            "neutral_weight_multiplier": 1.00,
            "notes": "larger GRU hidden state with slightly lower dropout",
        },
        {
            "experiment_id": "gru_hidden96_neutral_boost",
            "recurrent_type": "GRU",
            "embedding_dim": 64,
            "hidden_dim": 96,
            "dropout_rate": 0.30,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 4,
            "early_stopping_patience": 2,
            "gradient_clip_norm": 1.0,
            "neutral_weight_multiplier": 1.20,
            "notes": "larger GRU with extra neutral-class weight",
        },
        {
            "experiment_id": "lstm_hidden64_dropout03",
            "recurrent_type": "LSTM",
            "embedding_dim": 64,
            "hidden_dim": 64,
            "dropout_rate": 0.30,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 4,
            "early_stopping_patience": 2,
            "gradient_clip_norm": 1.0,
            "neutral_weight_multiplier": 1.00,
            "notes": "same hidden size as baseline, LSTM recurrent cell",
        },
    ]

    all_history_rows: list[dict[str, object]] = []
    experiment_result_rows: list[dict[str, object]] = []
    best_experiment_payload: dict[str, object] | None = None
    phase9_started_at = time.perf_counter()

    for experiment_index, config in enumerate(experiment_configs, start=1):
        experiment_id = config["experiment_id"]
        random.seed(PHASE9_RANDOM_SEED + experiment_index)
        np.random.seed(PHASE9_RANDOM_SEED + experiment_index)
        torch.manual_seed(PHASE9_RANDOM_SEED + experiment_index)

        train_loader = DataLoader(
            train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0
        )

        model = RecurrentSentimentClassifier(
            vocab_size=vocab_size,
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_classes=num_classes,
            padding_idx=padding_idx,
            recurrent_type=str(config["recurrent_type"]),
            dropout_rate=float(config["dropout_rate"]),
        ).to(device)
        class_weights = base_class_weights.astype(np.float32).copy()
        class_weights[label_to_id["Neutral"]] *= float(config["neutral_weight_multiplier"])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        best_validation_macro_f1 = -1.0
        best_epoch = None
        best_state_dict = None
        best_validation_metrics = None
        best_validation_confusion_matrix = None
        best_validation_loss = None
        best_validation_accuracy = None
        epochs_without_improvement = 0
        experiment_started_at = time.perf_counter()
        epochs_run = 0

        print(f"Starting {experiment_id}: {config['notes']}")
        for epoch in range(1, int(config["max_epochs"]) + 1):
            epochs_run = epoch
            epoch_started_at = time.perf_counter()
            train_loss, train_accuracy, train_macro_f1, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                class_names,
                num_classes,
                optimizer=optimizer,
                gradient_clip_norm=float(config["gradient_clip_norm"]),
            )
            (
                validation_loss,
                validation_accuracy,
                validation_macro_f1,
                validation_confusion_matrix,
                validation_metrics,
            ) = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                class_names,
                num_classes,
                optimizer=None,
                gradient_clip_norm=float(config["gradient_clip_norm"]),
            )
            epoch_seconds = time.perf_counter() - epoch_started_at
            all_history_rows.append(
                {
                    "experiment_id": experiment_id,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_macro_f1": train_macro_f1,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "validation_macro_f1": validation_macro_f1,
                    "epoch_seconds": round(epoch_seconds, 2),
                }
            )
            print(
                f"  Epoch {epoch}/{config['max_epochs']} | "
                f"train macro F1 {train_macro_f1:.4f} | "
                f"val loss {validation_loss:.4f}, acc {validation_accuracy:.4f}, macro F1 {validation_macro_f1:.4f} | "
                f"{epoch_seconds:.1f}s"
            )

            if validation_macro_f1 > best_validation_macro_f1 + 1e-6:
                best_validation_macro_f1 = validation_macro_f1
                best_epoch = epoch
                best_state_dict = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                best_validation_metrics = validation_metrics.copy()
                best_validation_confusion_matrix = validation_confusion_matrix.copy()
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(config["early_stopping_patience"]):
                    print(f"  Early stopping {experiment_id} after epoch {epoch}.")
                    break

        experiment_seconds = time.perf_counter() - experiment_started_at
        assert best_validation_metrics is not None
        assert best_validation_confusion_matrix is not None
        assert best_state_dict is not None
        neutral_row = best_validation_metrics.loc[best_validation_metrics["label"].eq("Neutral")].iloc[0]
        negative_row = best_validation_metrics.loc[best_validation_metrics["label"].eq("Negative")].iloc[0]
        positive_row = best_validation_metrics.loc[best_validation_metrics["label"].eq("Positive")].iloc[0]
        result_row = {
            "experiment_id": experiment_id,
            "recurrent_type": config["recurrent_type"],
            "embedding_dim": int(config["embedding_dim"]),
            "hidden_dim": int(config["hidden_dim"]),
            "dropout_rate": float(config["dropout_rate"]),
            "neutral_weight_multiplier": float(config["neutral_weight_multiplier"]),
            "parameter_count": int(parameter_count),
            "epochs_run": int(epochs_run),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "best_validation_accuracy": float(best_validation_accuracy),
            "best_validation_macro_f1": float(best_validation_macro_f1),
            "negative_f1": float(negative_row["f1_score"]),
            "neutral_precision": float(neutral_row["precision"]),
            "neutral_recall": float(neutral_row["recall"]),
            "neutral_f1": float(neutral_row["f1_score"]),
            "positive_f1": float(positive_row["f1_score"]),
            "improvement_over_phase7_validation_macro_f1": float(
                best_validation_macro_f1 - baseline_validation_macro_f1
            ),
            "experiment_seconds": round(experiment_seconds, 2),
            "notes": config["notes"],
        }
        experiment_result_rows.append(result_row)
        candidate_payload = {
            "experiment_id": experiment_id,
            "config": config,
            "result_row": result_row,
            "state_dict": best_state_dict,
            "validation_metrics": best_validation_metrics,
            "validation_confusion_matrix": best_validation_confusion_matrix,
            "class_weights": {class_names[index]: float(class_weights[index]) for index in range(num_classes)},
        }
        if (
            best_experiment_payload is None
            or result_row["best_validation_macro_f1"]
            > best_experiment_payload["result_row"]["best_validation_macro_f1"]
        ):
            best_experiment_payload = candidate_payload

    assert best_experiment_payload is not None
    phase9_total_seconds = time.perf_counter() - phase9_started_at
    phase9_training_history = pd.DataFrame(all_history_rows)
    phase9_experiment_results = (
        pd.DataFrame(experiment_result_rows)
        .sort_values("best_validation_macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    baseline_reference_row = {
        "experiment_id": "phase7_baseline_reference",
        "recurrent_type": "GRU",
        "embedding_dim": 64,
        "hidden_dim": 64,
        "dropout_rate": 0.30,
        "neutral_weight_multiplier": 1.00,
        "parameter_count": 1172291,
        "epochs_run": 3,
        "best_epoch": 3,
        "best_validation_loss": np.nan,
        "best_validation_accuracy": baseline_validation_accuracy,
        "best_validation_macro_f1": baseline_validation_macro_f1,
        "negative_f1": float(
            baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("Negative"), "f1_score"].iat[0]
        ),
        "neutral_precision": float(
            baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("Neutral"), "precision"].iat[0]
        ),
        "neutral_recall": float(
            baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("Neutral"), "recall"].iat[0]
        ),
        "neutral_f1": baseline_neutral_f1,
        "positive_f1": float(
            baseline_validation_metrics.loc[baseline_validation_metrics["label"].eq("Positive"), "f1_score"].iat[0]
        ),
        "improvement_over_phase7_validation_macro_f1": 0.0,
        "experiment_seconds": np.nan,
        "notes": "saved Phase 7 baseline validation result; not retrained in Phase 9",
    }
    phase9_experiment_results_with_baseline = pd.concat(
        [pd.DataFrame([baseline_reference_row]), phase9_experiment_results], ignore_index=True
    )

    best_payload = best_experiment_payload
    best_config = best_payload["config"]
    best_result = best_payload["result_row"]
    best_validation_metrics = best_payload["validation_metrics"].copy()
    for metric_column in ["precision", "recall", "f1_score"]:
        best_validation_metrics[metric_column] = best_validation_metrics[metric_column].round(4)
    best_validation_confusion_matrix = pd.DataFrame(
        best_payload["validation_confusion_matrix"], index=class_names, columns=class_names
    )
    best_validation_confusion_matrix.index.name = "actual_label"

    model_checkpoint = {
        "model_state_dict": best_payload["state_dict"],
        "model_config": {
            "model_name": "phase9_best_validation_rnn",
            "experiment_id": best_payload["experiment_id"],
            "vocab_size": vocab_size,
            "embedding_dim": int(best_config["embedding_dim"]),
            "hidden_dim": int(best_config["hidden_dim"]),
            "recurrent_type": best_config["recurrent_type"],
            "dropout_rate": float(best_config["dropout_rate"]),
            "batch_size": int(best_config["batch_size"]),
            "padding_idx": padding_idx,
            "num_classes": num_classes,
            "class_weights": best_payload["class_weights"],
        },
        "phase6_config": phase6_config,
        "class_names": class_names,
        "selection_metric": "validation_macro_f1",
        "best_epoch": int(best_result["best_epoch"]),
        "best_validation_macro_f1": float(best_result["best_validation_macro_f1"]),
        "phase7_baseline_validation_macro_f1": baseline_validation_macro_f1,
        "test_split_used": False,
    }
    torch.save(model_checkpoint, PHASE9_BEST_CHECKPOINT_PATH)
    phase9_metadata = {
        "best_experiment_id": best_payload["experiment_id"],
        "best_checkpoint_path": str(PHASE9_BEST_CHECKPOINT_PATH),
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": float(best_result["best_validation_macro_f1"]),
        "phase7_baseline_validation_macro_f1": baseline_validation_macro_f1,
        "validation_macro_f1_improvement": float(best_result["improvement_over_phase7_validation_macro_f1"]),
        "phase8_test_macro_f1_reference_only": phase8_test_macro_f1,
        "total_experiment_seconds": round(float(phase9_total_seconds), 2),
        "test_split_used": False,
        "best_config": json_ready(best_config),
        "class_names": class_names,
    }
    PHASE9_BEST_METADATA_PATH.write_text(json.dumps(phase9_metadata, indent=2), encoding="utf-8")
    phase9_experiment_results_with_baseline.to_csv(PHASE9_EXPERIMENT_RESULTS_PATH, index=False)
    phase9_training_history.to_csv(PHASE9_TRAINING_HISTORY_PATH, index=False)
    best_validation_metrics.to_csv(PHASE9_BEST_VALIDATION_METRICS_PATH, index=False)
    best_validation_confusion_matrix.to_csv(PHASE9_BEST_CONFUSION_MATRIX_PATH)
    phase9_model_configs = pd.DataFrame(
        [
            {key: (json.dumps(value) if isinstance(value, (dict, list)) else value) for key, value in config.items()}
            for config in experiment_configs
        ]
    )
    phase9_model_configs.to_csv(PHASE9_MODEL_CONFIGS_PATH, index=False)
    save_bar_comparison_svg(phase9_experiment_results_with_baseline, PHASE9_COMPARISON_FIGURE_PATH)
    save_best_learning_curve_svg(phase9_training_history, str(best_payload["experiment_id"]), PHASE9_BEST_LEARNING_CURVE_PATH)

    best_experiment_id = str(best_payload["experiment_id"])
    best_macro_f1 = float(best_result["best_validation_macro_f1"])
    improvement_delta = float(best_result["improvement_over_phase7_validation_macro_f1"])
    best_neutral_f1 = float(best_result["neutral_f1"])
    phase9_summary_lines = [
        "# Phase 9 Model Improvement Summary",
        "",
        f"- Source sequence arrays: {PHASE6_SEQUENCES_PATH}",
        f"- Baseline validation metrics: {PHASE7_VALIDATION_METRICS_PATH}",
        f"- Held-out test metrics used as reference only: {PHASE8_TEST_METRICS_PATH}",
        f"- Best validation checkpoint: {PHASE9_BEST_CHECKPOINT_PATH}",
        f"- Best validation metadata: {PHASE9_BEST_METADATA_PATH}",
        f"- Experiment results: {PHASE9_EXPERIMENT_RESULTS_PATH}",
        f"- Training history: {PHASE9_TRAINING_HISTORY_PATH}",
        f"- Best validation metrics: {PHASE9_BEST_VALIDATION_METRICS_PATH}",
        f"- Best validation confusion matrix: {PHASE9_BEST_CONFUSION_MATRIX_PATH}",
        f"- Comparison figure: {PHASE9_COMPARISON_FIGURE_PATH}",
        f"- Best learning curve: {PHASE9_BEST_LEARNING_CURVE_PATH}",
        f"- Candidate experiments run: {len(experiment_configs)}",
        f"- Best experiment by validation macro F1: {best_experiment_id}",
        f"- Phase 7 baseline validation macro F1: {baseline_validation_macro_f1:.4f}",
        f"- Best Phase 9 validation macro F1: {best_macro_f1:.4f}",
        f"- Validation macro F1 delta: {improvement_delta:+.4f}",
        f"- Best Phase 9 neutral F1: {best_neutral_f1:.4f}",
        f"- Phase 8 held-out test accuracy reference: {baseline_test_accuracy:.4f}",
        f"- Phase 8 held-out test macro F1 reference: {phase8_test_macro_f1:.4f}",
        f"- Total experiment time: {phase9_total_seconds:.1f} seconds",
        "- Held-out test split used in this phase: no.",
        "",
        "## Experiment Notes",
        "",
    ]
    for row in phase9_experiment_results.itertuples(index=False):
        phase9_summary_lines.append(
            f"- {row.experiment_id}: validation macro F1 {row.best_validation_macro_f1:.4f}, "
            f"neutral F1 {row.neutral_f1:.4f}, delta {row.improvement_over_phase7_validation_macro_f1:+.4f}."
        )
    phase9_summary_lines.extend(
        [
            "",
            "## Carry-Forward Notes",
            "",
            "- Phase 10 can present the Phase 7/8 baseline and Phase 9 validation improvement as separate, clearly labeled results.",
            "- Do not replace the Phase 8 held-out test result unless a final improved model is selected from validation evidence first.",
            "- The best Phase 9 checkpoint can be used for one final test-set comparison only if the project requires an improved final-model test result.",
        ]
    )
    PHASE9_SUMMARY_PATH.write_text("\n".join(phase9_summary_lines) + "\n", encoding="utf-8")

    print("Phase 9 validation comparison:")
    print(phase9_experiment_results_with_baseline.to_string(index=False))
    print(f"Best Phase 9 candidate: {best_experiment_id}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")
    print(f"Improvement over Phase 7 validation macro F1: {improvement_delta:+.4f}")
    print(f"Total Phase 9 experiment time: {phase9_total_seconds:.1f}s")
    print(f"Saved Phase 9 summary to {PHASE9_SUMMARY_PATH}")
    return phase9_metadata


if __name__ == "__main__":
    main()
