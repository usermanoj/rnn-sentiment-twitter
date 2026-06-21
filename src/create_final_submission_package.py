"""Create the final PDF report and submission zip.

This script packages existing project outputs only. It does not train, tune,
evaluate, or otherwise modify model evidence.
"""

from __future__ import annotations

import csv
import json
import shutil
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
REPORTS = OUTPUTS / "reports"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
DATA = OUTPUTS / "data"
SUBMISSION = OUTPUTS / "submission"
TMP = ROOT / "tmp" / "pdfs" / "final_submission"

PDF_PATH = SUBMISSION / "Module_31_Graded_Mini_Project_Final_Report.pdf"
CANONICAL_PDF_PATH = SUBMISSION / "Module 31 - Graded Mini Project_Bhardwaj.pdf"
ZIP_PATH = SUBMISSION / "Module_31_Graded_Mini_Project_Submission.zip"
README_PATH = SUBMISSION / "README_SUBMISSION.md"
MANIFEST_PATH = SUBMISSION / "submission_manifest.csv"

NEG = "#D75A4A"
NEU = "#6877C8"
POS = "#2E9F72"
DARK = "#17202A"
MUTED = "#59616E"
GRID = "#E6E9EF"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric(rows: list[dict[str, str]], label: str, column: str) -> str:
    for row in rows:
        if row.get("label") == label:
            return row[column]
    raise KeyError(f"Metric row not found: {label}.{column}")


def setting(rows: list[dict[str, str]], key: str, key_col: str = "metric") -> str:
    for row in rows:
        if row.get(key_col) == key:
            return row["value"]
    raise KeyError(key)


def as_float(value: str) -> float:
    return float(str(value).replace(",", ""))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def short_num(value: str | float) -> str:
    number = as_float(str(value))
    if abs(number - int(number)) < 1e-9:
        return f"{int(number):,}"
    return f"{number:.4f}"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    fnt: ImageFont.ImageFont,
    max_width: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    line_height = draw.textbbox((0, 0), "Ag", font=fnt)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, fill=fill, font=fnt)
        y += line_height
    return y


def create_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    value_suffix: str = "",
    bar_colors: list[str] | None = None,
    y_floor: float = 0.0,
) -> None:
    width, height = 1200, 620
    margin_l, margin_r, margin_t, margin_b = 105, 55, 95, 110
    img = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(22)
    small_font = font(19)
    draw.text((margin_l, 34), title, fill=DARK, font=title_font)

    max_value = max(values) if values else 1
    y_max = max_value * 1.18
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    plot_bottom = margin_t + chart_h
    draw.line((margin_l, margin_t, margin_l, plot_bottom), fill="#B8C0CC", width=2)
    draw.line((margin_l, plot_bottom, width - margin_r, plot_bottom), fill="#B8C0CC", width=2)
    for i in range(5):
        y = plot_bottom - int(chart_h * i / 4)
        draw.line((margin_l, y, width - margin_r, y), fill=GRID, width=1)
        tick = y_floor + (y_max - y_floor) * i / 4
        draw.text((20, y - 12), f"{tick:.2f}{value_suffix}", fill=MUTED, font=small_font)

    n = len(labels)
    slot = chart_w / max(n, 1)
    bar_w = int(slot * 0.58)
    colors_used = bar_colors or ["#4C78A8"] * n
    for i, (label, value) in enumerate(zip(labels, values)):
        x0 = int(margin_l + slot * i + (slot - bar_w) / 2)
        bar_h = int(chart_h * (value - y_floor) / max(y_max - y_floor, 1e-9))
        y0 = plot_bottom - bar_h
        draw.rounded_rectangle((x0, y0, x0 + bar_w, plot_bottom), radius=10, fill=colors_used[i])
        value_text = f"{value:.4f}" if value < 1 else f"{value:,.0f}"
        if value_suffix:
            value_text += value_suffix
        bbox = draw.textbbox((0, 0), value_text, font=small_font)
        draw.text((x0 + bar_w / 2 - (bbox[2] - bbox[0]) / 2, y0 - 32), value_text, fill=DARK, font=small_font)
        wrapped = textwrap.wrap(label, width=14)[:2]
        y_label = plot_bottom + 18
        for part in wrapped:
            bbox = draw.textbbox((0, 0), part, font=label_font)
            draw.text((x0 + bar_w / 2 - (bbox[2] - bbox[0]) / 2, y_label), part, fill=DARK, font=label_font)
            y_label += 27
    img.save(path)


def create_line_chart(path: Path, rows: list[dict[str, str]], title: str) -> None:
    width, height = 1200, 620
    margin_l, margin_r, margin_t, margin_b = 105, 60, 95, 90
    img = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(22)
    small_font = font(19)
    draw.text((margin_l, 34), title, fill=DARK, font=title_font)
    epochs = [int(r["epoch"]) for r in rows]
    train = [float(r["train_macro_f1"]) for r in rows]
    val = [float(r["validation_macro_f1"]) for r in rows]
    series = [("Train macro F1", train, "#6E6E6E"), ("Validation macro F1", val, POS)]
    all_values = train + val
    y_min = max(0.0, min(all_values) - 0.08)
    y_max = min(1.0, max(all_values) + 0.08)
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    bottom = margin_t + chart_h
    right = width - margin_r
    draw.line((margin_l, margin_t, margin_l, bottom), fill="#B8C0CC", width=2)
    draw.line((margin_l, bottom, right, bottom), fill="#B8C0CC", width=2)
    for i in range(5):
        y = bottom - int(chart_h * i / 4)
        tick = y_min + (y_max - y_min) * i / 4
        draw.line((margin_l, y, right, y), fill=GRID, width=1)
        draw.text((22, y - 12), f"{tick:.2f}", fill=MUTED, font=small_font)
    x_positions = []
    for epoch in epochs:
        x = margin_l + int(chart_w * (epoch - min(epochs)) / max(max(epochs) - min(epochs), 1))
        x_positions.append(x)
        draw.text((x - 8, bottom + 22), str(epoch), fill=DARK, font=label_font)
    draw.text((right - 90, bottom + 22), "epoch", fill=MUTED, font=small_font)
    for name, values, color in series:
        points = []
        for x, value in zip(x_positions, values):
            y = bottom - int(chart_h * (value - y_min) / max(y_max - y_min, 1e-9))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    legend_y = height - 42
    legend_x = margin_l
    for name, _, color in series:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 30, legend_y + 15), radius=4, fill=color)
        draw.text((legend_x + 40, legend_y - 4), name, fill=DARK, font=small_font)
        legend_x += 250
    img.save(path)


def create_confusion_matrix(path: Path, matrix_rows: list[dict[str, str]]) -> None:
    labels = ["Negative", "Neutral", "Positive"]
    values = [[int(row[label]) for label in labels] for row in matrix_rows]
    max_value = max(max(row) for row in values)
    width, height = 1120, 760
    img = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(23, True)
    cell_font = font(30, True)
    small_font = font(20)
    draw.text((80, 35), "Held-Out Test Confusion Matrix", fill=DARK, font=title_font)
    x0, y0, cell = 315, 165, 165
    draw.text((x0 + 155, 105), "Predicted label", fill=MUTED, font=small_font)
    draw.text((80, y0 + 220), "Actual label", fill=MUTED, font=small_font)
    for j, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x0 + j * cell + cell / 2 - (bbox[2] - bbox[0]) / 2, y0 - 42), label, fill=DARK, font=label_font)
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x0 - 35 - (bbox[2] - bbox[0]), y0 + i * cell + 64), label, fill=DARK, font=label_font)
        for j, value in enumerate(values[i]):
            intensity = value / max_value
            base = int(245 - 105 * intensity)
            fill = (base, int(base + 10), 255)
            if i == j:
                fill = (int(224 - 120 * intensity), int(248 - 80 * intensity), int(231 - 90 * intensity))
            x = x0 + j * cell
            y = y0 + i * cell
            draw.rounded_rectangle((x + 6, y + 6, x + cell - 6, y + cell - 6), radius=12, fill=fill, outline="#FFFFFF", width=3)
            text = f"{value:,}"
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            draw.text((x + cell / 2 - (bbox[2] - bbox[0]) / 2, y + 56), text, fill=DARK, font=cell_font)
    draw.text((80, 715), "Largest error: Neutral predicted as Negative = 757 rows.", fill=MUTED, font=small_font)
    img.save(path)


def create_length_chart(path: Path, rows: list[dict[str, str]]) -> None:
    labels = [row["sentiment"] for row in rows]
    means = [float(row["model_token_mean"]) for row in rows]
    medians = [float(row["model_token_median"]) for row in rows]
    width, height = 1200, 620
    margin_l, margin_r, margin_t, margin_b = 105, 55, 95, 110
    img = PILImage.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(34, True)
    label_font = font(22)
    small_font = font(19)
    draw.text((margin_l, 34), "Tweet Length By Sentiment", fill=DARK, font=title_font)
    max_value = max(means + medians) * 1.25
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    bottom = margin_t + chart_h
    draw.line((margin_l, margin_t, margin_l, bottom), fill="#B8C0CC", width=2)
    draw.line((margin_l, bottom, width - margin_r, bottom), fill="#B8C0CC", width=2)
    for i in range(5):
        y = bottom - int(chart_h * i / 4)
        tick = max_value * i / 4
        draw.line((margin_l, y, width - margin_r, y), fill=GRID, width=1)
        draw.text((30, y - 12), f"{tick:.0f}", fill=MUTED, font=small_font)
    slot = chart_w / len(labels)
    bar_w = int(slot * 0.25)
    for i, label in enumerate(labels):
        group_x = margin_l + int(slot * i + slot * 0.22)
        for k, (name, value, color) in enumerate(
            [("Mean", means[i], "#4C78A8"), ("Median", medians[i], "#F2A541")]
        ):
            x = group_x + k * (bar_w + 10)
            bar_h = int(chart_h * value / max_value)
            y = bottom - bar_h
            draw.rounded_rectangle((x, y, x + bar_w, bottom), radius=8, fill=color)
            draw.text((x + 4, y - 28), f"{value:.1f}", fill=DARK, font=small_font)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((group_x + bar_w - (bbox[2] - bbox[0]) / 2, bottom + 20), label, fill=DARK, font=label_font)
    legend_y = height - 42
    for i, (name, color) in enumerate([("Mean tokens", "#4C78A8"), ("Median tokens", "#F2A541")]):
        x = margin_l + i * 210
        draw.rounded_rectangle((x, legend_y, x + 30, legend_y + 15), radius=4, fill=color)
        draw.text((x + 40, legend_y - 4), name, fill=DARK, font=small_font)
    img.save(path)


def create_charts() -> dict[str, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    sentiment_rows = read_csv(TABLES / "phase5_sentiment_distribution.csv")
    phase7_history = read_csv(TABLES / "phase7_training_history.csv")
    phase8_matrix = read_csv(TABLES / "phase8_test_confusion_matrix.csv")
    length_rows = read_csv(TABLES / "phase5_length_by_sentiment.csv")
    phase9_results = read_csv(TABLES / "phase9_experiment_results.csv")

    charts = {
        "sentiment": TMP / "sentiment_distribution.png",
        "length": TMP / "tweet_length_by_sentiment.png",
        "learning": TMP / "phase7_learning_curve.png",
        "confusion": TMP / "phase8_confusion_matrix.png",
        "improvement": TMP / "phase9_validation_comparison.png",
    }
    sentiment_colors = [{"Negative": NEG, "Neutral": NEU, "Positive": POS}[row["sentiment"]] for row in sentiment_rows]
    create_bar_chart(
        charts["sentiment"],
        "Sentiment Distribution",
        [row["sentiment"] for row in sentiment_rows],
        [float(row["percent"]) for row in sentiment_rows],
        value_suffix="%",
        bar_colors=sentiment_colors,
    )
    create_length_chart(charts["length"], length_rows)
    create_line_chart(charts["learning"], phase7_history, "Baseline GRU Learning Curve")
    create_confusion_matrix(charts["confusion"], phase8_matrix)
    create_bar_chart(
        charts["improvement"],
        "Validation Macro F1 By Candidate",
        [row["experiment_id"].replace("_", " ") for row in phase9_results],
        [float(row["best_validation_macro_f1"]) for row in phase9_results],
        bar_colors=["#8E99A8", POS, "#4C78A8", "#F2A541"],
        y_floor=0.55,
    )
    return charts


def report_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=31,
            textColor=colors.HexColor(DARK),
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor(MUTED),
            alignment=TA_CENTER,
            spaceAfter=24,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=22,
            textColor=colors.HexColor(DARK),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor(DARK),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=9.7,
            leading=13.5,
            textColor=colors.HexColor("#2B3036"),
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor(MUTED),
            spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontSize=9.3,
            leading=12.5,
            leftIndent=10,
            bulletIndent=0,
            textColor=colors.HexColor("#2B3036"),
        ),
        "Metric": ParagraphStyle(
            "Metric",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor(DARK),
        ),
        "MetricLabel": ParagraphStyle(
            "MetricLabel",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor(MUTED),
        ),
    }
    return styles


def table_style(header_fill: str = "#17202A") -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_fill)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD2DD")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def bullet_items(items: list[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles["Bullet"]), bulletColor=colors.HexColor(POS)) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
    )


def image(path: Path, width_cm: float) -> Image:
    with PILImage.open(path) as source:
        pixel_width, pixel_height = source.size
    draw_width = width_cm * cm
    draw_height = draw_width * (pixel_height / float(pixel_width))
    return Image(str(path), width=draw_width, height=draw_height)


def make_metric_cards(rows: list[tuple[str, str]], styles: dict[str, ParagraphStyle]) -> Table:
    cells = []
    for value, label in rows:
        cells.append([Paragraph(value, styles["Metric"]), Paragraph(label, styles["MetricLabel"])])
    table_data = []
    for value, label in rows:
        table_data.append([Paragraph(value, styles["Metric"]), Paragraph(label, styles["MetricLabel"])])
    card_tables = []
    for value, label in rows:
        card_tables.append(
            Table(
                [[Paragraph(value, styles["Metric"])], [Paragraph(label, styles["MetricLabel"])]],
                colWidths=[4.1 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FB")),
                        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D8DEE8")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            )
        )
    outer = Table([card_tables], colWidths=[4.25 * cm] * len(rows))
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return outer


def paragraph_table(
    rows: list[list[str]],
    col_widths: list[float],
    styles: dict[str, ParagraphStyle],
    header_fill: str = "#17202A",
) -> Table:
    data = []
    for row_index, row in enumerate(rows):
        style = styles["Small"] if row_index else ParagraphStyle("HeaderCell", parent=styles["Small"], textColor=colors.white, fontName="Helvetica-Bold")
        data.append([Paragraph(str(cell).replace("&", "&amp;"), style) for cell in row])
    table = Table(data, colWidths=[w * cm for w in col_widths], repeatRows=1)
    table.setStyle(table_style(header_fill))
    return table


def top_tokens_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = read_csv(TABLES / "phase5_top_tokens_by_sentiment.csv")
    sentiments = ["Negative", "Neutral", "Positive"]
    table = [["Sentiment", "Top terms from cleaned analysis tokens"]]
    for sentiment in sentiments:
        terms = [r["token"] for r in rows if r["sentiment"] == sentiment and int(r["rank"]) <= 8]
        table.append([sentiment, ", ".join(terms)])
    return paragraph_table(table, [3.1, 13.7], styles)


def sample_predictions(styles: dict[str, ParagraphStyle]) -> Table:
    rows = read_csv(TABLES / "phase8_test_prediction_samples.csv")
    blocked_terms = {"fuck", "fucking", "shit", "niggas"}
    selected: list[dict[str, str]] = []
    for desired_type in ["correct", "incorrect"]:
        for row in rows:
            text = row["model_text"].lower()
            if row["sample_type"] != desired_type:
                continue
            if int(row["model_token_count"]) > 31:
                continue
            if any(term in text.split() for term in blocked_terms):
                continue
            selected.append(row)
            if len([r for r in selected if r["sample_type"] == desired_type]) >= 3:
                break
    table = [["Type", "Sample tweet text", "True", "Predicted", "Confidence"]]
    for row in selected[:6]:
        table.append(
            [
                row["sample_type"].title(),
                textwrap.shorten(row["model_text"], width=112, placeholder="..."),
                row["true_label"],
                row["predicted_label"],
                pct(float(row["predicted_probability"])),
            ]
        )
    return paragraph_table(table, [2.0, 8.4, 2.0, 2.2, 2.2], styles)


def build_pdf(charts: dict[str, Path]) -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    styles = report_styles()

    phase2 = read_csv(TABLES / "phase2_dataset_overview.csv")
    phase3 = read_csv(TABLES / "phase3_cleaning_steps.csv")
    phase5 = read_csv(TABLES / "phase5_sentiment_distribution.csv")
    phase6_vocab = read_csv(TABLES / "phase6_vocabulary_summary.csv")
    phase7_config = read_csv(TABLES / "phase7_model_config.csv")
    phase7_metrics = read_csv(TABLES / "phase7_validation_metrics.csv")
    phase8_metrics = read_csv(TABLES / "phase8_test_metrics.csv")
    phase9 = read_csv(TABLES / "phase9_experiment_results.csv")
    phase9_best = max(phase9, key=lambda row: float(row["best_validation_macro_f1"]))
    key_metrics = read_csv(TABLES / "phase10_final_key_metrics.csv")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.25 * cm,
        bottomMargin=1.25 * cm,
        title="Module 31 Graded Mini Project Final Report",
        author="RNN Twitter Sentiment Analysis Project",
    )

    story = []
    story.append(Spacer(1, 1.4 * cm))
    story.append(p("Module 31: Graded Mini Project", styles["Title"]))
    story.append(p("RNN Twitter Sentiment Analysis - Final Submission Report", styles["Subtitle"]))
    story.append(make_metric_cards(
        [
            ("58,841", "cleaned rows"),
            ("0.6262", "held-out test macro F1"),
            ("0.6874", "best validation macro F1"),
            ("+0.0689", "validation macro F1 lift"),
        ],
        styles,
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(p(
        "This report documents the end-to-end sentiment analysis pipeline: data ingestion, cleaning, exploratory analysis, sequence feature engineering, GRU-based RNN modeling, held-out evaluation, and validation-driven model improvement.",
        styles["Body"],
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(paragraph_table(
        [
            ["Submission file", "Purpose"],
            ["Final PDF report", "Primary document for grading and PDF upload."],
            ["Jupyter notebook", "Full code path and phase-by-phase implementation."],
            ["Executive presentation", "Class-facing summary deck."],
            ["Artifacts zip", "Metrics, figures, model checkpoints, manifest, and supporting evidence."],
        ],
        [4.5, 12.3],
        styles,
    ))
    story.append(Spacer(1, 0.8 * cm))
    story.append(p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Small"]))
    story.append(PageBreak())

    story.append(p("Executive Summary", styles["H1"]))
    story.append(p(
        "The project delivers a complete RNN sentiment classifier for a three-class Twitter sentiment task: Negative, Neutral, and Positive. The official held-out evidence is the Phase 8 baseline GRU result, evaluated once on the reserved test split. Phase 9 then performs validation-only improvement experiments and selects a stronger GRU candidate without reusing the test set.",
        styles["Body"],
    ))
    story.append(bullet_items(
        [
            "Data pipeline: raw Twitter rows were validated, cleaned, tokenized, normalized, and split with duplicate-text leakage controls.",
            "Baseline model: embedding plus GRU classifier with dropout, class-weighted loss, checkpointing, and validation macro F1 monitoring.",
            "Held-out result: 0.6387 test accuracy and 0.6262 test macro F1 for the audited Phase 7 baseline checkpoint.",
            "Improvement result: GRU-96 with dropout 0.20 reached 0.6874 validation macro F1, a +0.0689 lift over the Phase 7 validation baseline.",
            "Main limitation: Neutral sentiment is the weakest class and is often absorbed into Negative or Positive predictions.",
        ],
        styles,
    ))
    story.append(Spacer(1, 0.25 * cm))
    story.append(paragraph_table(
        [["Area", "Metric", "Value"]] + [[r["area"], r["metric"], r["value"]] for r in key_metrics],
        [3.2, 8.4, 5.2],
        styles,
    ))

    story.append(p("Rubric Coverage", styles["H1"]))
    rubric = [
        ["Rubric item", "Status", "Evidence"],
        ["Load dataset", "Complete", "Raw CSV loaded with explicit column names, shape, schema, dtypes, missingness, and sample rows."],
        ["Data cleaning", "Complete", "Missing text and duplicate rows handled; URLs, mentions, hashtags, special characters, lowercasing, stop words, and stemming documented."],
        ["Feature engineering", "Complete", "Training-derived vocabulary, padded token sequences, label mapping, TF-IDF vocabulary, and sequence arrays saved."],
        ["EDA", "Complete", "Sentiment distribution, top terms, word clouds, entity frequency, and tweet-length analysis produced."],
        ["RNN model", "Complete", "Embedding plus GRU baseline trained with dropout, class weights, validation monitoring, and saved checkpoint."],
        ["Evaluation", "Complete", "Accuracy, precision, recall, F1, confusion matrix, learning curves, prediction distributions, and sample predictions saved."],
        ["Improvement", "Complete", "Controlled validation experiments compare larger GRU, neutral weighting, and LSTM candidates."],
        ["Presentation/demo", "Complete in package", "Executive PPTX plus this report's sample prediction demo table."],
    ]
    story.append(paragraph_table(rubric, [4.1, 2.3, 10.4], styles))
    story.append(PageBreak())

    story.append(p("Part 1 - Data Processing", styles["H1"]))
    story.append(p(
        "The raw dataset contains tweet IDs, entities, sentiment labels, and tweet text. The pipeline preserves the raw CSV and creates cleaned downstream datasets for modeling and EDA.",
        styles["Body"],
    ))
    overview_table = [["Metric", "Value"]] + [[r["metric"], r["value"]] for r in phase2]
    story.append(paragraph_table(overview_table, [7.0, 9.8], styles))
    story.append(Spacer(1, 0.25 * cm))
    story.append(paragraph_table(
        [["Cleaning step", "Rows removed", "Rows remaining"]]
        + [[r["step"], r["rows_removed"], r["rows_remaining"]] for r in phase3],
        [7.4, 4.0, 5.4],
        styles,
    ))
    story.append(p(
        "Final cleaned rows: 58,841. The Irrelevant label was excluded because the assignment defines a three-class Positive, Negative, and Neutral sentiment task.",
        styles["Body"],
    ))
    story.append(p("Feature Engineering", styles["H2"]))
    story.append(bullet_items(
        [
            f"Vocabulary built from training rows only: {setting(phase6_vocab, 'final_vocabulary_size')} total tokens including PAD and OOV.",
            "Texts were converted into fixed-length token ID sequences with a max sequence length of 60.",
            "Train, validation, and test splits were grouped by model text to prevent duplicate cleaned text from crossing split boundaries.",
            "TF-IDF vocabulary terms were also saved as a numerical representation and EDA comparison artifact.",
        ],
        styles,
    ))

    story.append(p("Part 2 - Exploratory Data Analysis", styles["H1"]))
    story.append(image(charts["sentiment"], 16.6))
    story.append(Spacer(1, 0.15 * cm))
    story.append(paragraph_table(
        [["Sentiment", "Count", "Percent"]]
        + [[r["sentiment"], f"{int(r['count']):,}", f"{float(r['percent']):.2f}%"] for r in phase5],
        [4.5, 5.5, 6.8],
        styles,
    ))
    story.append(Spacer(1, 0.25 * cm))
    story.append(KeepTogether([p("Top Terms By Sentiment", styles["H2"]), top_tokens_table(styles)]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(image(charts["length"], 16.6))
    story.append(p(
        "EDA shows a moderately imbalanced but workable class distribution. Negative is the largest class, Positive is close behind, and Neutral is the smallest and most ambiguous class. The word-cloud SVGs and detailed top-token tables are included in the submission package.",
        styles["Body"],
    ))
    story.append(PageBreak())

    story.append(p("Part 3 - RNN Model Development", styles["H1"]))
    config_rows = {row["setting"]: row["value"] for row in phase7_config}
    story.append(paragraph_table(
        [
            ["Model element", "Chosen setting"],
            ["Architecture", "Embedding layer followed by single-layer GRU classifier"],
            ["Vocabulary size", config_rows["vocab_size"]],
            ["Embedding dimension", config_rows["embedding_dim"]],
            ["Hidden dimension", config_rows["hidden_dim"]],
            ["Dropout", config_rows["dropout_rate"]],
            ["Batch size", config_rows["batch_size"]],
            ["Optimizer settings", f"Learning rate {config_rows['learning_rate']}; weight decay {config_rows['weight_decay']}"],
            ["Trainable parameters", f"{int(config_rows['trainable_parameter_count']):,}"],
        ],
        [5.2, 11.6],
        styles,
    ))
    story.append(Spacer(1, 0.25 * cm))
    story.append(image(charts["learning"], 16.6))
    story.append(p(
        "The baseline improves steadily across three epochs. Checkpoint selection used validation macro F1, keeping the held-out test split reserved for Phase 8.",
        styles["Body"],
    ))
    story.append(p("Validation Metrics", styles["H2"]))
    story.append(paragraph_table(
        [["Label", "Precision", "Recall", "F1", "Support"]]
        + [[r["label"], r["precision"], r["recall"], r["f1_score"], r["support"]] for r in phase7_metrics],
        [4.0, 3.2, 3.2, 3.2, 3.2],
        styles,
    ))

    story.append(p("Held-Out Test Evaluation", styles["H1"]))
    story.append(make_metric_cards(
        [
            (metric(phase8_metrics, "accuracy", "f1_score"), "test accuracy"),
            (metric(phase8_metrics, "macro_avg", "f1_score"), "test macro F1"),
            (metric(phase8_metrics, "weighted_avg", "f1_score"), "test weighted F1"),
            (metric(phase8_metrics, "Neutral", "f1_score"), "Neutral F1"),
        ],
        styles,
    ))
    story.append(Spacer(1, 0.25 * cm))
    story.append(image(charts["confusion"], 16.3))
    story.append(paragraph_table(
        [["Label", "Precision", "Recall", "F1", "Support"]]
        + [[r["label"], r["precision"], r["recall"], r["f1_score"], r["support"]] for r in phase8_metrics if r["label"] != "weighted_cross_entropy_loss"],
        [4.0, 3.2, 3.2, 3.2, 3.2],
        styles,
    ))
    story.append(p(
        "Neutral is the clearest weakness. The largest error pattern is Neutral predicted as Negative, which matches the qualitative challenge of brief entity-specific tweets with limited context.",
        styles["Body"],
    ))
    story.append(PageBreak())

    story.append(p("Model Improvement", styles["H1"]))
    story.append(p(
        "Phase 9 uses a validation-only model selection process. It does not tune on the held-out test split. Three candidates are compared against the Phase 7 validation baseline: a larger GRU with lower dropout, a larger GRU with extra Neutral weighting, and an LSTM alternative.",
        styles["Body"],
    ))
    story.append(image(charts["improvement"], 16.6))
    story.append(Spacer(1, 0.15 * cm))
    story.append(paragraph_table(
        [["Experiment", "Cell", "Hidden", "Dropout", "Validation macro F1", "Neutral F1", "Notes"]]
        + [
            [
                r["experiment_id"],
                r["recurrent_type"],
                r["hidden_dim"],
                r["dropout_rate"],
                f"{float(r['best_validation_macro_f1']):.4f}",
                f"{float(r['neutral_f1']):.4f}",
                r["notes"],
            ]
            for r in phase9
        ],
        [4.2, 1.7, 1.8, 1.8, 2.8, 2.2, 2.3],
        styles,
    ))
    story.append(p(
        f"Selected candidate: {phase9_best['experiment_id']} with validation macro F1 {float(phase9_best['best_validation_macro_f1']):.4f}. The test split should be used for this selected candidate exactly once only if the assessment requires a final improved-model test result.",
        styles["Body"],
    ))

    story.append(p("Sample Tweet Prediction Demo", styles["H1"]))
    story.append(p(
        "The table below demonstrates how the saved baseline model behaves on held-out sample tweets. It includes both correct and incorrect predictions to show realistic model behavior rather than only best-case examples.",
        styles["Body"],
    ))
    story.append(sample_predictions(styles))
    story.append(Spacer(1, 0.25 * cm))
    story.append(p(
        "Full prediction samples are saved in outputs/tables/phase8_test_prediction_samples.csv, and all held-out predictions are saved in outputs/data/phase8_test_predictions.csv.",
        styles["Small"],
    ))

    story.append(p("Conclusion And Recommendations", styles["H1"]))
    story.append(bullet_items(
        [
            "Submit the Phase 8 held-out test metrics as the audited baseline result.",
            "Present the Phase 9 GRU-96 result as a validation-selected improvement candidate unless one final test evaluation is explicitly required.",
            "Future improvement should focus on Neutral recall through richer contextual embeddings, neutral-specific data review, and calibrated thresholds.",
            "The package includes the notebook, final report, deck, saved metrics, figures, and checkpoints needed for review.",
        ],
        styles,
    ))
    story.append(p("Artifact Index", styles["H2"]))
    story.append(paragraph_table(
        [
            ["Artifact", "Description"],
            ["notebooks/RNN_Sentiment_Analysis_Twitter.ipynb", "End-to-end code notebook covering Phases 1-10."],
            ["outputs/reports/rnn_twitter_sentiment_final_presentation.pptx", "Executive presentation deck."],
            ["outputs/tables/phase10_final_key_metrics.csv", "Final metric summary."],
            ["outputs/tables/phase8_test_metrics.csv", "Held-out test metrics."],
            ["outputs/tables/phase9_experiment_results.csv", "Validation improvement experiment results."],
            ["outputs/models/*.pt", "Saved PyTorch checkpoints for baseline and selected validation model."],
            ["outputs/figures/*.svg", "EDA, learning curve, confusion matrix, and improvement figures."],
        ],
        [8.6, 8.2],
        styles,
    ))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(1.45 * cm, 0.7 * cm, "RNN Twitter Sentiment Analysis - Module 31 Graded Mini Project")
        canvas.drawRightString(A4[0] - 1.45 * cm, 0.7 * cm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build_readme_and_manifest(package_entries: list[Path]) -> None:
    rows = []
    for path in sorted(package_entries):
        rows.append(
            {
                "path": rel(path),
                "bytes": str(path.stat().st_size),
                "purpose": classify_artifact(path),
            }
        )
    write_csv(MANIFEST_PATH, rows, ["path", "bytes", "purpose"])
    README_PATH.write_text(
        "\n".join(
            [
                "# Module 31 Graded Mini Project Submission Package",
                "",
                "Primary file for grading:",
                f"- {CANONICAL_PDF_PATH.name}",
                "",
                "Also included:",
                f"- {PDF_PATH.name} as an internal descriptive copy of the same final report.",
                "- Jupyter notebook with the complete phase-by-phase implementation.",
                "- Final executive PowerPoint presentation.",
                "- Metrics tables, figures, reports, model metadata, and model checkpoints.",
                "- Selected feature artifacts and held-out prediction outputs for auditability.",
                "",
                "Important evaluation note:",
                "- The package uses existing project outputs only. It does not run additional training, tuning, or test-set evaluation.",
                "- Phase 8 is the official held-out test baseline result.",
                "- Phase 9 is a validation-selected improvement candidate and keeps the held-out test split protected.",
                "",
                f"Manifest: {MANIFEST_PATH.name}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def classify_artifact(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if path == PDF_PATH:
        return "primary final PDF report"
    if suffix == ".ipynb":
        return "notebook source and implementation"
    if suffix == ".pptx":
        return "final presentation deck"
    if suffix == ".csv":
        return "metrics, audit table, or prediction output"
    if suffix == ".svg":
        return "figure or visualization"
    if suffix == ".pt":
        return "saved PyTorch model checkpoint"
    if suffix == ".json":
        return "configuration, vocabulary, or metadata"
    if suffix == ".py":
        return "source script"
    if suffix == ".md":
        return "written summary or package documentation"
    if name == "requirements.txt":
        return "Python dependency list"
    return "supporting artifact"


def collect_package_entries() -> list[Path]:
    entries: list[Path] = [
        CANONICAL_PDF_PATH,
        PDF_PATH,
        ROOT / "README.md",
        README_PATH,
        MANIFEST_PATH,
        ROOT / "requirements.txt",
        ROOT / "PROJECT_CONTEXT.md",
        ROOT / "twitter_training.csv",
        ROOT / "notebooks" / "RNN_Sentiment_Analysis_Twitter.ipynb",
        ROOT / "src" / "__init__.py",
        ROOT / "src" / "phase9_model_improvement.py",
        ROOT / "src" / "phase10_final_deliverables.py",
        ROOT / "src" / "create_final_submission_package.py",
        REPORTS / "rnn_twitter_sentiment_final_presentation.pptx",
        DATA / "phase6_feature_config.json",
        DATA / "phase6_sequences.npz",
        DATA / "phase6_tfidf_vocabulary.csv",
        DATA / "phase6_vocabulary.json",
        DATA / "phase8_test_predictions.csv",
    ]
    entries.extend(sorted(REPORTS.glob("*.md")))
    entries.extend(sorted(TABLES.glob("*.csv")))
    entries.extend(sorted(FIGURES.glob("*.svg")))
    entries.extend(sorted(MODELS.glob("*.json")))
    entries.extend(sorted(MODELS.glob("*.pt")))
    return sorted({path for path in entries if path.exists()})


def build_zip(package_entries: list[Path]) -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in package_entries:
            archive.write(path, arcname=rel(path))


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    charts = create_charts()
    build_pdf(charts)
    shutil.copy2(PDF_PATH, CANONICAL_PDF_PATH)
    provisional_entries = collect_package_entries()
    build_readme_and_manifest(provisional_entries)
    package_entries = collect_package_entries()
    build_readme_and_manifest(package_entries)
    package_entries = collect_package_entries()
    build_zip(package_entries)
    print(f"Wrote final PDF: {PDF_PATH}")
    print(f"Wrote package README: {README_PATH}")
    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Wrote submission zip: {ZIP_PATH}")
    print(f"Packaged files: {len(package_entries)}")
    print(f"Zip size bytes: {ZIP_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
