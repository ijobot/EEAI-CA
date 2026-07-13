# pre-label metrics, multi-output metrics, error analysis

# This file replaces the prototype's classification_report in a more structured way.  
# It saves the various outputs to actual files in the /outputs folder for future discovery

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from src.config import Config

logger = logging.getLogger(__name__)

# timestamping actions with iso format
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# metrics
def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    model_name: str = "unknown",
    model_version: str = Config.MODEL_VERSION,
) -> dict:
    labels = labels or Config.CLASS_COLS
    per_label: dict[str, dict] = {}

    for label_index, label in enumerate(labels):
        # true and predicted values for just this one label's column
        true_col = y_true[:, label_index]
        pred_col = y_pred[:, label_index]

        # each class weighted equally
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_col, pred_col, average="macro", zero_division=0
        )

        # classes weighted by their sample count
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            true_col, pred_col, average="weighted", zero_division=0
        )

        per_label[label] = {
            "accuracy": float(accuracy_score(true_col, pred_col)),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "n_classes": int(len(np.unique(true_col))),
            "support": int(len(true_col)),
        }

    overall = {
        # Fraction of individual (row, label) cells predicted wrong.
        "hamming_loss": float((y_true != y_pred).mean()),
        # Fraction of rows where ALL labels are correct (strict).
        "exact_match_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "mean_f1_macro": float(
            np.mean([per_label[label]["f1_macro"] for label in labels])
        ),
    }

    return {
        "model_name": model_name,
        "model_version": model_version,
        "timestamp": _now_iso(),
        "n_test_samples": int(y_true.shape[0]),
        "labels": list(labels),
        "per_label": per_label,
        "overall": overall,
    }

# build the report
def build_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    labels = labels or Config.CLASS_COLS
    rows = []
    for i, label in enumerate(labels):
        report = classification_report(
            y_true[:, i], y_pred[:, i], output_dict=True, zero_division=0
        )
        for cls, scores in report.items():
            if isinstance(scores, dict):
                rows.append(
                    {
                        "label": label,
                        "class": cls,
                        "precision": round(scores["precision"], 4),
                        "recall": round(scores["recall"], 4),
                        "f1": round(scores["f1-score"], 4),
                        "support": int(scores["support"]),
                    }
                )
    return pd.DataFrame(rows)


# error analysis (Task 4)
def collect_misclassifications(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    text_col: str = Config.INTERACTION_CONTENT,
    max_examples: int | None = 50,
    snippet_len: int = 160,
) -> pd.DataFrame:
    # outputs a table for individual errors, one row for each incorrect prediction
    labels = labels or Config.CLASS_COLS
    texts = test_df[text_col].fillna("").astype(str).str.slice(0, snippet_len).to_numpy()

    rows = []
    for i, label in enumerate(labels):
        wrong = np.where(y_true[:, i] != y_pred[:, i])[0]
        for r in wrong:
            rows.append(
                {
                    "row_index": int(test_df.index[r]),
                    "text_snippet": texts[r],
                    "label": label,
                    "true": y_true[r, i],
                    "predicted": y_pred[r, i],
                }
            )
    out = pd.DataFrame(rows)
    if max_examples is not None and len(out) > max_examples:
        out = out.head(max_examples)
    return out


# saving outputs and reports
def save_metrics(metrics: dict, path: Path = Config.METRICS_PATH) -> Path:
    Config.ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics -> %s", path)
    return path


def save_report(report_df: pd.DataFrame, path: Path = Config.CLASSIFICATION_REPORT_PATH) -> Path:
    Config.ensure_dirs()
    report_df.to_csv(path, index=False)
    logger.info("Saved classification report -> %s", path)
    return path

# evaluate, build, and call both of the above save functions
def evaluate_and_save(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    labels: list[str] | None = None,
    metrics_path: Path = Config.METRICS_PATH,
    report_path: Path = Config.CLASSIFICATION_REPORT_PATH,
) -> dict:
    labels = labels or Config.CLASS_COLS
    metrics = evaluate(y_true, y_pred, labels, model_name=model_name)
    report_df = build_classification_report_df(y_true, y_pred, labels)
    save_metrics(metrics, metrics_path)
    save_report(report_df, report_path)
    return metrics


def print_summary(metrics: dict) -> None:
    # Again, got some help from Claude on this one because I wanted to print the output nicely for when you just run the code and aren't looking at the actual reports.
    """Console summary — readable at a glance without opening the JSON."""
    print(f"\nModel: {metrics['model_name']} (v{metrics['model_version']})")
    print(f"Test samples: {metrics['n_test_samples']}")
    print("-" * 60)
    for label, m in metrics["per_label"].items():
        print(
            f"  {label:>3} | acc {m['accuracy']:.3f} | "
            f"F1-macro {m['f1_macro']:.3f} | F1-weighted {m['f1_weighted']:.3f} "
            f"| {m['n_classes']} classes"
        )
    o = metrics["overall"]
    print("-" * 60)
    print(
        f"  overall | hamming {o['hamming_loss']:.3f} | "
        f"exact-match {o['exact_match_accuracy']:.3f} | "
        f"mean F1-macro {o['mean_f1_macro']:.3f}"
    )