"""Model evaluation: per-label metrics, multi-output metrics, and error analysis.

Replaces the prototype's single printed classification_report with structured,
saved outputs suitable for comparing models (Task 4):

    evaluate(...)                      -> metrics dict (per-label + overall)
    build_classification_report_df(...)-> tidy per-class precision/recall/F1 table
    collect_misclassifications(...)    -> examples of errors for the error analysis
    save_metrics(...) / save_report(...) / save_predictions(...)
    evaluate_and_save(...)             -> run all of the above in one call
"""

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    model_name: str = "unknown",
    model_version: str = Config.MODEL_VERSION,
) -> dict:
    """Compute per-label and overall multi-output metrics.

    y_true / y_pred are 2D arrays of shape (n_samples, n_labels).
    """
    labels = labels or Config.CLASS_COLS
    per_label: dict[str, dict] = {}

    for i, label in enumerate(labels):
        yt, yp = y_true[:, i], y_pred[:, i]
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0
        )
        _, _, f_wt, _ = precision_recall_fscore_support(
            yt, yp, average="weighted", zero_division=0
        )
        per_label[label] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision_macro": float(p_mac),
            "recall_macro": float(r_mac),
            "f1_macro": float(f_mac),
            "f1_weighted": float(f_wt),
            "n_classes": int(len(np.unique(yt))),
            "support": int(len(yt)),
        }

    overall = {
        # Fraction of individual (row, label) cells predicted wrong.
        "hamming_loss": float((y_true != y_pred).mean()),
        # Fraction of rows where ALL labels are correct (strict).
        "exact_match_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "mean_f1_macro": float(np.mean([per_label[l]["f1_macro"] for l in labels])),
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


def build_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Tidy per-class precision/recall/F1/support table across all labels."""
    labels = labels or Config.CLASS_COLS
    rows = []
    for i, label in enumerate(labels):
        report = classification_report(
            y_true[:, i], y_pred[:, i], output_dict=True, zero_division=0
        )
        for cls, scores in report.items():
            if isinstance(scores, dict):  # skip the scalar 'accuracy' entry
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


# --------------------------------------------------------------------------- #
# Error analysis (Task 4)
# --------------------------------------------------------------------------- #
def collect_misclassifications(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    text_col: str = Config.INTERACTION_CONTENT,
    max_examples: int | None = 50,
    snippet_len: int = 160,
) -> pd.DataFrame:
    """Long-form table of individual label errors, for the error-analysis table.

    One row per (sample, label) where the prediction was wrong.
    """
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


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
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


def evaluate_and_save(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    labels: list[str] | None = None,
    metrics_path: Path = Config.METRICS_PATH,
    report_path: Path = Config.CLASSIFICATION_REPORT_PATH,
) -> dict:
    """Compute metrics + report, save both, and return the metrics dict."""
    labels = labels or Config.CLASS_COLS
    metrics = evaluate(y_true, y_pred, labels, model_name=model_name)
    report_df = build_classification_report_df(y_true, y_pred, labels)
    save_metrics(metrics, metrics_path)
    save_report(report_df, report_path)
    return metrics


def print_summary(metrics: dict) -> None:
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