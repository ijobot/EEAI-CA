"""Training pipeline entry point.

Orchestrates the full flow in leakage-safe order:
    load -> preprocess -> SPLIT -> fit features on TRAIN only -> transform
         -> train each model -> evaluate -> save artefacts + outputs

Run:
    python -m src.train
    python -m src.train --input data/AppGallery.csv data/Purchasing.csv

Designed around a LIST of models: adding a new model requires only appending an
instance to DEFAULT_MODELS — no changes to this file's logic.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data_loader import load_training_data
from src.evaluation import (
    build_classification_report_df,
    collect_misclassifications,
    evaluate,
    print_summary,
    save_metrics,
    save_report,
)
from src.features import FeatureExtractor
from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestModel
from src.models.logistic_regression_model import LogisticRegressionModel
from src.preprocessing import preprocess

logger = logging.getLogger(__name__)

# Add models here to include them in the comparison. Logistic Regression joins
# this list next — no other change to train.py is needed.
DEFAULT_MODELS: list[BaseModel] = [
    RandomForestModel(),
    LogisticRegressionModel(),
]


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the core label y2 when safe.

    y2 is fully populated and its rarest class is large, so stratifying on it
    keeps the primary target balanced across splits. We fall back to a random
    split if any y2 class is too small to stratify. This replaces the prototype's
    fragile `new_test_size = X.shape[0] * 0.2 / X_good.shape[0]` formula, which
    could exceed 1 and crash.
    """
    stratify = df["y2"] if df["y2"].value_counts().min() >= 2 else None
    if stratify is None:
        logger.warning("y2 has a singleton class; using a non-stratified split.")

    train_df, test_df = train_test_split(
        df,
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=stratify,
    )
    logger.info(
        "Split: %s train / %s test (stratified on y2: %s)",
        len(train_df), len(test_df), stratify is not None,
    )
    _warn_unseen_classes(train_df, test_df)
    return train_df, test_df


def _warn_unseen_classes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Flag test classes absent from train — these are unavoidable errors.

    Useful context for error analysis (Task 4): the model literally never saw
    these classes, so it cannot predict them regardless of quality.
    """
    for col in Config.CLASS_COLS:
        unseen = set(test_df[col]) - set(train_df[col])
        if unseen:
            logger.warning(
                "%s: %s test class(es) never seen in training: %s",
                col, len(unseen), sorted(unseen)[:5],
            )


# --------------------------------------------------------------------------- #
# Per-model output paths
# --------------------------------------------------------------------------- #
def _metrics_path(name: str) -> Path:
    return Config.OUTPUTS_DIR / f"metrics_{name}.json"


def _report_path(name: str) -> Path:
    return Config.OUTPUTS_DIR / f"classification_report_{name}.csv"


def _errors_path(name: str) -> Path:
    return Config.OUTPUTS_DIR / f"errors_{name}.csv"


def _model_path(name: str) -> Path:
    return Config.ARTIFACTS_DIR / f"model_{name}.joblib"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    models: list[BaseModel] | None = None,
    input_files: list[Path] | None = None,
) -> dict:
    """Train and evaluate every model; save artefacts and outputs.

    The best model by mean macro-F1 is also saved as the default artefacts
    (model.joblib) that predict.py loads.
    """
    Config.ensure_dirs()
    models = models if models is not None else DEFAULT_MODELS

    # 1. Data
    df = preprocess(load_training_data(input_files))

    # 2. Split BEFORE fitting features (this is the leakage fix)
    train_df, test_df = split_data(df)

    # 3. Fit features on TRAIN only, transform both; vectoriser is shared by
    #    all models, so we fit it once.
    fx = FeatureExtractor().fit(train_df)
    X_train, X_test = fx.transform(train_df), fx.transform(test_df)
    Y_train = train_df[Config.CLASS_COLS].to_numpy()
    Y_test = test_df[Config.CLASS_COLS].to_numpy()
    fx.save()  # artifacts/vectorizer.joblib

    # 4. Train + evaluate each model
    results: dict[str, dict] = {}
    best_name: str | None = None
    best_model: BaseModel | None = None
    best_pred: np.ndarray | None = None

    for model in models:
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate(Y_test, y_pred, model_name=model.name)
        save_metrics(metrics, _metrics_path(model.name))
        save_report(build_classification_report_df(Y_test, y_pred), _report_path(model.name))
        collect_misclassifications(test_df, Y_test, y_pred).to_csv(
            _errors_path(model.name), index=False
        )
        model.save(_model_path(model.name))

        results[model.name] = metrics
        print_summary(metrics)

        score = metrics["overall"]["mean_f1_macro"]
        if best_name is None or score > results[best_name]["overall"]["mean_f1_macro"]:
            best_name, best_model, best_pred = model.name, model, y_pred

    # 5. Promote the best model to the default artefacts + write comparison
    best_model.save(Config.MODEL_PATH)
    save_report(build_classification_report_df(Y_test, best_pred), Config.CLASSIFICATION_REPORT_PATH)
    save_metrics(
        {
            "best_model": best_name,
            "selection_metric": "mean_f1_macro",
            "models": {n: m["overall"] for n, m in results.items()},
        },
        Config.METRICS_PATH,
    )

    logger.info("Best model: '%s' (saved as %s)", best_name, Config.MODEL_PATH.name)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-label classifier.")
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=None,
        help="One or more training CSVs (defaults to the files in Config).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    run(input_files=args.input)