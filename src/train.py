# training pipeline

# Orchestrates the full pipeline flow: load, preprocess, split, fit features on train, transform, train each model, evaluate/save/output.
# Allows new models to be added in easily without having to change this file

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


# split data for train/test
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Stratification ensures we don't end up with lopsided instances of classes in train or test rows.
    # This replaces the prototype's `new_test_size = X.shape[0] * 0.2 / X_good.shape[0]` formula, which could exceed 1 and crash.
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

# Helper function to ensure that classes absent from the training run are flagged.
# The model can't predict what is has never seen during training, and with a small dataset, this is possible.
def _warn_unseen_classes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for col in Config.CLASS_COLS:
        unseen = set(test_df[col]) - set(train_df[col])
        if unseen:
            logger.warning(
                "%s: %s test class(es) never seen in training: %s",
                col, len(unseen), sorted(unseen)[:5],
            )


# output paths
def _metrics_path(name: str) -> Path:
    return Config.OUTPUTS_DIR / f"metrics_{name}.json"


def _report_path(name: str) -> Path:
    return Config.OUTPUTS_DIR / f"classification_report_{name}.csv"


def _errors_path(name: str) -> Path:
    return Config.OUTPUTS_DIR / f"errors_{name}.csv"


def _model_path(name: str) -> Path:
    return Config.ARTIFACTS_DIR / f"model_{name}.joblib"


# run the training models
def run(
    models: list[BaseModel] | None = None,
    input_files: list[Path] | None = None,
) -> dict:
    Config.ensure_dirs()
    models = models if models is not None else DEFAULT_MODELS

    # data
    df = preprocess(load_training_data(input_files))

    # split BEFORE fitting features (leakage fix)
    train_df, test_df = split_data(df)

    # fit features on TRAIN only and transform both; vectoriser is shared by all models
    fx = FeatureExtractor().fit(train_df)
    X_train, X_test = fx.transform(train_df), fx.transform(test_df)
    Y_train = train_df[Config.CLASS_COLS].to_numpy()
    Y_test = test_df[Config.CLASS_COLS].to_numpy()
    fx.save()  # artifacts/vectorizer.joblib

    # train and evaluate
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

    # best model saved to the default artefacts, log comparison
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

# help with flag inputs in command line, same as other instances
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