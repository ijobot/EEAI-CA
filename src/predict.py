"""Batch inference pipeline (Task 3).

Loads the persisted model + vectoriser, runs the IDENTICAL preprocessing and
feature extraction used in training, and writes a predictions CSV.

Run:
    python -m src.predict --input data/new_messages.csv --output outputs/predictions.csv

Training-serving consistency is guaranteed structurally: this module calls the
same preprocess() and the same fitted FeatureExtractor as train.py, rather than
re-implementing any cleaning or vectorisation. There is no parallel inference-only
data path that could drift from training.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data_loader import load_inference_data
from src.features import FeatureExtractor
from src.models.base_model import BaseModel
from src.preprocessing import preprocess

logger = logging.getLogger(__name__)

# Preferred identifier columns, in priority order; falls back to the row index.
_ID_CANDIDATES = ("message_id", "Ticket id", "id", "row_id")


def _resolve_ids(df: pd.DataFrame) -> tuple[list[str], str]:
    """Return a per-row identifier and the column it came from."""
    for col in _ID_CANDIDATES:
        if col in df.columns:
            return df[col].astype(str).tolist(), col
    return df.index.astype(str).tolist(), "row_index"


def predict_file(
    input_path: str | Path,
    output_path: Path = Config.PREDICTIONS_PATH,
    model_path: Path = Config.MODEL_PATH,
    vectorizer_path: Path = Config.VECTORIZER_PATH,
) -> pd.DataFrame:
    """Run batch inference on a CSV of new messages and write predictions."""
    # 1. Load artefacts — same feature space and model produced by training.
    fx = FeatureExtractor.load(vectorizer_path)
    model = BaseModel.load(model_path)
    labels = model.labels_ or list(Config.CLASS_COLS)

    # 2. Load raw messages (labels not required). Keep df_raw for reference text.
    df_raw = load_inference_data(input_path)

    # 3. IDENTICAL preprocessing + features as training.
    df_clean = preprocess(df_raw)
    X = fx.transform(df_clean)

    # 4. Predict + per-label confidence.
    preds = model.predict(X)
    conf = model.confidence(X)

    # 5. Assemble output.
    ids, id_source = _resolve_ids(df_raw)
    timestamp = datetime.now(timezone.utc).isoformat()

    out = pd.DataFrame({"id": ids, "id_source": id_source})
    for i, label in enumerate(labels):
        out[f"pred_{label}"] = preds[:, i]
        out[f"confidence_{label}"] = conf[:, i].round(4)
    out["text_reference"] = (
        df_raw[Config.TICKET_SUMMARY].fillna("").astype(str).str.slice(0, 120)
    )
    out["model_name"] = model.name
    out["model_version"] = model.model_version
    out["predicted_at"] = timestamp

    # 6. Save.
    Config.ensure_dirs()
    out.to_csv(output_path, index=False)
    logger.info("Wrote %s prediction(s) -> %s", len(out), output_path)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference on new customer messages.")
    parser.add_argument("--input", type=Path, required=True, help="CSV of new messages.")
    parser.add_argument("--output", type=Path, default=Config.PREDICTIONS_PATH, help="Output CSV path.")
    parser.add_argument("--model", type=Path, default=Config.MODEL_PATH, help="Model artefact.")
    parser.add_argument("--vectorizer", type=Path, default=Config.VECTORIZER_PATH, help="Vectoriser artefact.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    result = predict_file(args.input, args.output, args.model, args.vectorizer)
    print(result.to_string(index=False))