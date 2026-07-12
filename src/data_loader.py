"""Data loading, validation, and multi-label target construction.

Public entry points:
    load_training_data(input_files=None) -> pd.DataFrame
        Loads + concatenates the raw CSVs, renames Type 1-4 -> y1-y4, validates
        required columns, cleans label values, and builds the sentinel-filled
        multi-label targets (y2/y3/y4).

Run standalone to smoke-test:
    python -m src.data_loader
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_single_csv(path: Path) -> pd.DataFrame:
    """Read one CSV, failing early with a clear message if it's missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected input CSV not found: {path}")
    # skipinitialspace mirrors the prototype; the source files have padded fields.
    return pd.read_csv(path, skipinitialspace=True)


def load_raw(input_files: list[Path] | None = None) -> pd.DataFrame:
    """Load and concatenate the raw CSVs, then rename the label columns."""
    input_files = input_files or Config.INPUT_FILES

    frames = []
    for path in input_files:
        df = _read_single_csv(path)
        df = df.rename(columns=Config.RENAME_MAP)
        logger.info("Loaded %s rows from %s", len(df), path.name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %s rows, %s columns", *combined.shape)
    return combined


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame, require_labels: bool = True) -> None:
    """Fail early on structural problems instead of producing bad predictions.

    require_labels=False is used at inference time, where new messages arrive
    without the y2/y3/y4 columns.
    """
    required = list(Config.TEXT_COLS)
    if require_labels:
        required += Config.CLASS_COLS

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # At least one text column must contain usable (non-empty) text.
    text_filled = (
        df[Config.TEXT_COLS]
        .apply(lambda s: s.astype(str).str.strip().replace({"nan": ""}))
        .replace("", pd.NA)
    )
    if text_filled.dropna(how="all").empty:
        raise ValueError("No usable text found in any text column.")


# --------------------------------------------------------------------------- #
# Cleaning / target construction
# --------------------------------------------------------------------------- #
def _coerce_text(df: pd.DataFrame) -> pd.DataFrame:
    """Force text columns to clean strings (handles NaN -> '' up front)."""
    for col in Config.TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)
    return df


def _clean_label(series: pd.Series) -> pd.Series:
    """Strip whitespace and normalise missing values to <NA>.

    The raw data has trailing spaces (e.g. 'In-App Purchase '), which would
    otherwise create spurious duplicate classes.
    """
    cleaned = series.astype(str).str.strip()
    # Treat empties and stringified NaNs as genuinely missing.
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return cleaned


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Clean label columns; require the core label y2; sentinel-fill y3/y4.

    Policy:
      - y2 is the core target and must be present -> rows missing it are dropped
        (a no-op on the provided data, but a safeguard against malformed input).
      - y3/y4 are secondary and default to SENTINEL_LABEL when absent, so every
        retained row is predictable across all three targets.
    """
    for col in Config.CLASS_COLS:
        df[col] = _clean_label(df[col])

    # Drop rows missing the core label, with an audit log of how many.
    n_before = len(df)
    df = df.loc[df["y2"].notna()].copy()
    dropped = n_before - len(df)
    if dropped:
        logger.warning("Dropped %s row(s) missing core label y2", dropped)

    # Sentinel-fill secondary labels, logging the fill counts for auditability.
    for col in ("y3", "y4"):
        n_missing = int(df[col].isna().sum())
        if n_missing:
            logger.info(
                "Filled %s missing '%s' value(s) with sentinel '%s'",
                n_missing, col, Config.SENTINEL_LABEL,
            )
        df[col] = df[col].fillna(Config.SENTINEL_LABEL)

    return df


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_training_data(input_files: list[Path] | None = None) -> pd.DataFrame:
    """Full training-data path: load -> validate -> coerce text -> build targets."""
    df = load_raw(input_files)
    validate(df, require_labels=True)
    df = _coerce_text(df)
    df = build_targets(df)
    logger.info("Final training dataset: %s rows", len(df))
    return df

def load_inference_data(input_file: str | Path) -> pd.DataFrame:
    """Load a single CSV of new messages for inference.

    Unlike load_training_data, this does NOT require label columns, does NOT
    build targets, and does NOT drop any rows — every input message must receive
    a prediction. Validation runs in label-free mode (text columns only).
    """
    input_file = Path(input_file)
    df = _read_single_csv(input_file)
    df = df.rename(columns=Config.RENAME_MAP)  # harmless if Type cols are absent
    validate(df, require_labels=False)
    df = _coerce_text(df)
    logger.info("Loaded %s message(s) for inference from %s", len(df), input_file.name)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = load_training_data()
    print("\nShape:", data.shape)
    print("\nLabel distributions:")
    for c in Config.CLASS_COLS:
        print(f"\n[{c}]")
        print(data[c].value_counts(dropna=False))
    print("\nSample rows:")
    print(data[[*Config.TEXT_COLS, *Config.CLASS_COLS]].head())