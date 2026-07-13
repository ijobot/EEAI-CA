# Base class for every model in the system.  Defines one shared interface (fit, predict, confidence, save, load) so each
# model works the same way and can be swapped in the pipeline without changes.

import logging
from pathlib import Path

import pandas as pd

from src.config import Config

# logger for output, set to __name__ so the readouts can be used as breadcrumbs for debugging.
# This is the same setup for all files in the project, so I'll only comment it here.
logger = logging.getLogger(__name__)

# read a CSV and provide failure message if missing
def _read_single_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    # handle padded fields with skipinitialspace
    return pd.read_csv(path, skipinitialspace=True)

# load the CSVs and concatenate them
def load_raw(input_files: list[Path] | None = None) -> pd.DataFrame:
    input_files = input_files or Config.INPUT_FILES

    frames = []
    for path in input_files:
        df = _read_single_csv(path)
        # update the column names from the CSV versions to the project label versions
        df = df.rename(columns=Config.RENAME_MAP)
        logger.info("Loaded %s rows from %s", len(df), path.name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %s rows, %s columns", *combined.shape)
    return combined


# validation
def validate(df: pd.DataFrame, require_labels: bool = True) -> None:
    # Fails early on structural problems rather than producing bad predictions.
    # Uses require_labels=False during inference because many messages don't have y2,y3,y4 columns.
    required = list(Config.TEXT_COLS)
    if require_labels:
        required += Config.CLASS_COLS

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # At least one text column must contain usable text.
    text_filled = (
        df[Config.TEXT_COLS]
        .apply(lambda s: s.astype(str).str.strip().replace({"nan": ""}))
        .replace("", pd.NA)
    )
    if text_filled.dropna(how="all").empty:
        raise ValueError("No usable text found in any text column.")


# cleaning text for processing (changes NaN to empty string instead)
def _coerce_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in Config.TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)
    return df

# strips whitespace
def _clean_label(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    # treat empties and stringified NaNs as missing
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return cleaned

# requires y2 as the core column, and adds the SENTINEL_LABEL to any empties in y3/y4
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    for col in Config.CLASS_COLS:
        df[col] = _clean_label(df[col])

    # drop rows missing the core label and log them
    n_before = len(df)
    df = df.loc[df["y2"].notna()].copy()
    dropped = n_before - len(df)
    if dropped:
        logger.warning("Dropped %s row(s) missing core label y2", dropped)

    # logging all SENTINEL_LABEL injections
    for col in ("y3", "y4"):
        n_missing = int(df[col].isna().sum())
        if n_missing:
            logger.info(
                "Filled %s missing '%s' value(s) with sentinel '%s'",
                n_missing, col, Config.SENTINEL_LABEL,
            )
        df[col] = df[col].fillna(Config.SENTINEL_LABEL)
    return df


# loading data from sources, then running the various processing steps
def load_training_data(input_files: list[Path] | None = None) -> pd.DataFrame:
    df = load_raw(input_files)
    validate(df, require_labels=True)
    df = _coerce_text(df)
    df = build_targets(df)
    logger.info("Final training dataset: %s rows", len(df))
    return df

# loading data from the new_messages CSV
# This function is different from above because it doesn't require label columns or build targets and it doesn't drop rows.
# It simply reads the messages and gets them ready for predictions no matter what.
def load_inference_data(input_file: str | Path) -> pd.DataFrame:
    input_file = Path(input_file)
    df = _read_single_csv(input_file)
    df = df.rename(columns=Config.RENAME_MAP)  # harmless if Type cols are absent
    validate(df, require_labels=False)
    df = _coerce_text(df)
    logger.info("Loaded %s message(s) for inference from %s", len(df), input_file.name)
    return df


if __name__ == "__main__":
    # Got a bit of guidance from Claude on this one, as I've never set up a logger before and wasn't sure how to write everything out.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    data = load_training_data()
    print("\nShape:", data.shape)
    print("\nLabel distributions:")
    for c in Config.CLASS_COLS:
        print(f"\n[{c}]")
        print(data[c].value_counts(dropna=False))
    print("\nSample rows:")
    print(data[[*Config.TEXT_COLS, *Config.CLASS_COLS]].head())