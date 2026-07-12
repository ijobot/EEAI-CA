"""Central configuration for the customer interaction classification system.

Every path, column name, label definition, and key hyperparameter lives here so
the rest of the pipeline never hardcodes them. Paths resolve relative to the
project root via pathlib, which keeps the project portable across machines.
"""

from pathlib import Path

class Config:
    # -------------------------------------------------------------- paths
    # src/config.py -> src/ -> project root
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = PROJECT_ROOT / "data"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

    # Default training inputs (overridable via CLI / function arguments).
    INPUT_FILES: list[Path] = [
        DATA_DIR / "AppGallery.csv",
        DATA_DIR / "Purchasing.csv",
    ]

    # Artefact locations (written by the training pipeline).
    MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
    VECTORIZER_PATH: Path = ARTIFACTS_DIR / "vectorizer.joblib"

    # Evaluation / prediction outputs.
    METRICS_PATH: Path = OUTPUTS_DIR / "metrics.json"
    CLASSIFICATION_REPORT_PATH: Path = OUTPUTS_DIR / "classification_report.csv"
    PREDICTIONS_PATH: Path = OUTPUTS_DIR / "predictions.csv"

    # ------------------------------------------------------------ columns
    TICKET_SUMMARY: str = "Ticket Summary"
    INTERACTION_CONTENT: str = "Interaction content"
    TEXT_COLS: list[str] = [TICKET_SUMMARY, INTERACTION_CONTENT]

    # Raw label columns in the CSVs -> internal names.
    RENAME_MAP: dict[str, str] = {
        "Type 1": "y1",
        "Type 2": "y2",
        "Type 3": "y3",
        "Type 4": "y4",
    }

    # Multi-label targets. y2 is always populated; y3/y4 may be missing and get
    # filled with SENTINEL_LABEL so every row is predictable across all targets.
    CLASS_COLS: list[str] = ["y2", "y3", "y4"]
    SENTINEL_LABEL: str = "unknown"

    # ----------------------------------------------------------- features
    TFIDF_MAX_FEATURES: int = 2000
    TFIDF_MIN_DF: int = 4
    TFIDF_MAX_DF: float = 0.90

    # -------------------------------------------------------- train / eval
    SEED: int = 0
    TEST_SIZE: float = 0.2

    # ------------------------------------------------------ model metadata
    MODEL_VERSION: str = "0.1.0"

    # ---------------------------------------------------------- utilities
    @classmethod
    def ensure_dirs(cls) -> None:
        """Create artefact/output directories if they don't already exist."""
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)