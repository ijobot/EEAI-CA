# Config for the entire system.
# All paths, column names, labels, and hyperparameters live in this file.
# Pathlib introduced for all file pathing so the project can be run on any machine.

from pathlib import Path

class Config:
    # root path
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    # sub-file paths
    DATA_DIR: Path = PROJECT_ROOT / "data"
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

    # data paths
    INPUT_FILES: list[Path] = [
        DATA_DIR / "AppGallery.csv",
        DATA_DIR / "Purchasing.csv",
    ]

    # artefact paths
    MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
    VECTORIZER_PATH: Path = ARTIFACTS_DIR / "vectorizer.joblib"

    # output paths
    METRICS_PATH: Path = OUTPUTS_DIR / "metrics.json"
    CLASSIFICATION_REPORT_PATH: Path = OUTPUTS_DIR / "classification_report.csv"
    PREDICTIONS_PATH: Path = OUTPUTS_DIR / "predictions.csv"

    # columns
    TICKET_SUMMARY: str = "Ticket Summary"
    INTERACTION_CONTENT: str = "Interaction content"
    TEXT_COLS: list[str] = [TICKET_SUMMARY, INTERACTION_CONTENT]

    # mapping CSV column names to pipeline label names
    RENAME_MAP: dict[str, str] = {
        "Type 1": "y1",
        "Type 2": "y2",
        "Type 3": "y3",
        "Type 4": "y4",
    }

    # multi-label for class columns and SENTINEL_LABEL adds "unknown" so every row is predictable
    CLASS_COLS: list[str] = ["y2", "y3", "y4"]
    SENTINEL_LABEL: str = "unknown"

    # hyperparameters for TF-IDF
    TFIDF_MAX_FEATURES: int = 2000
    TFIDF_MIN_DF: int = 4
    TFIDF_MAX_DF: float = 0.90

    # training and evaluation
    SEED: int = 0
    TEST_SIZE: float = 0.2

    # model
    MODEL_VERSION: str = "0.1.0"

    # utilities
    @classmethod
    def ensure_dirs(cls) -> None:
        # Create artefact/output dirs if they don't already exist
        cls.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)