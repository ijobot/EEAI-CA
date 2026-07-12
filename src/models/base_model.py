"""Abstract base class defining the common multi-label model interface.

All concrete models (random forest, logistic regression) share fit / predict /
predict_proba / confidence / save / load here, so train.py and predict.py can
treat any model interchangeably. Subclasses only implement _build_model().

This is the evolution of the prototype's base.py, redesigned to:
  - decouple from the old Data container (fit takes X, Y directly),
  - support multi-output prediction,
  - standardise persistence and confidence scoring in one place.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Common interface + shared behaviour for multi-label classifiers."""

    name: str = "base"

    def __init__(self) -> None:
        # Subclasses set any hyperparameters before calling super().__init__().
        self.model = self._build_model()
        self.labels_: list[str] | None = None
        self.model_version: str = Config.MODEL_VERSION

    # ------------------------------------------------------------------ #
    # Subclass responsibility
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build_model(self):
        """Return the constructed (unfitted) multi-output estimator."""

    # ------------------------------------------------------------------ #
    # Shared fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X, Y) -> "BaseModel":
        """Fit on features X and a 2D label array Y of shape (n_samples, n_labels)."""
        self.model.fit(X, Y)
        self.labels_ = list(Config.CLASS_COLS)
        logger.info("Trained '%s' on %s samples, %s labels", self.name, X.shape[0], len(self.labels_))
        return self

    def predict(self, X) -> np.ndarray:
        """Return predictions of shape (n_samples, n_labels)."""
        return self.model.predict(X)

    def predict_proba(self, X) -> list:
        """Return a list of per-label probability arrays (one per output)."""
        return self.model.predict_proba(X)

    def confidence(self, X) -> np.ndarray:
        """Max class probability per label -> shape (n_samples, n_labels).

        Used as the confidence score in batch inference output (Task 3).
        """
        return np.column_stack([p.max(axis=1) for p in self.predict_proba(X)])

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path = Config.MODEL_PATH) -> Path:
        Config.ensure_dirs()
        joblib.dump(self, path)
        logger.info("Saved model '%s' -> %s", self.name, path)
        return path

    @classmethod
    def load(cls, path: Path = Config.MODEL_PATH) -> "BaseModel":
        if not Path(path).exists():
            raise FileNotFoundError(f"No model at {path}. Run the training pipeline first.")
        model = joblib.load(path)
        logger.info("Loaded model '%s' <- %s", model.name, path)
        return model