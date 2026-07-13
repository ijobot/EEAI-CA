# abstract base model class

# All models passed into this will share fit, predict, confidence, save, etc.
# This ensures train and predict can treat any model interchangeably, and replaces the prototype's base.py file.  This new version
# supports multi-output prediction, persistance, and confidnce scoring all in the same place.

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.model = self._build_model()
        self.labels_: list[str] | None = None
        self.model_version: str = Config.MODEL_VERSION

    @abstractmethod
    def _build_model(self):
        """Return the constructed (unfitted) multi-output estimator."""

    def fit(self, X, Y) -> "BaseModel":
        self.model.fit(X, Y)
        self.labels_ = list(Config.CLASS_COLS)
        logger.info("Trained '%s' on %s samples, %s labels", self.name, X.shape[0], len(self.labels_))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_probability(self, X) -> list:
        return self.model.predict_proba(X)

    def confidence(self, X) -> np.ndarray:
        return np.column_stack([p.max(axis=1) for p in self.predict_probability(X)])

    # save
    def save(self, path: Path = Config.MODEL_PATH) -> Path:
        Config.ensure_dirs()
        joblib.dump(self, path)
        logger.info("Saved model '%s' -> %s", self.name, path)
        return path

    # load model and flag if not found
    @classmethod
    def load(cls, path: Path = Config.MODEL_PATH) -> "BaseModel":
        if not Path(path).exists():
            raise FileNotFoundError(f"No model at {path}. Run the training pipeline first.")
        model = joblib.load(path)
        logger.info("Loaded model '%s' <- %s", model.name, path)
        return model