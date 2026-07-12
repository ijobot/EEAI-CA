"""Logistic Regression model wrapped for multi-output (multi-label) classification.

Provides the second model for the Task 4 comparison. Mirrors the random forest
model's structure, so it drops into the training pipeline with no other changes.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from src.config import Config
from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        **kwargs,
    ) -> None:
        # lbfgs handles multiclass natively and converges cleanly on this TF-IDF
        # data; max_iter is raised from the default 100 to avoid non-convergence
        # on the sparse, high-dimensional features.
        self._C = C
        self._class_weight = class_weight
        self._solver = solver
        self._max_iter = max_iter
        self._kwargs = kwargs
        super().__init__()

    def _build_model(self):
        base = LogisticRegression(
            C=self._C,
            class_weight=self._class_weight,
            solver=self._solver,
            max_iter=self._max_iter,
            random_state=Config.SEED,
            **self._kwargs,
        )
        # One independent LR per label column (y2, y3, y4).
        return MultiOutputClassifier(base)