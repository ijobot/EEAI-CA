"""Random Forest model wrapped for multi-output (multi-label) classification."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.config import Config
from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 300,
        class_weight: str = "balanced_subsample",
        **kwargs,
    ) -> None:
        # Hyperparameters are now constructor arguments (configurable), not
        # literals buried in the class body as in the prototype.
        self._n_estimators = n_estimators
        self._class_weight = class_weight
        self._kwargs = kwargs
        super().__init__()

    def _build_model(self):
        base = RandomForestClassifier(
            n_estimators=self._n_estimators,
            class_weight=self._class_weight,
            random_state=Config.SEED,
            n_jobs=-1,
            **self._kwargs,
        )
        # One independent RF per label column (y2, y3, y4).
        return MultiOutputClassifier(base)