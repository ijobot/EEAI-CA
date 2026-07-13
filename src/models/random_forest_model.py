# random forest model (Task 4 comparison)

# built the same as logistic regression model so it can be dropped in without other changes needed

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
        # hyperparameters were hardcoded in the prototype, but can now be changed here in the constructor as needed
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
        # one RF per label column (y2, y3, y4)
        return MultiOutputClassifier(base)