# logistic regression model (Task 4 comparison)

# built the same as random forest model so it can be dropped in without other changes needed

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
        # max_iter is much higher than the default 100 because we're dealing with a huge amount of features (1199),
        # but very sparse data in them (lots of zeroes in our matrix).  1000 iterations gives the model a longer runway to reach convergence.
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
        # one LR per label column (y2, y3, y4)
        return MultiOutputClassifier(base)