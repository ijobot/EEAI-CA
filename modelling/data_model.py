import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data:
    """Simple container for train/test data used by the starter prototype."""

    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        y = df.y.to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value) < 1:
            print("None of the classes have more than 3 records: skipping ...")
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good,
            y_good,
            test_size=new_test_size,
            random_state=seed,
            stratify=y_good,
        )
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_embeddings(self):
        return self.embeddings
