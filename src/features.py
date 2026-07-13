# featrure extraction and leakage prevention
# Vectorizer is fit ONLY to the training data.  We use the transform function for the test set and inference steps so there is no leakage.  This is persisted so
# the same features and weights are used during predictions as well.

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import Config

logger = logging.getLogger(__name__)

# This helps to ensure that all text is built/cleaned/processed in the same way for training and inference.
class FeatureExtractor:

    def __init__(
        self,
        max_features: int = Config.TFIDF_MAX_FEATURES,
        min_df: int = Config.TFIDF_MIN_DF,
        max_df: float = Config.TFIDF_MAX_DF,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
        )
        self._fitted = False

    # text assembly (shared by train and inference)
    @staticmethod
    def build_text(df: pd.DataFrame) -> pd.Series:
        # combines the 2 text columns similar to the `summary + " " + content` from the prototype
        summary = df[Config.TICKET_SUMMARY].fillna("").astype(str)
        content = df[Config.INTERACTION_CONTENT].fillna("").astype(str)
        return (summary + " " + content).str.strip()

    # fit / transform
    def fit(self, train_df: pd.DataFrame) -> "FeatureExtractor":
        # fitting on training rows only
        self.vectorizer.fit(self.build_text(train_df))
        self._fitted = True
        logger.info("Fitted TF-IDF vocabulary: %s features", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        # returns a sparse matrix to reduce memory usage and wasted compute
        if not self._fitted:
            raise RuntimeError("FeatureExtractor.transform() called before fit()/load().")
        return self.vectorizer.transform(self.build_text(df))

    def fit_transform(self, train_df: pd.DataFrame) -> csr_matrix:
        # only called on training rows and never full data
        X = self.vectorizer.fit_transform(self.build_text(train_df))
        self._fitted = True
        logger.info("Fitted TF-IDF vocabulary: %s features", len(self.vectorizer.vocabulary_))
        return X

    @property
    def n_features(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0

    # saving output
    def save(self, path: Path = Config.VECTORIZER_PATH) -> Path:
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted FeatureExtractor.")
        Config.ensure_dirs()
        joblib.dump(self, path)
        logger.info("Saved vectoriser -> %s", path)
        return path

    @classmethod
    def load(cls, path: Path = Config.VECTORIZER_PATH) -> "FeatureExtractor":
        if not Path(path).exists():
            # notify user to train first before trying to access the vectoriser
            raise FileNotFoundError(
                f"No vectoriser at {path}. Run the training pipeline first."
            )
        fx = joblib.load(path)
        logger.info("Loaded vectoriser <- %s (%s features)", path, fx.n_features)
        return fx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from sklearn.model_selection import train_test_split

    from src.data_loader import load_training_data
    from src.preprocessing import preprocess

    df = preprocess(load_training_data())
    train_df, test_df = train_test_split(df, test_size=Config.TEST_SIZE, random_state=Config.SEED)

    # again, fit on train ONLY, never full data
    fx = FeatureExtractor().fit(train_df)
    X_train, X_test = fx.transform(train_df), fx.transform(test_df)
    print(f"\nX_train {X_train.shape} | X_test {X_test.shape} | sparse={hasattr(X_train, 'nnz')}")
    print(f"Shared feature space: {X_train.shape[1] == X_test.shape[1]}")
    fx.save()
    print("Reloaded features:", FeatureExtractor.load().n_features)