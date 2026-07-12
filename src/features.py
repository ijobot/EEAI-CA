"""TF-IDF feature extraction with leakage-free fitting and persistence.

The core fix over the prototype: the vectoriser is fit ONLY on training data and
then reused via .transform() for the test set and for inference. It is persisted
to disk so the exact same feature space is reconstructed at prediction time.

Typical training use (fit on train rows only, transform the rest):
    fx = FeatureExtractor().fit(train_df)
    X_train = fx.transform(train_df)
    X_test  = fx.transform(test_df)
    fx.save()                         # -> artifacts/vectorizer.joblib

Typical inference use:
    fx = FeatureExtractor.load()      # <- same fitted vocabulary
    X_new = fx.transform(new_df)
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import Config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Wraps a TfidfVectorizer plus the text-assembly logic as one artefact.

    Persisting the whole object (not just the raw vectoriser) guarantees that the
    way text is assembled from the two columns is identical at train and serve
    time — there is no separate, drift-prone text-building step to keep in sync.
    """

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

    # ------------------------------------------------------------------ #
    # Text assembly (shared by train + inference)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_text(df: pd.DataFrame) -> pd.Series:
        """Combine the two text columns into a single string per row.

        Mirrors the prototype's `summary + " " + content`, but null-safe.
        """
        summary = df[Config.TICKET_SUMMARY].fillna("").astype(str)
        content = df[Config.INTERACTION_CONTENT].fillna("").astype(str)
        return (summary + " " + content).str.strip()

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame) -> "FeatureExtractor":
        """Fit the vocabulary on TRAINING rows only (prevents leakage)."""
        self.vectorizer.fit(self.build_text(train_df))
        self._fitted = True
        logger.info("Fitted TF-IDF vocabulary: %s features", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """Map rows into the fitted feature space. Returns a SPARSE matrix."""
        if not self._fitted:
            raise RuntimeError("FeatureExtractor.transform() called before fit()/load().")
        return self.vectorizer.transform(self.build_text(df))

    def fit_transform(self, train_df: pd.DataFrame) -> csr_matrix:
        """Convenience for the training set only. Never call on full data."""
        X = self.vectorizer.fit_transform(self.build_text(train_df))
        self._fitted = True
        logger.info("Fitted TF-IDF vocabulary: %s features", len(self.vectorizer.vocabulary_))
        return X

    @property
    def n_features(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
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

    fx = FeatureExtractor().fit(train_df)  # fit on train ONLY
    X_train, X_test = fx.transform(train_df), fx.transform(test_df)
    print(f"\nX_train {X_train.shape} | X_test {X_test.shape} | sparse={hasattr(X_train, 'nnz')}")
    print(f"Shared feature space: {X_train.shape[1] == X_test.shape[1]}")
    fx.save()
    print("Reloaded features:", FeatureExtractor.load().n_features)