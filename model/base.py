from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Minimal abstract model interface for the starter prototype.

    Students may extend or redesign this interface as part of the assessment.
    """

    @abstractmethod
    def train(self, data) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Generate predictions."""
        pass

    @abstractmethod
    def data_transform(self) -> None:
        """Optional model-specific data transformation hook."""
        pass
