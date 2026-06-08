"""Logistic-regression wrapper for cat classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class CatClassifier:
    """Binary cat classifier backed by sklearn LogisticRegression."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=1000)
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: list[str]) -> "CatClassifier":
        """Train the classifier.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Class labels, e.g. ["cat_a", "cat_b", ...].

        Returns:
            self, for method chaining.
        """
        self._model.fit(X, y)
        self.classes_ = list(self._model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels for each row of *X*."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities; columns ordered by self.classes_."""
        return self._model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """Persist the fitted classifier to *path* using joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CatClassifier":
        """Load and return a CatClassifier previously saved with save()."""
        return joblib.load(path)
