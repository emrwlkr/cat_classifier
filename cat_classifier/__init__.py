"""cat_classifier — feature extraction and classification for cat photos."""

from .features import extract_features
from .classify import CatClassifier

__all__ = ["extract_features", "CatClassifier"]
