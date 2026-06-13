"""Feature extraction from cat images."""

from pathlib import Path

import numpy as np
from PIL import Image

_RESIZE = (256, 256)


def extract_features(path: str | Path) -> np.ndarray:
    """Return a fixed-length feature vector for the image at *path*.

    Features (in order):
        0  mean_r        — mean red channel value (0–255)
        1  mean_g        — mean green channel value (0–255)
        2  mean_b        — mean blue channel value (0–255)
        3  width_orig    — original image width in pixels
        4  height_orig   — original image height in pixels
        5  aspect_ratio  — width / height of the original image

    The image is resized to 256×256 internally before computing colour
    statistics so that all inputs produce the same feature length regardless
    of resolution.
    """
    img = Image.open(path).convert("RGB")
    width_orig, height_orig = img.size
    aspect_ratio = width_orig / height_orig

    img_resized = img.resize(_RESIZE)
    arr = np.asarray(img_resized, dtype=np.float32)  # shape (256, 256, 3)
    brightness = arr.mean()

    mean_r, mean_g, mean_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()

    return np.array([mean_r, mean_g, mean_b, width_orig, height_orig, aspect_ratio, brightness], dtype=np.float64)
