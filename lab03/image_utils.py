"""
image_utils.py
--------------
Utility for converting any image to a fixed-length grayscale feature vector.
"""
import numpy as np
from PIL import Image


def image_to_features(path: str, size: tuple = (64, 64)) -> np.ndarray:
    """Load an image, resize it, convert to grayscale and flatten.

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.
    size : tuple of int, optional
        Target (width, height) in pixels.  Default is (64, 64).

    Returns
    -------
    np.ndarray, shape (size[0] * size[1],), dtype float32
        Normalised pixel values in the range [0, 1].
    """
    img = Image.open(path).convert("RGB").resize(size)
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return gray.flatten()
