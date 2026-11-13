"""Load and validate the built-in scikit-learn handwritten-digit images."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
IMAGE_SHAPE = (8, 8)
CLASSES = tuple(range(10))


def load_dataset():
    dataset = load_digits()
    images = np.asarray(dataset.images, dtype=np.float64)
    labels = np.asarray(dataset.target, dtype=np.int64)
    validate_dataset(images, labels)
    return images, labels


def validate_dataset(images, labels):
    if images.ndim != 3 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError("Expected images with shape (N, 8, 8)")
    if labels.shape != (len(images),):
        raise ValueError("Expected one label per image")
    if len(images) < 1000:
        raise ValueError("Unexpectedly small dataset")
    if not np.isfinite(images).all():
        raise ValueError("Images have non-finite values")
    if np.any((images < 0) | (images > 16)):
        raise ValueError("Pixel values must be in [0, 16]")
    if set(np.unique(labels)) != set(CLASSES):
        raise ValueError("Labels must cover classes 0 through 9")
    counts = np.bincount(labels, minlength=10)
    if counts.min() < 100:
        raise ValueError("Insufficient examples for at least one class")


def split_data(images, labels):
    # 60/20/20; first isolate test, then validation from development set.
    indices = np.arange(len(labels))
    dev_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        dev_idx, test_size=0.25, random_state=SEED, stratify=labels[dev_idx]
    )
    return (train_idx, val_idx, test_idx)


def to_features(images):
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError("Expected a batch of 8x8 images")
    return images.reshape(len(images), 64)
