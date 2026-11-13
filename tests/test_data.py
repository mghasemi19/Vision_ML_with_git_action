import numpy as np
import pytest

from vision_mlops.data import load_dataset, split_data, to_features, validate_dataset


def test_schema_and_valid_ranges():
    images, labels = load_dataset()
    assert images.shape[1:] == (8, 8)
    assert len(images) == len(labels)
    assert set(np.unique(labels)) == set(range(10))


def test_no_split_overlap_and_full_coverage():
    images, labels = load_dataset()
    split = split_data(images, labels)
    sets = [set(x) for x in split]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == len(labels)


def test_reproducible_splits():
    images, labels = load_dataset()
    a, b = split_data(images, labels), split_data(images, labels)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_rejects_nan():
    images, labels = load_dataset()
    images[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset(images, labels)


def test_rejects_invalid_image_shape():
    with pytest.raises(ValueError, match="8x8"):
        to_features(np.zeros((1, 10, 10)))
