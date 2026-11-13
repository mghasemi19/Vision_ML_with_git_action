import numpy as np
import pytest

from vision_mlops.data import load_dataset, split_data, to_features
from vision_mlops.model import fit_models, metrics_for, quality_failures


@pytest.fixture(scope="module")
def trained():
    images, labels = load_dataset()
    train, _, test = split_data(images, labels)
    model, reference = fit_models(images, labels, train)
    return images, labels, test, model, reference


def test_quality_gate_on_held_out_test(trained):
    images, labels, test, candidate, reference = trained
    candidate_metrics = metrics_for(candidate, images[test], labels[test])
    ref_metrics = metrics_for(reference, images[test], labels[test])
    assert not quality_failures(candidate_metrics, ref_metrics)


def test_dummy_classifier_is_rejected(trained):
    images, labels, test, _, reference = trained
    candidate = metrics_for(reference, images[test], labels[test])
    baseline = metrics_for(reference, images[test], labels[test])
    assert quality_failures(candidate, baseline)


def test_prediction_contract(trained):
    images, labels, test, candidate, _ = trained
    predictions = candidate.predict(to_features(images[test[:20]]))
    probabilities = candidate.predict_proba(to_features(images[test[:20]]))
    assert predictions.shape == (20,)
    assert set(np.unique(predictions)).issubset(set(range(10)))
    assert probabilities.shape == (20, 10)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
