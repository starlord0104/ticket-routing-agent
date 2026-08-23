import numpy as np

from src.classifier import (
    train_classifier,
    fit_temperature,
    predict_proba_calibrated,
)


def test_training_and_calibration_shape():
    rng = np.random.RandomState(42)

    X = rng.normal(size=(60, 8))
    y = np.repeat([0, 1, 2], 20)

    clf = train_classifier(
        X[:45],
        y[:45],
    )

    T = fit_temperature(
        clf,
        X[45:],
        y[45:],
    )

    p = predict_proba_calibrated(
        clf,
        X[45:],
        T,
    )

    assert p.shape == (15, 3)
    assert np.allclose(
        p.sum(axis=1),
        1,
        atol=1e-5,
    )
    assert T > 0