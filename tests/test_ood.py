import numpy as np

from src.ood import (
    normalized_entropy,
    ood_decision,
)


def test_entropy_is_low_for_confident_prediction():
    assert normalized_entropy(
        np.array([0.98, 0.01, 0.01])
    ) < 0.2


def test_entropy_is_high_for_uniform_prediction():
    assert normalized_entropy(
        np.ones(7) / 7
    ) > 0.99


def test_ood_gate_triggers_on_low_similarity():
    out = ood_decision(
        np.array([
            0.90,
            0.02,
            0.02,
            0.02,
            0.01,
            0.01,
            0.02,
        ]),
        0.1,
        entropy_threshold=0.8,
        similarity_threshold=0.45,
    )

    assert out["ood"] is True
    assert "low similarity to known tickets" in out["ood_reasons"]