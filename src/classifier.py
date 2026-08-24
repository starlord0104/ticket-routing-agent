"""
classifier.py
Logistic Regression on sentence embeddings + Temperature Scaling calibration.

Why Logistic Regression:
  • Trains in seconds once embeddings are computed
  • Easy to inspect (coefficients, feature importance via projection)
  • Calibration is straightforward to apply and explain

Temperature Scaling (Guo et al., 2017):
  Fits a single scalar T on a validation set by minimising NLL.
  Scaled probability = softmax(logits / T).
  T > 1  → flattens the distribution (reduces overconfidence).
  T = 1  → no change (raw softmax).
  T < 1  → sharpens the distribution.
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CATEGORIES, MODELS_DIR, RANDOM_STATE,
    DEFAULT_THRESHOLD, TEMPERATURE_INIT
)


# paths
_CLF_PATH    = MODELS_DIR / "classifier.pkl"
_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
_TEMP_PATH   = MODELS_DIR / "temperature.pkl"


# training

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Fit Logistic Regression on embedding vectors."""
    print("[classifier] Training Logistic Regression …")
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_STATE,
        solver="lbfgs",   # lbfgs uses multinomial loss by default (multi_class removed in sklearn 1.5)
    )
    clf.fit(X_train, y_train)
    print(f"[classifier] Training accuracy: {clf.score(X_train, y_train):.4f}")
    return clf


def get_logits(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return raw decision-function scores (logits) for each sample."""
    return clf.decision_function(X)


# temperature scaling

def _nll_loss(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood with temperature T (scalar to minimise)."""
    scaled_probs = softmax(logits / T, axis=1)
    eps = 1e-8
    nll = -np.mean(
        np.log(scaled_probs[np.arange(len(labels)), labels] + eps)
    )
    return nll


def fit_temperature(
    clf: LogisticRegression,
    X_val: np.ndarray,
    y_val: np.ndarray,
    init: float = TEMPERATURE_INIT,
) -> float:
    """Find optimal temperature T on the validation set.

    This is the key calibration step. After fitting, the model's
    confidence scores will match observed accuracy — a 0.9 confidence
    prediction will be correct ~90% of the time.
    """
    logits = get_logits(clf, X_val)
    result = minimize_scalar(
        lambda T: _nll_loss(T, logits, y_val),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    T_opt = result.x
    print(f"[classifier] Optimal temperature: {T_opt:.4f} (init={init})")
    print(f"[classifier] NLL before: {_nll_loss(1.0, logits, y_val):.4f} "
          f"→ after: {result.fun:.4f}")
    return float(T_opt)


# inference

def predict_proba_calibrated(
    clf: LogisticRegression,
    X: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return calibrated class probabilities for embedding matrix X."""
    logits = get_logits(clf, X)
    return softmax(logits / temperature, axis=1).astype(np.float32)


class TicketRouter:
    """High-level inference object — this is what the API uses."""

    def __init__(
        self,
        clf: LogisticRegression,
        label_encoder: LabelEncoder,
        temperature: float,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.clf           = clf
        self.label_encoder = label_encoder
        self.temperature   = temperature
        self.threshold     = threshold

    def route(
        self,
        embeddings: np.ndarray,
    ) -> list[dict]:
        """Route a batch of embedding vectors.

        Returns a list of dicts:
          {
            category:   str,   # predicted class label
            confidence: float, # calibrated max probability
            escalate:   bool,  # True if confidence < threshold
            probs:      dict,  # {category: prob} for all classes
          }
        """
        probs  = predict_proba_calibrated(self.clf, embeddings, self.temperature)
        idxs   = probs.argmax(axis=1)
        confs  = probs.max(axis=1)
        labels = self.label_encoder.inverse_transform(idxs)
        classes = self.label_encoder.classes_

        results = []
        for label, conf, prob_row in zip(labels, confs, probs):
            results.append({
                "category":   str(label),
                "confidence": float(conf),
                "escalate":   bool(conf < self.threshold),
                "probs":      {str(c): float(p) for c, p in zip(classes, prob_row)},
            })
        return results

    # persistence

    def save(
        self,
        clf_path    = _CLF_PATH,
        enc_path    = _ENCODER_PATH,
        temp_path   = _TEMP_PATH,
    ):
        joblib.dump(self.clf,           clf_path)
        joblib.dump(self.label_encoder, enc_path)
        joblib.dump(
            {"temperature": self.temperature, "threshold": self.threshold},
            temp_path,
        )
        print(f"[classifier] Saved classifier, encoder, and calibration params.")

    @classmethod
    def load(
        cls,
        clf_path    = _CLF_PATH,
        enc_path    = _ENCODER_PATH,
        temp_path   = _TEMP_PATH,
    ) -> "TicketRouter":
        clf     = joblib.load(clf_path)
        enc     = joblib.load(enc_path)
        params  = joblib.load(temp_path)
        return cls(
            clf=clf,
            label_encoder=enc,
            temperature=params["temperature"],
            threshold=params["threshold"],
        )

    def set_threshold(self, tau: float):
        """Adjust escalation threshold at runtime (e.g. from the Streamlit slider)."""
        self.threshold = tau
