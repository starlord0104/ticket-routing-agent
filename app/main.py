"""
app/main.py
───────────
FastAPI backend for the Confidence-Aware IT Ticket Routing & Escalation System.

Endpoints:
  POST /predict       — classify + retrieve historical tickets
  GET  /health        — liveness check
  GET  /categories    — list of operational queues
  POST /set-threshold — adjust τ at runtime

Run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    CATEGORIES, DEFAULT_THRESHOLD, TOP_K, MODELS_DIR,
    OOD_ENTROPY_THRESHOLD, OOD_SIMILARITY_THRESHOLD,
)
from src.ood import ood_decision


# ── State ──────────────────────────────────────────────────────────────────────
_clf        = None
_le         = None
_tfidf      = None
_svd        = None
_index      = None
_meta       = None
_temperature = 1.0
_threshold   = DEFAULT_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _clf, _le, _tfidf, _svd, _index, _meta, _temperature, _threshold
    import faiss
    from sklearn.preprocessing import normalize

    print("[api] Loading models …")
    _clf   = joblib.load(MODELS_DIR / "classifier.pkl")
    _le    = joblib.load(MODELS_DIR / "label_encoder.pkl")
    params = joblib.load(MODELS_DIR / "temperature.pkl")
    _temperature = params["temperature"]
    _threshold   = params["threshold"]

    # Determine which embedding backend was used at training time.
    # encode_clf  → input for the classifier  (sparse for tfidf, dense for minilm)
    # encode_faiss → input for FAISS search   (always dense float32, L2-normalised)
    _mode_path = MODELS_DIR / "embedding_mode.pkl"
    _embedding_mode = joblib.load(_mode_path)["mode"] if _mode_path.exists() else "minilm"
    print(f"[api] Embedding mode: {_embedding_mode}")

    _expected_dim = _clf.coef_.shape[1]

    if _embedding_mode == "minilm":
        try:
            from src.embeddings import encode as _encode_minilm
        except Exception as exc:
            raise RuntimeError(
                "[api] MiniLM embedding model could not be loaded. "
                "Ensure 'sentence-transformers' is installed and the model cache "
                f"is accessible. Original error: {exc}"
            ) from exc
        _probe = _encode_minilm(["probe"], show_progress=False)
        if _probe.shape[1] != _expected_dim:
            raise RuntimeError(
                f"[api] Embedding dim mismatch: encoder → {_probe.shape[1]}-dim, "
                f"classifier expects {_expected_dim}-dim. Re-run train.py."
            )
        print(f"[api] MiniLM embeddings OK (dim={_probe.shape[1]}).")
        _encode_clf   = _encode_minilm   # dense → classifier
        _encode_faiss = _encode_minilm   # same dense → FAISS

    elif _embedding_mode == "tfidf":
        from sklearn.preprocessing import normalize as _sk_norm
        _tfidf_v = joblib.load(MODELS_DIR / "tfidf.pkl")
        _svd_v   = joblib.load(MODELS_DIR / "svd.pkl")

        def _encode_clf(texts, show_progress=False):
            """Sparse TF-IDF features — what the classifier was trained on."""
            return _tfidf_v.transform(texts)

        def _encode_faiss(texts, show_progress=False):
            """Dense SVD-reduced features — what the FAISS index stores."""
            X = _tfidf_v.transform(texts)
            return _sk_norm(_svd_v.transform(X).astype(np.float32), norm="l2")

        _probe_sparse = _encode_clf(["probe"])
        if _probe_sparse.shape[1] != _expected_dim:
            raise RuntimeError(
                f"[api] TF-IDF dim mismatch: vectorizer → {_probe_sparse.shape[1]}-dim, "
                f"classifier expects {_expected_dim}-dim. Re-run train.py."
            )
        print(f"[api] TF-IDF embeddings OK "
              f"(classifier dim={_probe_sparse.shape[1]}, "
              f"FAISS dim={_svd_v.components_.shape[0]}).")

    else:  # hybrid — TF-IDF classifier + MiniLM FAISS retrieval
        from sklearn.preprocessing import normalize as _sk_norm
        _tfidf_v = joblib.load(MODELS_DIR / "tfidf.pkl")

        def _encode_clf(texts, show_progress=False):
            """Sparse TF-IDF features — what the TF-IDF classifier was trained on."""
            return _tfidf_v.transform(texts)

        try:
            from src.embeddings import encode as _encode_minilm
        except Exception as exc:
            raise RuntimeError(
                "[api] Hybrid mode requires MiniLM for retrieval but it could not be "
                f"loaded: {exc}"
            ) from exc

        def _encode_faiss(texts, show_progress=False):
            """Dense MiniLM embeddings — what the FAISS index stores (384-dim)."""
            return _encode_minilm(texts, show_progress=show_progress)

        _probe_sparse = _encode_clf(["probe"])
        if _probe_sparse.shape[1] != _expected_dim:
            raise RuntimeError(
                f"[api] TF-IDF dim mismatch: vectorizer → {_probe_sparse.shape[1]}-dim, "
                f"classifier expects {_expected_dim}-dim. Re-run train.py."
            )
        _probe_dense = _encode_faiss(["probe"])
        print(f"[api] Hybrid embeddings OK "
              f"(clf TF-IDF dim={_probe_sparse.shape[1]}, "
              f"FAISS MiniLM dim={_probe_dense.shape[1]}).")

    app.state.encode_clf   = _encode_clf
    app.state.encode_faiss = _encode_faiss
    app.state.embedding_mode = _embedding_mode

    # RAG index
    try:
        _index = faiss.read_index(str(MODELS_DIR / "faiss_index.bin"))
        _meta  = joblib.load(MODELS_DIR / "faiss_metadata.pkl")
        print(f"[api] FAISS index loaded: {_index.ntotal:,} vectors.")
    except FileNotFoundError:
        print("[api] WARNING: FAISS index not found. Historical retrieval disabled.")

    print("[api] Ready.")
    yield


app = FastAPI(
    title="IT Ticket Routing & Escalation System",
    description=(
        "Confidence-aware ticket classification with calibrated escalation. "
        "Routes tickets to one of 7 operational queues when confidence ≥ τ; "
        "escalates to a human agent when confidence < τ."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text:      str   = Field(..., min_length=5, max_length=5000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class HistoricalTicket(BaseModel):
    rank:       int
    similarity: float
    category:   str
    preview:    str


class PredictResponse(BaseModel):
    category:          str
    confidence:        float
    escalate:          bool
    threshold_used:    float
    class_probs:       dict[str, float]
    historical_tickets: list[HistoricalTicket]
    ood:               bool
    entropy:           float
    ood_reasons:       list[str]
    latency_ms:        float


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "temperature":    _temperature,
        "threshold":      _threshold,
        "rag_loaded":     _index is not None,
        "embedding_mode": getattr(app.state, "embedding_mode", "unknown"),
    }


@app.get("/categories")
def categories():
    return {"categories": CATEGORIES}


@app.post("/set-threshold")
def set_threshold(req: ThresholdRequest):
    global _threshold
    _threshold = req.threshold
    return {"threshold": _threshold}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    from scipy.special import softmax
    t0  = time.perf_counter()
    tau = req.threshold if req.threshold is not None else _threshold

    # Embed for classifier (sparse in tfidf mode, dense in minilm mode)
    emb    = app.state.encode_clf([req.text], show_progress=False)
    logits = _clf.decision_function(emb)
    probs  = softmax(logits / _temperature, axis=1)[0]
    idx    = int(probs.argmax())
    conf   = float(probs[idx])
    cat    = str(_le.classes_[idx])
    escalate = conf < tau

    class_probs = {str(c): round(float(p), 4) for c, p in zip(_le.classes_, probs)}

    # Nearest-neighbour retrieval. Always search (it's cheap and the top
    # similarity feeds OOD detection); only surface the neighbours when we're
    # not escalating.
    historical: list[HistoricalTicket] = []
    neighbours: list[tuple[int, float]] = []
    top_similarity = 0.0
    if _index is not None:
        # encode_faiss always returns dense float32, L2-normalised — no dim check needed.
        q = app.state.encode_faiss([req.text], show_progress=False)[0:1].astype(np.float32)

        # Vectors are L2-normalised, so the inner-product scores FAISS returns
        # are cosine similarities — use them directly instead of recomputing.
        D, I = _index.search(q, TOP_K + 1)
        neighbours = [(int(j), float(s)) for j, s in zip(I[0], D[0]) if j >= 0][:TOP_K]
        if neighbours:
            top_similarity = neighbours[0][1]

    if not escalate:
        for rank, (i, sim) in enumerate(neighbours, 1):
            row = _meta.iloc[i]
            historical.append(HistoricalTicket(
                rank=rank,
                similarity=round(sim, 4),
                category=str(row["label"]),
                preview=str(row["text"])[:300],
            ))

    # Out-of-distribution check: flag inputs the model is likely guessing on
    # (high entropy) or that resemble nothing in the corpus (low similarity).
    ood_info = ood_decision(
        probs,
        top_similarity,
        entropy_threshold=OOD_ENTROPY_THRESHOLD,
        similarity_threshold=OOD_SIMILARITY_THRESHOLD,
    )

    latency = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        category=cat,
        confidence=round(conf, 4),
        escalate=escalate,
        threshold_used=tau,
        class_probs=class_probs,
        historical_tickets=historical,
        ood=ood_info["ood"],
        entropy=round(ood_info["entropy"], 4),
        ood_reasons=ood_info["ood_reasons"],
        latency_ms=round(latency, 1),
    )
