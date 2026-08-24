"""
app/main.py
FastAPI backend for the Confidence-Aware IT Ticket Routing & Escalation System.

Endpoints:
  POST /predict       — classify + retrieve historical tickets
  POST /feedback      — record agent correction for a previously routed ticket
  GET  /monitor       — escalation-rate & OOD-rate KPIs over a rolling window
  GET  /health        — liveness check
  GET  /categories    — list of operational queues
  POST /set-threshold — adjust τ at runtime

Run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations
import time
import uuid
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
from src.ood   import ood_decision
from src.audit import log_prediction, log_feedback, compute_kpis


# state
_clf         = None
_le          = None
_index       = None
_meta        = None
_temperature = 1.0
_threshold   = DEFAULT_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _clf, _le, _index, _meta, _temperature, _threshold
    import faiss
    from sklearn.preprocessing import normalize

    print("[api] Loading models …")
    _clf   = joblib.load(MODELS_DIR / "classifier.pkl")
    _le    = joblib.load(MODELS_DIR / "label_encoder.pkl")
    params = joblib.load(MODELS_DIR / "temperature.pkl")
    _temperature = params["temperature"]
    _threshold   = params["threshold"]

    # Determine which embedding backend was used at training time.
    # encode_clf  → input for the classifier  (sparse for tfidf/hybrid, dense for minilm)
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
        _encode_clf   = _encode_minilm
        _encode_faiss = _encode_minilm

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

    app.state.encode_clf    = _encode_clf
    app.state.encode_faiss  = _encode_faiss
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
        "escalates to a human agent when confidence < τ. "
        "Provides OOD detection, historical retrieval, agent feedback, and monitoring."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# schemas

class PredictRequest(BaseModel):
    text:      str   = Field(..., min_length=5, max_length=5000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class HistoricalTicket(BaseModel):
    rank:       int
    similarity: float
    category:   str
    preview:    str


class PredictResponse(BaseModel):
    ticket_id:          str          # server-generated; pass back to /feedback
    category:           str
    confidence:         float
    escalate:           bool
    threshold_used:     float
    class_probs:        dict[str, float]
    historical_tickets: list[HistoricalTicket]
    ood:                bool
    entropy:            float
    ood_reasons:        list[str]
    latency_ms:         float


class FeedbackRequest(BaseModel):
    ticket_id:          str = Field(..., description="ticket_id returned by /predict")
    predicted_category: str
    correct_category:   str
    agent_id:           Optional[str]  = Field(None, description="Agent or queue ID")
    note:               Optional[str]  = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    accepted: bool
    message:  str


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


# endpoints

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

    # Unique ID so the caller can correlate /feedback with this prediction.
    ticket_id = str(uuid.uuid4())

    # Embed for classifier (sparse in tfidf/hybrid mode, dense in minilm mode)
    emb    = app.state.encode_clf([req.text], show_progress=False)
    logits = _clf.decision_function(emb)
    probs  = softmax(logits / _temperature, axis=1)[0]
    idx    = int(probs.argmax())
    conf   = float(probs[idx])
    cat    = str(_le.classes_[idx])
    escalate = conf < tau

    class_probs = {str(c): round(float(p), 4) for c, p in zip(_le.classes_, probs)}

    # Nearest-neighbour retrieval. Always search (cheap; top_similarity feeds OOD).
    historical: list[HistoricalTicket] = []
    neighbours: list[tuple[int, float]] = []
    top_similarity = 0.0
    if _index is not None:
        q = app.state.encode_faiss([req.text], show_progress=False)[0:1].astype(np.float32)
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

    # OOD check — flags inputs the model is likely guessing on
    ood_info = ood_decision(
        probs, top_similarity,
        entropy_threshold=OOD_ENTROPY_THRESHOLD,
        similarity_threshold=OOD_SIMILARITY_THRESHOLD,
    )

    latency = (time.perf_counter() - t0) * 1000

    # Audit log — fire-and-forget (non-blocking; any IO error is swallowed to
    # keep the prediction path fast and fault-tolerant).
    try:
        log_prediction(
            ticket_id=ticket_id,
            text=req.text,
            category=cat,
            confidence=conf,
            escalate=escalate,
            ood=ood_info["ood"],
            embedding_mode=getattr(app.state, "embedding_mode", "unknown"),
            latency_ms=latency,
        )
    except Exception:
        pass   # audit failure must never break routing

    return PredictResponse(
        ticket_id=ticket_id,
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


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """
    Record an agent's correction for a previously routed ticket.

    Callers pass the `ticket_id` from the /predict response so each
    feedback entry can be correlated with the original prediction in
    the audit log.

    The data is appended to AUDIT_LOG_PATH (logs/audit.jsonl) and is
    used by /monitor to compute the correction_rate KPI.  It does NOT
    retrain the model — that is a separate offline step.
    """
    if req.predicted_category not in CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"predicted_category '{req.predicted_category}' is not a known queue.",
        )
    if req.correct_category not in CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"correct_category '{req.correct_category}' is not a known queue.",
        )

    try:
        log_feedback(
            ticket_id=req.ticket_id,
            predicted_category=req.predicted_category,
            correct_category=req.correct_category,
            agent_id=req.agent_id,
            note=req.note,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit log write failed: {exc}")

    same = req.predicted_category == req.correct_category
    msg  = "Routing confirmed." if same else (
        f"Correction recorded: {req.predicted_category!r} → {req.correct_category!r}."
    )
    return FeedbackResponse(accepted=True, message=msg)


@app.get("/monitor")
def monitor(window_hours: int = 24):
    """
    Rolling-window KPIs from the audit log.

    Returns escalation_rate, ood_rate, correction_rate over the last
    `window_hours` (default 24).  If any KPI exceeds its alert threshold,
    `alert=True` and `alert_reasons` lists what crossed the limit.

    Alert thresholds (configurable in src/audit.py):
      escalation_rate  > 40%  → model confidence may have degraded
      ood_rate         > 20%  → traffic may have shifted off-domain
      correction_rate  > 10%  → agents are overriding routing frequently
    """
    if window_hours < 1 or window_hours > 168:
        raise HTTPException(
            status_code=422,
            detail="window_hours must be between 1 and 168 (one week).",
        )
    return compute_kpis(window_hours=window_hours)
