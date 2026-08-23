"""
src/audit.py
────────────
Append-only JSONL audit log + escalation-rate monitor.

Every /predict call appends one line; /feedback appends another.
The monitor reads the last N hours of entries to compute KPIs.

Log location: AUDIT_LOG_PATH in src/config.py  (default: logs/audit.jsonl).
The directory is created on first write if it does not exist.

Log entry shapes
────────────────
predict:
  {"ts": "...", "type": "predict", "ticket_id": "...", "text_hash": "...",
   "category": "...", "confidence": 0.92, "escalate": false,
   "ood": false, "embedding_mode": "hybrid", "latency_ms": 45.2}

feedback:
  {"ts": "...", "type": "feedback", "ticket_id": "...",
   "predicted_category": "...", "correct_category": "...",
   "agent_id": "...", "note": "..."}
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import AUDIT_LOG_PATH


# One global write lock — the FastAPI server is multi-threaded.
_lock = threading.Lock()


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(entry: dict) -> None:
    """Thread-safe append of a JSON object to AUDIT_LOG_PATH."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def text_hash(text: str) -> str:
    """SHA-256 of the raw text (first 16 hex chars — enough for correlation)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Writers ───────────────────────────────────────────────────────────────────

def log_prediction(
    *,
    ticket_id:      str,
    text:           str,
    category:       str,
    confidence:     float,
    escalate:       bool,
    ood:            bool,
    embedding_mode: str,
    latency_ms:     float,
) -> None:
    _append({
        "ts":             _ts(),
        "type":           "predict",
        "ticket_id":      ticket_id,
        "text_hash":      text_hash(text),
        "category":       category,
        "confidence":     round(confidence, 4),
        "escalate":       escalate,
        "ood":            ood,
        "embedding_mode": embedding_mode,
        "latency_ms":     round(latency_ms, 1),
    })


def log_feedback(
    *,
    ticket_id:          str,
    predicted_category: str,
    correct_category:   str,
    agent_id:           Optional[str] = None,
    note:               Optional[str] = None,
) -> None:
    _append({
        "ts":                 _ts(),
        "type":               "feedback",
        "ticket_id":          ticket_id,
        "predicted_category": predicted_category,
        "correct_category":   correct_category,
        "agent_id":           agent_id,
        "note":               note,
    })


# ── Monitor ───────────────────────────────────────────────────────────────────

def compute_kpis(window_hours: int = 24) -> dict:
    """
    Read the audit log and compute KPIs for the last `window_hours`.

    Returns a dict with:
      total_predictions   int
      escalation_rate     float   fraction that were escalated
      ood_rate            float   fraction flagged OOD
      correction_rate     float   fraction of predictions that received a
                                  feedback correction  (requires /feedback calls)
      alert               bool    True if any KPI exceeds its alert threshold
      alert_reasons       list[str]
      window_hours        int
    """
    if not AUDIT_LOG_PATH.exists():
        return _empty_kpis(window_hours)

    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    predictions: list[dict] = []
    feedbacks:   list[dict] = []

    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                ts = datetime.fromisoformat(entry["ts"])
                # Make tz-aware if it isn't (old entries written without tz)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            if ts < cutoff:
                continue
            if entry.get("type") == "predict":
                predictions.append(entry)
            elif entry.get("type") == "feedback":
                feedbacks.append(entry)

    n = len(predictions)
    if n == 0:
        return _empty_kpis(window_hours)

    escalation_rate = sum(1 for p in predictions if p.get("escalate")) / n
    ood_rate        = sum(1 for p in predictions if p.get("ood"))       / n

    # Correction rate: feedbacks where predicted ≠ correct / total predictions
    corrections   = sum(
        1 for fb in feedbacks
        if fb.get("predicted_category") != fb.get("correct_category")
    )
    correction_rate = corrections / n

    # Alert thresholds — tune in config if you have real traffic data.
    # Defaults are conservative for a freshly deployed system.
    alert_reasons: list[str] = []
    if escalation_rate > 0.40:
        alert_reasons.append(
            f"escalation_rate={escalation_rate:.1%} > 40% (confidence degraded?)"
        )
    if ood_rate > 0.20:
        alert_reasons.append(
            f"ood_rate={ood_rate:.1%} > 20% (traffic shifted off-domain?)"
        )
    if correction_rate > 0.10:
        alert_reasons.append(
            f"correction_rate={correction_rate:.1%} > 10% (agents overriding often)"
        )

    return {
        "total_predictions": n,
        "escalation_rate":   round(escalation_rate, 4),
        "ood_rate":          round(ood_rate, 4),
        "correction_rate":   round(correction_rate, 4),
        "alert":             bool(alert_reasons),
        "alert_reasons":     alert_reasons,
        "window_hours":      window_hours,
    }


def _empty_kpis(window_hours: int) -> dict:
    return {
        "total_predictions": 0,
        "escalation_rate":   0.0,
        "ood_rate":          0.0,
        "correction_rate":   0.0,
        "alert":             False,
        "alert_reasons":     [],
        "window_hours":      window_hours,
    }
