"""
Unit tests for src/audit.py — all run without a trained model or CSV.
"""
import json
import uuid
from pathlib import Path

import pytest


def test_log_prediction_creates_file(tmp_path, monkeypatch):
    """log_prediction should append a valid JSON line to AUDIT_LOG_PATH."""
    audit_path = tmp_path / "logs" / "audit.jsonl"

    import src.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "AUDIT_LOG_PATH", audit_path)

    # Re-import so audit.py picks up the monkeypatched path.
    import importlib
    import src.audit as audit_mod
    importlib.reload(audit_mod)

    audit_mod.log_prediction(
        ticket_id="test-id-1",
        text="VPN is down",
        category="Infrastructure",
        confidence=0.91,
        escalate=False,
        ood=False,
        embedding_mode="hybrid",
        latency_ms=12.3,
    )

    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["type"]     == "predict"
    assert entry["category"] == "Infrastructure"
    assert entry["ticket_id"] == "test-id-1"


def test_log_feedback_appends(tmp_path, monkeypatch):
    audit_path = tmp_path / "logs" / "audit.jsonl"

    import src.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "AUDIT_LOG_PATH", audit_path)

    import importlib
    import src.audit as audit_mod
    importlib.reload(audit_mod)

    audit_mod.log_feedback(
        ticket_id="test-id-2",
        predicted_category="Infrastructure",
        correct_category="Access Management",
        agent_id="agent-1",
    )

    entry = json.loads(audit_path.read_text(encoding="utf-8").strip())
    assert entry["type"]               == "feedback"
    assert entry["predicted_category"] == "Infrastructure"
    assert entry["correct_category"]   == "Access Management"


def test_compute_kpis_empty_log(tmp_path, monkeypatch):
    """compute_kpis on a missing log should return zero counts, no alert."""
    missing_path = tmp_path / "logs" / "nonexistent.jsonl"

    import src.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "AUDIT_LOG_PATH", missing_path)

    import importlib
    import src.audit as audit_mod
    importlib.reload(audit_mod)

    kpis = audit_mod.compute_kpis(window_hours=24)
    assert kpis["total_predictions"] == 0
    assert kpis["alert"] is False


def test_compute_kpis_detects_high_escalation(tmp_path, monkeypatch):
    """compute_kpis should set alert=True when escalation_rate > 40%."""
    audit_path = tmp_path / "logs" / "audit.jsonl"

    import src.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "AUDIT_LOG_PATH", audit_path)

    import importlib
    import src.audit as audit_mod
    importlib.reload(audit_mod)

    # Write 5 predictions: 3 escalated → 60% escalation rate
    for i, esc in enumerate([True, True, True, False, False]):
        audit_mod.log_prediction(
            ticket_id=str(i),
            text="test",
            category="General IT",
            confidence=0.5 if esc else 0.9,
            escalate=esc,
            ood=False,
            embedding_mode="hybrid",
            latency_ms=10.0,
        )

    kpis = audit_mod.compute_kpis(window_hours=24)
    assert kpis["total_predictions"] == 5
    assert kpis["escalation_rate"] == pytest.approx(0.6)
    assert kpis["alert"] is True
    assert any("escalation_rate" in r for r in kpis["alert_reasons"])
