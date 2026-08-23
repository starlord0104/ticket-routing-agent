"""
spaces_app.py
─────────────
Standalone Streamlit app for HuggingFace Spaces.

Identical UI to app/streamlit_app.py, but routing logic is called directly
(no HTTP / no FastAPI process needed). Models are loaded once at startup via
@st.cache_resource.

Run locally:    streamlit run spaces_app.py
On HF Spaces:   set app_file: spaces_app.py in README.md frontmatter
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from scipy.special import softmax

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    CATEGORIES, DEFAULT_THRESHOLD, PLOTS_DIR, MODELS_DIR,
    OOD_ENTROPY_THRESHOLD, OOD_SIMILARITY_THRESHOLD, TOP_K,
)
from src.ood import ood_decision
from src.audit import log_prediction, log_feedback, compute_kpis


# ── Model bootstrap (Streamlit Community Cloud / HF Spaces) ───────────────────
# If running in the cloud and models/ isn't committed, download them from the
# HuggingFace model repo specified by the HF_MODEL_REPO env var.
# Set this in the Streamlit Cloud "Secrets" or as an environment variable:
#   HF_MODEL_REPO = starlord0104/ticket-routing-models
_HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")

def _maybe_download_models() -> None:
    """Download models from HuggingFace if MODELS_DIR is empty."""
    if (MODELS_DIR / "classifier.pkl").exists():
        return                            # already present — nothing to do
    if not _HF_MODEL_REPO:
        return                            # no repo configured — show error in UI
    try:
        from huggingface_hub import snapshot_download
        st.info(f"Downloading model artefacts from `{_HF_MODEL_REPO}` …")
        snapshot_download(
            repo_id=_HF_MODEL_REPO,
            repo_type="model",
            local_dir=str(Path(__file__).parent),   # downloads models/ and plots/ here
            ignore_patterns=["*.md", ".gitattributes"],
        )
        st.rerun()   # reload page so load_models() picks up the new files
    except Exception as exc:
        st.error(f"Model download failed: {exc}")

_maybe_download_models()


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Ticket Routing System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Model loading (cached — runs once per Spaces instance) ─────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    """
    Load all inference artefacts from models/.

    Returns (clf, le, temperature, threshold, index, meta, encode_clf, encode_faiss).
    encode_clf  → feature matrix for the classifier
    encode_faiss → dense float32 L2-normalised vectors for FAISS
    """
    import faiss

    clf        = joblib.load(MODELS_DIR / "classifier.pkl")
    le         = joblib.load(MODELS_DIR / "label_encoder.pkl")
    params     = joblib.load(MODELS_DIR / "temperature.pkl")
    temperature = params["temperature"]
    threshold   = params["threshold"]

    mode_path = MODELS_DIR / "embedding_mode.pkl"
    mode = joblib.load(mode_path)["mode"] if mode_path.exists() else "minilm"

    if mode == "minilm":
        from src.embeddings import encode as _enc
        encode_clf   = _enc
        encode_faiss = _enc

    elif mode == "tfidf":
        from sklearn.preprocessing import normalize as _sk_norm
        _tv = joblib.load(MODELS_DIR / "tfidf.pkl")
        _sv = joblib.load(MODELS_DIR / "svd.pkl")

        def encode_clf(texts, **_):
            return _tv.transform(texts)

        def encode_faiss(texts, **_):
            return _sk_norm(_sv.transform(_tv.transform(texts)).astype(np.float32), norm="l2")

    else:  # hybrid
        from sklearn.preprocessing import normalize as _sk_norm
        _tv = joblib.load(MODELS_DIR / "tfidf.pkl")
        from src.embeddings import encode as _enc_mini

        def encode_clf(texts, **_):
            return _tv.transform(texts)

        def encode_faiss(texts, **_):
            return _enc_mini(texts, show_progress=False)

    # FAISS index + metadata
    index, meta = None, None
    idx_path = MODELS_DIR / "faiss_index.bin"
    meta_path = MODELS_DIR / "faiss_metadata.pkl"
    if idx_path.exists() and meta_path.exists():
        index = faiss.read_index(str(idx_path))
        meta  = joblib.load(meta_path)

    return clf, le, temperature, threshold, index, meta, encode_clf, encode_faiss


def _models_present() -> bool:
    return (MODELS_DIR / "classifier.pkl").exists()


# ── Core routing function (mirrors /predict logic in app/main.py) ───────────────
def route_ticket(text: str, tau: float) -> dict:
    clf, le, temperature, _, index, meta, encode_clf, encode_faiss = load_models()

    emb    = encode_clf([text])
    logits = clf.decision_function(emb)
    probs  = softmax(logits / temperature, axis=1)[0]
    idx    = int(probs.argmax())
    conf   = float(probs[idx])
    cat    = str(le.classes_[idx])

    class_probs = {str(c): round(float(p), 4) for c, p in zip(le.classes_, probs)}

    historical:    list[dict] = []
    top_similarity = 0.0
    if index is not None:
        q    = encode_faiss([text])[0:1].astype(np.float32)
        D, I = index.search(q, TOP_K + 1)
        neighbours = [(int(j), float(s)) for j, s in zip(I[0], D[0]) if j >= 0][:TOP_K]
        if neighbours:
            top_similarity = neighbours[0][1]
        if conf >= tau and meta is not None:
            for rank, (i, sim) in enumerate(neighbours, 1):
                row = meta.iloc[i]
                historical.append({
                    "rank": rank, "similarity": round(sim, 4),
                    "category": str(row["label"]),
                    "preview": str(row["text"])[:300],
                })

    ood_info = ood_decision(
        probs, top_similarity,
        entropy_threshold=OOD_ENTROPY_THRESHOLD,
        similarity_threshold=OOD_SIMILARITY_THRESHOLD,
    )

    return {
        "category":          cat,
        "confidence":        round(conf, 4),
        "escalate":          conf < tau,
        "threshold_used":    tau,
        "class_probs":       class_probs,
        "historical_tickets": historical,
        "ood":               ood_info["ood"],
        "entropy":           round(ood_info["entropy"], 4),
        "ood_reasons":       ood_info["ood_reasons"],
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    tau = st.slider(
        "Confidence threshold τ",
        min_value=0.50, max_value=0.99,
        value=DEFAULT_THRESHOLD, step=0.01,
        help="Tickets below this confidence are escalated to a human agent.",
    )
    st.divider()
    st.markdown("### How the system works")
    st.markdown("""
1. Ticket text is **vectorised** (TF-IDF / MiniLM)
2. **Logistic classifier** predicts the routing queue
3. **Temperature scaling** calibrates the confidence score
4. Below τ → **escalate**; above τ → **auto-route**
5. **FAISS retrieval** surfaces the 3 most similar past tickets
""")
    st.divider()
    st.markdown("**Routing queues**")
    for cat in CATEGORIES:
        st.caption(f"• {cat}")
    st.divider()

    if _models_present():
        try:
            _, _, T, thr, *_ = load_models()
            st.success("✅ Models loaded")
            st.caption(f"T = `{T:.4f}`   τ = `{thr}`")
        except Exception as e:
            st.error(f"Model load error: {e}")
    else:
        st.error("Models not found. Run `python train.py` then commit the `models/` directory.")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_route, tab_metrics, tab_clusters, tab_monitor = st.tabs(
    ["🎫 Route Ticket", "📊 System Metrics", "🔁 Recurring Issues", "📈 Monitor"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROUTE TICKET
# ══════════════════════════════════════════════════════════════════════════════
SAMPLES = {
    "Infrastructure":    "The physical server in rack 3B is showing high temperature alerts. "
                         "Fans running at maximum speed, CPU throttling detected.",
    "Access Management": "New hire cannot log into Confluence or Jira after account was created "
                         "yesterday. SSO is returning a 403 on every attempt.",
    "Storage":           "Backup job failed overnight. Disk array showing 94% capacity. "
                         "Log rotation does not appear to be running correctly.",
    "HR Support":        "Requesting setup of laptop, email account, and VPN credentials "
                         "for three new interns joining the analytics team next Monday.",
    "Procurement":       "Please log receipt of PO-20481 — partial shipment of headsets arrived "
                         "but the invoice shows full quantity. Need reconciliation.",
    "Internal Project":  "Please amend the task assignments in the Q3 pipeline project. "
                         "The current owner left the team; reassign to the new lead.",
    "General IT":        "Please log a restart for the reporting server — it has been "
                         "unresponsive since this morning. Ref: monitoring alert #3912.",
}

with tab_route:
    st.title("Confidence-Aware IT Ticket Routing & Escalation System")
    st.caption(
        "Classifies support tickets across 7 operational queues with calibrated confidence. "
        "Tickets below the confidence threshold are escalated rather than auto-routed."
    )
    st.divider()

    if not _models_present():
        st.error(
            "**Models not found.** "
            "Run `python train.py` locally, then commit the `models/` directory to this Space."
        )
        st.stop()

    col_input, col_samples = st.columns([3, 1])
    with col_input:
        ticket_text = st.text_area(
            "Paste support ticket text",
            height=140,
            placeholder="Describe the IT issue …",
        )
    with col_samples:
        st.markdown("**Quick examples**")
        for cat in SAMPLES:
            if st.button(cat, use_container_width=True, key=f"sample_{cat}"):
                st.session_state["_sample"] = SAMPLES[cat]
                st.rerun()

    if "_sample" in st.session_state:
        ticket_text = st.session_state.pop("_sample")

    submit = st.button("🚀  Route ticket", type="primary", disabled=not ticket_text)

    if submit and ticket_text:
        with st.spinner("Routing …"):
            import time
            t0   = time.perf_counter()
            data = route_ticket(ticket_text, tau)
            data["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Audit log (best-effort; never blocks the UI)
        try:
            import uuid
            log_prediction(
                ticket_id=str(uuid.uuid4()),
                text=ticket_text,
                category=data["category"],
                confidence=data["confidence"],
                escalate=data["escalate"],
                ood=data["ood"],
                embedding_mode="hybrid",
                latency_ms=data["latency_ms"],
            )
        except Exception:
            pass

        st.divider()
        escalate = data["escalate"]
        conf     = data["confidence"]
        category = data["category"]

        if escalate:
            st.error(
                f"⚠️  **ESCALATE TO HUMAN AGENT**  —  "
                f"Confidence `{conf:.2%}` is below threshold `{tau:.2%}`. "
                f"Predicted queue: **{category}** (not acted on)."
            )
        else:
            st.success(f"✅  **Auto-routed → {category}**  —  Confidence `{conf:.2%}`")

        if data.get("ood"):
            reasons = ", ".join(data.get("ood_reasons", [])) or "uncertain input"
            st.warning(
                f"🛑  **Possible out-of-distribution ticket** — {reasons}. "
                f"Prediction entropy `{data.get('entropy', 0):.2f}`. "
                "This input may not belong to any of the 7 queues."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted queue",  category)
        m2.metric("Confidence",       f"{conf:.2%}")
        m3.metric("Decision",         "Escalate ⚠️" if escalate else "Auto-route ✅")
        m4.metric("Latency",          f"{data['latency_ms']:.0f} ms")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### Class probabilities")
            probs = dict(sorted(data["class_probs"].items(), key=lambda x: x[1], reverse=True))
            fig = go.Figure(go.Bar(
                x=list(probs.values()), y=list(probs.keys()),
                orientation="h",
                marker_color=["#2ecc71" if k == category else "#bdc3c7" for k in probs],
                text=[f"{v:.1%}" for v in probs.values()],
                textposition="outside",
            ))
            fig.add_vline(x=tau, line_dash="dash", line_color="red",
                          annotation_text=f"τ={tau}", annotation_position="top right")
            fig.update_layout(
                xaxis=dict(range=[0, 1], title="Calibrated probability"),
                yaxis=dict(autorange="reversed"),
                height=260, margin=dict(l=10, r=60, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("#### Similar historical tickets")
            st.caption("Retrieved by FAISS nearest-neighbour. category-match@3 = 0.79")
            similar = data.get("historical_tickets", [])
            if escalate:
                st.info("Retrieval skipped — ticket escalated to human agent.")
            elif not similar:
                st.info("No similar tickets found in the index.")
            else:
                for t in similar:
                    with st.expander(
                        f"#{t['rank']}  ·  Queue: {t['category']}  ·  Similarity {t['similarity']:.3f}"
                    ):
                        st.markdown(t["preview"])

        st.markdown("#### Confidence gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e74c3c" if escalate else "#2ecc71"},
                "steps": [
                    {"range": [0, tau * 100],   "color": "#fadbd8"},
                    {"range": [tau * 100, 100], "color": "#d5f5e3"},
                ],
                "threshold": {"line": {"color": "red", "width": 3},
                              "thickness": 0.75, "value": tau * 100},
            },
            title={"text": f"Threshold τ = {tau:.2%}"},
        ))
        gauge.update_layout(height=200, margin=dict(t=20, b=0, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SYSTEM METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("## System metrics  (hybrid model — measured on 9,567 test tickets)")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Macro-F1",            "0.86")
    met2.metric("ECE after calibration","0.012")
    met3.metric("Coverage @ τ=0.75",   "74.7%")
    met4.metric("Routing acc @ τ=0.75","0.942")

    st.divider()
    pc1, pc2, pc3 = st.columns(3)
    for col, fname, title in zip(
        [pc1, pc2, pc3],
        ["reliability_diagram.png", "coverage_accuracy_curve.png", "confusion_matrix.png"],
        ["Reliability diagram", "Coverage–accuracy curve", "Confusion matrix"],
    ):
        path = PLOTS_DIR / fname
        with col:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Run `python evaluate.py` to generate.")

    st.divider()
    st.markdown("### Per-class breakdown")
    per_class = pd.DataFrame({
        "Queue":        ["Procurement","Storage","Access Management","HR Support",
                         "Infrastructure","General IT","Internal Project"],
        "Precision":    [0.97, 0.94, 0.89, 0.86, 0.80, 0.84, 0.91],
        "Recall":       [0.88, 0.81, 0.83, 0.87, 0.89, 0.83, 0.76],
        "F1":           [0.93, 0.87, 0.86, 0.86, 0.84, 0.84, 0.83],
        "Test support": [493, 555, 1777, 2183, 2723, 1412, 424],
    })
    st.dataframe(per_class.set_index("Queue"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RECURRING ISSUES (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.markdown("## Recurring issue detection")
    st.caption(
        "DBSCAN clustering on ticket embeddings surfaces groups of similar tickets "
        "that may be candidates for automation or SLA escalation."
    )

    run_btn = st.button("🔍  Run cluster analysis (takes ~20 s)", type="secondary")
    if run_btn:
        rag_emb_path = MODELS_DIR / "rag_embeddings.npy"
        meta_path    = MODELS_DIR / "faiss_metadata.pkl"

        if not rag_emb_path.exists() or not meta_path.exists():
            st.error("Model files not found. Run `python train.py` first.")
        else:
            with st.spinner("Running DBSCAN on ticket embeddings …"):
                from src.cluster import cluster_tickets, get_automation_candidates
                X_all = np.load(str(rag_emb_path))
                meta  = joblib.load(meta_path)
                rng   = np.random.RandomState(42)
                idx   = rng.choice(len(X_all), min(5000, len(X_all)), replace=False)
                cluster_df = cluster_tickets(
                    X_all[idx], meta.iloc[idx]["text"].tolist(),
                    meta.iloc[idx]["label"].tolist(), eps=0.35, min_samples=4,
                )
                flagged = get_automation_candidates(cluster_df)

            n_clusters = cluster_df["cluster_id"].nunique() - (1 if -1 in cluster_df["cluster_id"].values else 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters found",        n_clusters)
            c2.metric("Noise tickets",         f"{cluster_df['is_noise'].mean()*100:.1f}%")
            c3.metric("Automation candidates", len(flagged) if not flagged.empty else 0)
            st.divider()
            if flagged.empty:
                st.info("No clusters large enough to flag for automation at this sample size.")
            else:
                st.markdown("### Top automation candidates")
                for _, row in flagged.head(8).iterrows():
                    with st.expander(
                        f"Cluster {int(row['cluster_id'])}  ·  "
                        f"{int(row['size'])} tickets  ·  Queue: {row['top_label']}"
                    ):
                        st.markdown(f"**Sample:** {row['sample_text'][:300]}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.markdown("## Live monitoring")
    st.caption(
        "Rolling-window KPIs from the prediction audit log (`logs/audit.jsonl`). "
        "Alerts fire when escalation, OOD, or correction rates exceed configured thresholds."
    )

    window = st.selectbox("Window", [1, 6, 12, 24, 48, 168], index=3,
                          format_func=lambda h: f"Last {h} h")
    if st.button("🔄  Refresh"):
        st.rerun()

    kpis = compute_kpis(window_hours=window)

    if kpis["total_predictions"] == 0:
        st.info(f"No predictions logged in the last {window} h. Route some tickets first.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predictions",     kpis["total_predictions"])
        k2.metric("Escalation rate", f"{kpis['escalation_rate']:.1%}")
        k3.metric("OOD rate",        f"{kpis['ood_rate']:.1%}")
        k4.metric("Correction rate", f"{kpis['correction_rate']:.1%}")

        if kpis["alert"]:
            for reason in kpis["alert_reasons"]:
                st.error(f"🚨 {reason}")
        else:
            st.success("✅ All KPIs within normal range.")
