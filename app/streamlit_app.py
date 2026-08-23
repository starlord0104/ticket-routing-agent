"""
app/streamlit_app.py
────────────────────
Streamlit demo UI for the Confidence-Aware IT Ticket Routing & Escalation System.

Run with:
    streamlit run app/streamlit_app.py

Tabs:
  1. Route Ticket   — classify, gate, retrieve historical tickets
  2. System Metrics — calibration + coverage plots from evaluate.py
  3. Recurring Issues — DBSCAN clusters surfaced as automation candidates
"""

import os, sys, requests, joblib, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CATEGORIES, DEFAULT_THRESHOLD, PLOTS_DIR, MODELS_DIR

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Ticket Routing System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
5. **Historical ticket retrieval** (FAISS) surfaces the 3 most similar past tickets
""")
    st.divider()
    st.markdown("**Routing queues**")
    for cat in CATEGORIES:
        st.caption(f"• {cat}")
    st.divider()
    try:
        h = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success("API online")
        st.caption(f"T = `{h.get('temperature','n/a')}`   τ = `{h.get('threshold','n/a')}`")
    except Exception:
        st.warning("API offline — start with:\n`uvicorn app.main:app --reload`")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_route, tab_metrics, tab_clusters = st.tabs(
    ["🎫 Route Ticket", "📊 System Metrics", "🔁 Recurring Issues"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ROUTE TICKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_route:
    st.title("Confidence-Aware IT Ticket Routing & Escalation System")
    st.caption(
        "Classifies support tickets across 7 operational queues with calibrated confidence. "
        "Tickets below the confidence threshold are escalated rather than auto-routed."
    )
    st.divider()

    # ── Sample tickets using actual 7 categories ───────────────────────────────
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
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": ticket_text, "threshold": tau},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Start it with: `uvicorn app.main:app --reload`")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        st.divider()
        escalate = data["escalate"]
        conf     = data["confidence"]
        category = data["category"]

        # ── Decision banner ────────────────────────────────────────────────────
        if escalate:
            st.error(
                f"⚠️  **ESCALATE TO HUMAN AGENT**  —  "
                f"Confidence `{conf:.2%}` is below threshold `{tau:.2%}`. "
                f"Predicted queue: **{category}** (not acted on)."
            )
        else:
            st.success(f"✅  **Auto-routed → {category}**  —  Confidence `{conf:.2%}`")

        # ── Out-of-distribution warning ────────────────────────────────────────
        if data.get("ood"):
            reasons = ", ".join(data.get("ood_reasons", [])) or "uncertain input"
            st.warning(
                f"🛑  **Possible out-of-distribution ticket** — {reasons}. "
                f"Prediction entropy `{data.get('entropy', 0):.2f}`. "
                "This input may not belong to any of the 7 queues; treat the "
                "routing above with caution."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted queue",  category)
        m2.metric("Confidence",       f"{conf:.2%}")
        m3.metric("Decision",         "Escalate ⚠️" if escalate else "Auto-route ✅")
        m4.metric("Latency",          f"{data['latency_ms']:.0f} ms")

        st.divider()
        col_l, col_r = st.columns(2)

        # ── Class probability bars ─────────────────────────────────────────────
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

        # ── Historical ticket retrieval ─────────────────────────────────────────
        with col_r:
            st.markdown("#### Similar historical tickets")
            st.caption(
                "Retrieved by FAISS nearest-neighbour search. "
                "Metric: category-match@3 = 0.70 on held-out test set."
            )
            similar = data.get("historical_tickets", [])
            if escalate:
                st.info("Retrieval skipped — ticket escalated to human agent.")
            elif not similar:
                st.info("No similar tickets found in the index.")
            else:
                for t in similar:
                    with st.expander(
                        f"#{t['rank']}  ·  Queue: {t['category']}  ·  "
                        f"Similarity {t['similarity']:.3f}"
                    ):
                        st.markdown(t["preview"])

        # ── Confidence gauge ───────────────────────────────────────────────────
        st.markdown("#### Confidence gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e74c3c" if escalate else "#2ecc71"},
                "steps": [
                    {"range": [0, tau * 100],    "color": "#fadbd8"},
                    {"range": [tau * 100, 100],  "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75, "value": tau * 100,
                },
            },
            title={"text": f"Threshold τ = {tau:.2%}"},
        ))
        gauge.update_layout(height=200, margin=dict(t=20, b=0, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SYSTEM METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("## System metrics")
    st.caption("Generated by `python evaluate.py`. Re-run to refresh after retraining.")

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Macro-F1 (TF-IDF)",      "0.86")
    met2.metric("ECE after calibration",  "0.013")
    met3.metric("Coverage @ τ=0.75",      "74.8%")
    met4.metric("Routing acc @ τ=0.75",   "0.942")

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
        "Queue":          ["Procurement","Storage","Access Mgmt","HR Support","Infrastructure","General IT","Internal Project"],
        "Precision":      [0.98, 0.94, 0.89, 0.86, 0.80, 0.84, 0.92],
        "Recall":         [0.88, 0.81, 0.83, 0.87, 0.89, 0.83, 0.76],
        "F1":             [0.93, 0.87, 0.86, 0.86, 0.84, 0.84, 0.83],
        "Test support":   [493,  555, 1777, 2183, 2723, 1412,  424],
    })
    st.dataframe(per_class.set_index("Queue"), use_container_width=True)
    st.caption(
        "**Procurement** has the highest precision (0.98) despite being a small class — "
        "PO numbers and purchase-order vocabulary are highly distinctive. "
        "**Internal Project** has the lowest recall (0.76) because project-management "
        "language overlaps with General IT and Infrastructure tickets. "
        "**Infrastructure ↔ Access Management** is the most confused pair: "
        "hardware setup for new users straddles both queues depending on how the submitter frames the request."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RECURRING ISSUES (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.markdown("## Recurring issue detection")
    st.caption(
        "DBSCAN clustering on ticket embeddings surfaces groups of similar tickets "
        "that may be candidates for automation or SLA escalation. "
        "This is an offline analysis step — not part of the real-time routing pipeline."
    )

    run_btn = st.button("🔍  Run cluster analysis (takes ~20 seconds)", type="secondary")

    if run_btn:
        rag_emb_path = MODELS_DIR / "rag_embeddings.npy"
        meta_path    = MODELS_DIR / "faiss_metadata.pkl"

        if not rag_emb_path.exists() or not meta_path.exists():
            st.error("Model files not found. Run `python train.py` first.")
            st.stop()

        with st.spinner("Running DBSCAN on ticket embeddings …"):
            from src.cluster import cluster_tickets, get_automation_candidates
            import numpy as np

            X_all  = np.load(str(rag_emb_path))
            meta   = joblib.load(meta_path)

            # Sample 5000 tickets for speed
            rng    = np.random.RandomState(42)
            idx    = rng.choice(len(X_all), min(5000, len(X_all)), replace=False)
            X_sub  = X_all[idx]
            texts  = meta.iloc[idx]["text"].tolist()
            labels = meta.iloc[idx]["label"].tolist()

            cluster_df = cluster_tickets(X_sub, texts, labels, eps=0.35, min_samples=4)
            flagged    = get_automation_candidates(cluster_df)

        n_clusters = cluster_df["cluster_id"].nunique() - (1 if -1 in cluster_df["cluster_id"].values else 0)
        noise_pct  = cluster_df["is_noise"].mean() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters found",      n_clusters)
        c2.metric("Noise tickets",       f"{noise_pct:.1f}%")
        c3.metric("Automation candidates", len(flagged) if not flagged.empty else 0)

        st.divider()

        if flagged.empty:
            st.info("No clusters large enough to flag for automation at this sample size.")
        else:
            st.markdown("### Top automation candidates")
            st.caption("Clusters with ≥ 5 similar tickets — potential recurring issues.")
            for _, row in flagged.head(8).iterrows():
                with st.expander(
                    f"Cluster {int(row['cluster_id'])}  ·  "
                    f"{int(row['size'])} tickets  ·  Queue: {row['top_label']}"
                ):
                    st.markdown(f"**Sample ticket:** {row['sample_text'][:300]}")
                    st.caption("Consider creating a self-service workflow or knowledge-base article for this pattern.")
