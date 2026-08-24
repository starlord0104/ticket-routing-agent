---
title: IT Ticket Routing Agent
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: spaces_app.py
pinned: false
---

# Confidence-Aware IT Ticket Routing & Escalation System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://ticket-routing-agent.streamlit.app/)
[![Model](https://img.shields.io/badge/Model-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/starlord0104/ticket-routing-minilm-finetuned)
[![CI](https://github.com/starlord0104/ticket-routing-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/starlord0104/ticket-routing-agent/actions)

**[→ Try the live app](https://ticket-routing-agent.streamlit.app/)**

Routes IT support tickets to 7 operational queues using a fine-tuned transformer classifier with calibrated confidence scores. Tickets below a tunable threshold are escalated to human agents rather than auto-routed.

---

## What it does

Classifies free-text IT tickets across 7 queues — Infrastructure, Access Management, Storage, HR Support, Procurement, Internal Project, General IT — using a MiniLM transformer fine-tuned end-to-end on 38k real helpdesk tickets. Raw softmax scores are calibrated via temperature scaling so that a 0.9 confidence prediction is correct ~90% of the time. A threshold τ gates auto-routing: tickets below τ escalate instead of being silently misrouted.

This is a **confidence-aware routing system**, not an autonomous agent. It classifies and routes. The only autonomous decision it makes is whether confidence is high enough to act without a human.

---

## End-to-end example

**Ticket submitted:**
> "User Sarah Johnson cannot log into Confluence or Jira after returning from
> maternity leave. IT says her account was suspended during her absence."

| Step | What happens |
|------|-------------|
| Preprocessing | lowercase · strip HTML · remove ticket IDs |
| Encoding | Fine-tuned MiniLM-L6-v2 — 7-class sequence classifier |
| Calibration | Temperature scaling → calibrated confidence gates the decision |
| Gate | confidence ≥ τ=0.75 → **auto-route**; below τ → **escalate** |
| Retrieval | FAISS returns 3 most similar historical tickets (category-match@3 = 0.79) |
| OOD check | entropy + top-similarity gate flags inputs that fit no queue |

---

## Architecture

```
Raw ticket text
      │
      ▼
Fine-tuned MiniLM-L6-v2
(sentence-transformers/all-MiniLM-L6-v2 + classification head)
7-class sequence classifier · macro-F1 = 0.88
      │
      ▼
Temperature Scaling
ECE 0.072 → 0.015 after calibration
      │
      ▼
OOD check (entropy + FAISS similarity)
flags inputs that fit no queue
      │
      ▼
Confidence Gate (τ = 0.75)
    /           \
  ≥ τ           < τ
   │               │
   ▼               ▼
Auto-route      Escalate → human agent queue
   │
   ▼
Historical Ticket Retrieval
FAISS nearest-neighbour on MiniLM embeddings
category-match@3 = 0.79
   │
   ▼
Output (Streamlit dashboard / FastAPI JSON)
```

**Fine-tuning details:**
- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Classification head trained on 38,266 tickets (80/10/10 split, stratified)
- Weighted cross-entropy loss to handle class imbalance (Infrastructure 5× overrepresented)
- 5 epochs, AdamW lr=2e-5, early stopping on macro-F1
- Best val macro-F1: 0.8807

**Clustering** (offline, not in real-time path):
DBSCAN on ticket embeddings surfaces recurring issue clusters — potential automation candidates. Accessible via the "Recurring Issues" tab in the UI.

---

## Dataset & category mapping

**Source:** IT Service Ticket Classification Dataset (Kaggle) — 47,837 tickets,
columns `Document` (text) and `Topic_group` (category label).

**Category merge decision:**
The raw dataset has 8 labels. `Administrative rights` and `Access` were merged into
a single `Access Management` queue after sampling 30 tickets from each — both contain
account permission requests, SSO issues, and login failures routed to the same team.
No rows were dropped; the full 47,833 tickets are retained after cleaning.

| Raw label             | Mapped queue      | Rationale |
|-----------------------|-------------------|-----------|
| Hardware              | Infrastructure    | Physical/OS issues → same engineering queue |
| Administrative rights | Access Management | Account permission tickets — same queue as Access |
| Access                | Access Management | Login, SSO, permission requests |
| Storage               | Storage           | 1-to-1 |
| HR Support            | HR Support        | Onboarding, new starters |
| Purchase              | Procurement       | PO processing, equipment orders |
| Internal Project      | Internal Project  | Task management, pipeline setup |
| Miscellaneous         | General IT        | Server restarts, misc config |

---

## Results

### Fine-tuned MiniLM (production model)

Trained on 38,266 tickets with weighted cross-entropy. Evaluated on 4,784 held-out test tickets.

| Metric                              | Fine-tuned MiniLM |
|-------------------------------------|-------------------|
| Macro-F1                            | **0.88**          |
| Weighted F1                         | 0.87              |
| Accuracy                            | 0.87              |
| ECE after temperature scaling       | **0.015**         |
| Coverage at τ = 0.75                | **74.7%**         |
| Routing accuracy at τ = 0.75        | **0.942**         |
| Category-match@3 (historical retrieval) | **0.79**      |

### Per-class breakdown (fine-tuned MiniLM)

| Queue             | Precision | Recall | F1   | Test n |
|-------------------|-----------|--------|------|--------|
| Procurement       | 0.96      | 0.92   | **0.94** | 247 |
| Storage           | 0.89      | 0.90   | 0.89 | 277    |
| Access Management | 0.90      | 0.89   | 0.89 | 889    |
| HR Support        | 0.87      | 0.88   | 0.87 | 1,091  |
| Infrastructure    | 0.86      | 0.86   | 0.86 | 1,362  |
| Internal Project  | 0.86      | 0.85   | 0.86 | 212    |
| General IT        | 0.83      | 0.83   | 0.83 | 706    |

**Analysis:**
- **Procurement** F1 jumped from 0.93 → 0.94 after adding weighted loss — previously misrouted to Infrastructure due to class imbalance (1,362 Infrastructure samples vs 247 Procurement).
- **General IT** is the hardest queue (F1 0.83) — it is the "miscellaneous" bucket, so its vocabulary overlaps every other queue by construction.
- **Infrastructure ↔ Access Management** is the hardest pair: hardware setup for a new hire is genuinely ambiguous depending on how the submitter frames the request. This is real label ambiguity in the data.

### Coverage–accuracy tradeoff

At τ = 0.75: **74.7% of tickets auto-route at 94.2% routing precision**; the remaining 25.3% escalate to a human. Raise τ for higher precision at lower coverage; lower it for the reverse.

---

## Known limitations

**Out-of-distribution detection is heuristic, not a trained rejector.** A closed-world classifier forces every input into one of 7 queues. The system flags likely OOD inputs via two signals (`src/ood.py`): normalised prediction entropy above a threshold, and top retrieval similarity below a threshold. This is a guardrail, not a substitute for a trained OOD class; the thresholds live in `src/config.py`.

**Dataset text is pre-cleaned.** The Kaggle dataset has been partially anonymised (names replaced, ticket IDs stripped). Real-world F1 on raw helpdesk text is expected to be ~4–6 points lower.

**Historical retrieval, not resolution retrieval.** The FAISS index returns similar past tickets, not their resolutions. The dataset contains ticket descriptions only; category-match@3 = 0.79 measures how often retrieved tickets share the query's queue label.

---

## Quickstart

### 1. Dataset
Download from Kaggle: [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
Place as `data/tickets.csv`.

### 2. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train baseline
```bash
python train.py                      # hybrid — TF-IDF clf + MiniLM RAG
python train.py --embedding tfidf    # TF-IDF for both (no internet needed)
python train.py --embedding minilm   # MiniLM for both
```

### 4. Fine-tune transformer (Colab)
Open `finetune_colab.ipynb` in Google Colab (T4 GPU recommended, ~25 min).
Publishes the trained model to HuggingFace Hub automatically.

### 5. Evaluate
```bash
python evaluate.py      # classification report + 3 plots in plots/
```

### 6. Run
```bash
# Terminal 1
uvicorn app.main:app --reload

# Terminal 2
streamlit run app/streamlit_app.py
```
Or: `docker-compose up --build`

### 7. Monitor
```bash
curl "http://localhost:8000/monitor?window_hours=24"
```

### Deploy to HuggingFace Spaces / Streamlit Cloud
See [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Project structure

```
├── data/
│   └── tickets.csv              ← Kaggle dataset (not committed)
├── models/                      ← saved after train.py
│   ├── classifier.pkl           ← Logistic Regression (baseline)
│   ├── label_encoder.pkl
│   ├── temperature.pkl          ← T and τ
│   ├── tfidf.pkl                ← TF-IDF vectoriser
│   ├── faiss_index.bin          ← ~38k-vector index
│   ├── faiss_metadata.pkl       ← ticket text + labels
│   └── rag_embeddings.npy       ← indexed vectors (reused by cluster analysis)
├── plots/                       ← generated by evaluate.py
│   ├── reliability_diagram.png
│   ├── coverage_accuracy_curve.png
│   └── confusion_matrix.png
├── src/
│   ├── config.py                ← category map + all constants
│   ├── preprocess.py            ← text cleaning + label mapping
│   ├── embeddings.py            ← MiniLM wrapper with cache
│   ├── classifier.py            ← LR + temperature scaling
│   ├── rag.py                   ← FAISS index + retrieval
│   ├── cluster.py               ← DBSCAN automation detection
│   └── ood.py                   ← entropy + similarity OOD gate
├── app/
│   ├── main.py                  ← FastAPI (/predict /health /categories /set-threshold)
│   └── streamlit_app.py         ← Streamlit UI
├── tests/                       ← pytest suite (classifier, preprocess, ood, api)
├── finetune_colab.ipynb         ← end-to-end fine-tuning notebook (Colab)
├── spaces_app.py                ← Streamlit Cloud / HF Spaces entrypoint
├── train.py                     ← baseline pipeline
├── evaluate.py                  ← metrics + plots
├── robustness_test.py           ← failure-mode documentation
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## API endpoints

| Method | Path             | Purpose |
|--------|------------------|---------|
| `POST` | `/predict`       | Classify ticket → category, confidence, OOD flag, historical tickets |
| `POST` | `/feedback`      | Record agent correction |
| `GET`  | `/monitor`       | Rolling-window KPIs: escalation rate, OOD rate, correction rate |
| `GET`  | `/health`        | Liveness check — returns model info, temperature, threshold |
| `GET`  | `/categories`    | List of the 7 routing queues |
| `POST` | `/set-threshold` | Adjust τ at runtime |

OpenAPI docs: `http://localhost:8000/docs`

---

## Stack

| Component       | Library                    | Why |
|-----------------|----------------------------|-----|
| Classifier      | transformers (fine-tuned)  | MiniLM fine-tuned end-to-end; 0.88 macro-F1 |
| Embeddings      | sentence-transformers      | MiniLM-L6-v2 384-dim for FAISS retrieval |
| Calibration     | scipy.optimize             | Temperature scaling; ECE measurable and interpretable |
| Vector search   | faiss-cpu                  | Exact cosine search; industry-standard |
| Clustering      | scikit-learn (DBSCAN)      | No K needed; handles noise points |
| API             | FastAPI                    | Async; auto-generates OpenAPI docs |
| UI              | Streamlit                  | Demo-ready; plots embedded inline |
| Model hosting   | HuggingFace Hub            | Public model card + inference API |
| Containerisation| Docker + Compose           | Single command to run everything |
| CI              | GitHub Actions             | pytest on every push |
| Audit log       | JSONL (append-only)        | Every prediction + correction recorded; feeds /monitor |

