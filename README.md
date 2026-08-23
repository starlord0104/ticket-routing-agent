# Confidence-Aware IT Ticket Routing & Escalation System

Routes IT support tickets to 7 operational queues using calibrated confidence scores.
Tickets below a tunable threshold are escalated to human agents rather than auto-routed.

---

## What it does

Classifies free-text IT tickets across 7 queues (Infrastructure, Access Management,
Storage, HR Support, Procurement, Internal Project, General IT) using sentence embeddings
and a logistic classifier. Raw softmax scores are calibrated via temperature scaling so
that a 0.9 confidence prediction is correct ~90% of the time. A threshold τ gates
auto-routing: tickets below τ are escalated instead of being silently misrouted.

This is a **confidence-aware routing system**, not an autonomous agent. It classifies
and routes. The only autonomous decision it makes is whether confidence is high enough
to act without a human.

---

## End-to-end example

**Ticket submitted:**
> "User Sarah Johnson cannot log into Confluence or Jira after returning from
> maternity leave. IT says her account was suspended during her absence."

| Step | What happens |
|------|-------------|
| Preprocessing | lowercase · strip HTML · remove ticket IDs |
| Vectorisation | MiniLM-L6-v2 → 384-dim dense (shipped)  ·  TF-IDF 8k baseline available |
| Classification | Logistic Regression → **Access Management** |
| Calibration | Temperature T=0.87 → calibrated confidence gates the decision |
| Gate | confidence ≥ τ=0.75 → **auto-route**; below τ → **escalate** |
| Retrieval | FAISS returns 3 most similar historical tickets (category-match@3 = 0.79) |
| OOD check | entropy + top-similarity gate flags inputs that fit no queue |

*(Values above are illustrative of the flow. Reproduce the exact metrics with
`python evaluate.py`; the measured figures are in [Results](#results).)*

---

## Architecture

```
Raw ticket text
      │
      ▼
Preprocessing          lowercase · strip HTML · remove noise
      │
      ▼
Vectorisation          TF-IDF 8k (baseline)  OR  MiniLM-L6-v2 384-dim
      │
      ▼
Logistic Classifier    7 queues · macro-F1 reported per embedding type
      │
      ▼
Temperature Scaling    fitted on val set · ECE reported before & after
      │
      ▼
OOD check              entropy + top-similarity gate flags no-queue inputs
      │
      ▼
Confidence Gate (τ)
    /           \
  ≥ τ           < τ
   │               │
   ▼               ▼
Auto-route      Escalate → human agent queue
   │
   ▼
Historical Ticket Retrieval
FAISS nearest-neighbour · category-match@3 = 0.79
   │
   ▼
Output (Streamlit dashboard / FastAPI JSON)
```

**Clustering** (offline, not in real-time path):
DBSCAN on ticket embeddings surfaces recurring issue clusters — potential
automation candidates. Accessible via the "Recurring Issues" tab in the UI.

---

## Dataset & category mapping

**Source:** IT Service Ticket Classification Dataset (Kaggle) — 47,837 tickets,
columns `Document` (text) and `Topic_group` (category label).

**Category merge decision:**
The raw dataset has 8 labels. `Administrative rights` and `Access` were merged into
a single `Access Management` queue after sampling 30 tickets from each — both contain
account permission requests, SSO issues, and login failures routed to the same team.
No rows were dropped; the full 47,833 tickets are retained after cleaning.

| Raw label            | Mapped queue      | Rationale |
|----------------------|-------------------|-----------|
| Hardware             | Infrastructure    | Physical/OS issues → same engineering queue |
| Administrative rights| Access Management | Account permission tickets — same queue as Access |
| Access               | Access Management | Login, SSO, permission requests |
| Storage              | Storage           | 1-to-1 |
| HR Support           | HR Support        | Onboarding, new starters |
| Purchase             | Procurement       | PO processing, equipment orders |
| Internal Project     | Internal Project  | Task management, pipeline setup |
| Miscellaneous        | General IT        | Server restarts, misc config |

---

## Results

### Shipped model — MiniLM + Logistic Regression (measured)

These are the numbers produced by `python evaluate.py` on the trained artifact in
this repo (9,567 held-out test tickets). MiniLM-L6-v2 embeddings + a logistic head is
the model that ships; a TF-IDF baseline is also wired in the code for comparison.

| Metric                              | MiniLM + LR (shipped) |
|-------------------------------------|-----------------------|
| Macro-F1 (9,567 test tickets)       | **0.81**              |
| Weighted F1                         | 0.79                  |
| Accuracy                            | 0.79                  |
| ECE before calibration              | 0.0274                |
| ECE after temperature scaling       | **0.0122** (−55%)     |
| Temperature T                       | 0.866                 |
| Coverage at τ = 0.75                | **64.8%**             |
| Routing accuracy at τ = 0.75        | **0.926**             |
| Escalated tickets at τ = 0.75       | 3,365 / 9,567 (35.2%) |
| Category-match@3 (historical retrieval) | **0.79**          |

### Per-class breakdown (MiniLM + LR)

| Queue            | Precision | Recall | F1   | Test n |
|------------------|-----------|--------|------|--------|
| Procurement      | 0.94      | 0.86   | 0.90 | 493    |
| Access Management| 0.83      | 0.80   | 0.82 | 1,777  |
| Storage          | 0.86      | 0.77   | 0.81 | 555    |
| Internal Project | 0.84      | 0.77   | 0.80 | 424    |
| HR Support       | 0.80      | 0.80   | 0.80 | 2,183  |
| Infrastructure   | 0.75      | 0.81   | 0.78 | 2,723  |
| General IT       | 0.74      | 0.74   | 0.74 | 1,412  |

**Analysis:**
- **Procurement** has the highest precision (0.94) despite being one of the smaller
  classes — PO numbers, invoice, and asset-register vocabulary are highly distinctive.
- **General IT** is the weakest queue (F1 0.74) — it is the "miscellaneous" bucket, so
  its vocabulary overlaps with every other queue by construction.
- **Infrastructure ↔ Access Management** is the hardest pair: hardware setup for a new
  user straddles both queues depending on how the submitter frames the request. This is
  genuine label ambiguity in the data, not a model failure.

### Coverage–accuracy tradeoff

At τ = 0.75: **64.8% of tickets auto-route at 92.6% routing precision**; the remaining
35.2% escalate to a human. The coverage-accuracy curve (saved to `plots/`) shows the
full tradeoff — raise τ for higher precision at lower coverage, lower it for the reverse.

---

## Known limitations

**Out-of-distribution detection is heuristic, not a trained rejector.** A closed-world
classifier forces every input into one of 7 queues. The system now flags likely OOD
inputs via two signals (`src/ood.py`): normalised prediction entropy above a threshold,
and top retrieval similarity below a threshold — so a vegetarian meal request is flagged
rather than silently routed. This is a guardrail, not a substitute for a trained OOD
class; the thresholds live in `src/config.py` and should be tuned on real OOD traffic.

**Dataset text is pre-cleaned.** The Kaggle dataset has already been partially
anonymised (names replaced, ticket IDs stripped). Real-world F1 on raw helpdesk text
is expected to be ~4–6 points lower. Robustness tests document how the model
behaves on noisy, short, and ambiguous inputs.

**Historical retrieval, not resolution retrieval.** The FAISS index returns similar
past tickets, not their resolutions. The dataset contains ticket descriptions only,
no resolution field. Category-match@3 = 0.70 measures how often retrieved tickets
share the query's queue label.

**MiniLM is the trained artefact in this repo.** The saved classifier expects 384-dim
MiniLM embeddings, and the FAISS index is built from them. Reproducing from scratch
requires running `train.py` on a machine with internet access (first run downloads
~80MB from HuggingFace to build the embedding cache). The TF-IDF path is wired in the
code as a baseline but is not the shipped artifact.

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

### 3. Inspect categories
```bash
python -m src.config    # prints category mapping; add any unmapped labels
```

### 4. Train
```bash
python train.py         # ~4 min first run (embedding cache); instant after
```

### 5. Evaluate
```bash
python evaluate.py      # classification report + 3 plots in plots/
```

### 6. Robustness tests
```bash
python robustness_test.py   # 5 challenging tickets — documents failure modes
```

### 6b. Unit + API tests
```bash
pytest -q                   # classifier, preprocessing, OOD gate, /predict contract
```

### 7. Run
```bash
# Terminal 1
uvicorn app.main:app --reload

# Terminal 2
streamlit run app/streamlit_app.py
```
Or: `docker-compose up --build`

---

## Project structure

```
├── data/
│   └── tickets.csv              ← Kaggle dataset
├── models/                      ← saved after train.py
│   ├── classifier.pkl           ← Logistic Regression
│   ├── label_encoder.pkl
│   ├── temperature.pkl          ← T and τ
│   ├── tfidf.pkl                ← TF-IDF vectoriser
│   ├── svd.pkl                  ← 128-dim reduction for TF-IDF baseline
│   ├── faiss_index.bin          ← ~38k-vector index (train+val)
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
│   └── streamlit_app.py         ← Streamlit UI (route / metrics / clusters)
├── tests/                       ← pytest suite (classifier, preprocess, ood, api)
├── train.py                     ← end-to-end pipeline
├── evaluate.py                  ← metrics + plots
├── robustness_test.py           ← 5 failure-mode tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Stack

| Component       | Library                 | Why |
|-----------------|-------------------------|-----|
| Embeddings      | sentence-transformers   | MiniLM-L6-v2 is fast, 384-dim, and widely known |
| Classifier      | scikit-learn            | LR trains in seconds; calibration is clean to apply |
| Calibration     | scipy.optimize          | Temperature scaling in ~30 lines; ECE is measurable |
| Vector search   | faiss-cpu               | Exact cosine search; industry-standard |
| Clustering      | scikit-learn (DBSCAN)   | No K needed; handles noise points |
| API             | FastAPI                 | Async; auto-generates OpenAPI docs |
| UI              | Streamlit               | Demo-ready; plots embedded inline |
| Containerisation| Docker + Compose        | Single command to run everything |

---

## Interview answers

**"Why logistic regression over a fine-tuned transformer?"**
A logistic head on MiniLM embeddings gives macro-F1 = 0.81 and trains in seconds once
embeddings are cached. Fine-tuning the transformer itself would add at most a couple of
points at many times the compute — not worth it on this class count and dataset size.
A TF-IDF baseline is also in the repo; the decision is evidence-based.

**"What does your confidence score mean?"**
After temperature scaling (T=0.866), the confidence tracks observed accuracy far more
closely — ECE drops from 0.027 to 0.012 (−55%) on the validation set. The reliability
diagram shows the calibration curve before and after. Without calibration, the raw
softmax is overconfident and the threshold becomes meaningless.

**"How did you choose τ?"**
By sweeping 0.5 to 0.99 and plotting coverage vs. routing accuracy. At τ=0.75: 64.8%
coverage at 92.6% precision. The curve is in `plots/`. The right τ depends on whether
you optimise for throughput or accuracy — I picked 0.75 as a starting point, not a
universal answer.

**"What's the hardest category pair?"**
Infrastructure and Access Management. A ticket about hardware setup for a new hire is
genuinely ambiguous — it could go to either queue. The model's confusion here reflects
real label ambiguity in the training data.

**"Why not call it an agent?"**
It doesn't plan, use tools, or pursue goals across steps. It classifies a ticket,
estimates confidence, and routes or escalates. "Confidence-aware routing system" is
accurate. Calling it an agent would be a claim the architecture doesn't support.

**"What would you add in production?"**
Heuristic OOD detection is already in (`src/ood.py`); the next steps are a *trained*
OOD class for out-of-domain inputs, prediction-distribution monitoring for confidence
drift, and a retraining trigger based on escalation-rate increase. The cluster analysis
tab already surfaces recurring issues as automation candidates — that is the production
monitoring seed.
