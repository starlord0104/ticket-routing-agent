# Deployment Guide

## Option 1 — Local (Docker Compose)

```bash
# 1. Build image + start API + Streamlit
docker-compose up --build

# API  → http://localhost:8000
# UI   → http://localhost:8501
# Docs → http://localhost:8000/docs
```

---

## Option 2 — HuggingFace Spaces (public live demo)

HuggingFace Spaces runs a Streamlit app from a Git repo.
The models (~100 MB) live in the Space repo and are loaded at startup.

### Step-by-step

#### 1. Create the Space

Go to https://huggingface.co/new-space and pick:
- **SDK:** Streamlit  
- **Visibility:** Public (or Private for a restricted demo)

#### 2. Clone the Space repo

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>
```

#### 3. Copy project files into the Space repo

```bash
cp -r /path/to/ticket-routing-agent/. .
```

The Space root must have:
- `requirements.txt`  ← already exists
- `app/streamlit_app.py`  ← entry point
- A `README.md` with HF frontmatter (see below)

#### 4. Add HuggingFace frontmatter to README.md

Paste this block at the **very top** of `README.md`, replacing the first line:

```yaml
---
title: IT Ticket Router
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app/streamlit_app.py
pinned: false
---
```

#### 5. Train locally and commit the models

HuggingFace Spaces has no GPU or long build time for training.
Train locally first, then commit the model artefacts:

```bash
# In your local project directory:
python train.py               # produces models/ directory

# Back in the Space repo:
cp -r /path/to/ticket-routing-agent/models ./models
```

HuggingFace uses **Git LFS** for large files (`.bin`, `.pkl`, `.npy`).
LFS is auto-enabled on Spaces — just push normally.

#### 6. Push

```bash
git add .
git commit -m "deploy: hybrid model + Streamlit UI"
git push
```

The Space builds automatically. Watch the logs on the Space page.

#### 7. Verify

- Open `https://huggingface.co/spaces/<your-username>/<space-name>`
- Enter a ticket in the text box and click "Route Ticket"
- Check that the API sidebar connects: the app must point at the bundled API

> **Note:** The Streamlit app calls `http://localhost:8000` by default.  
> On HF Spaces, the FastAPI backend runs in the same process space only if you
> launch it from `app/streamlit_app.py` using `subprocess` or `threading`.  
> The simplest fix: run the API inline inside the Streamlit app at startup,
> or switch the Streamlit app to call the `/predict` logic directly.

---

## Option 3 — Fly.io / Railway / Render (full stack)

These platforms can run `docker-compose.yml` directly.

```bash
# Render: connect the GitHub repo, set build command to:
docker build -t ticket-router .
# Start command:
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Store model artefacts in a persistent disk or a private HuggingFace repo
and download them at startup with `huggingface_hub.snapshot_download()`.

---

## Environment variables

| Variable           | Default        | Purpose |
|--------------------|----------------|---------|
| `PYTHONIOENCODING` | `utf-8`        | Set in Dockerfile; prevents Windows cp1252 crash |
| `PYTHONUTF8`       | `1`            | Belt-and-suspenders UTF-8 on all platforms |
| `API_URL`          | `http://localhost:8000` | Streamlit app reads this to find the FastAPI backend |

---

## Health check

```bash
curl http://localhost:8000/health
# → {"status":"ok","embedding_mode":"hybrid","rag_loaded":true,...}
```

## Monitor endpoint

```bash
curl "http://localhost:8000/monitor?window_hours=24"
# → {"total_predictions":142,"escalation_rate":0.23,"ood_rate":0.04,
#    "correction_rate":0.02,"alert":false,"alert_reasons":[],"window_hours":24}
```
