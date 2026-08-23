"""
train.py
────────
End-to-end training pipeline.

Usage:
    python train.py                          # hybrid (default — recommended)
    python train.py --embedding tfidf        # TF-IDF classifier + SVD retrieval
    python train.py --embedding minilm       # MiniLM for both
    python train.py --force-recompute        # re-embed even if cache exists
    python train.py --threshold 0.80         # override default τ

Embedding modes
───────────────
  hybrid  — TF-IDF classifier (macro-F1 0.86, trains in seconds) +
             MiniLM FAISS index (category-match@3 0.79, better semantic retrieval).
             Best of both: right tool for each task.  ← DEFAULT

  tfidf   — TF-IDF for both classifier and FAISS (via TruncatedSVD 128-dim).
             No internet required. Good baseline.

  minilm  — MiniLM-L6-v2 for both. Weaker classifier (0.81 F1) but consistent
             embeddings throughout. First run downloads ~80 MB from HuggingFace.

Steps:
  1. Load + clean dataset             (preprocess.py)
  2. Encode tickets                   (per mode above)
  3. Split train / val / test
  4. Train Logistic Regression        (classifier.py)
  5. Fit temperature scaling on val   (classifier.py)
  6. Save TicketRouter to models/
  7. Build FAISS index on train set   (rag.py)
"""

import argparse
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Windows consoles default to cp1252 — force UTF-8 for the box-drawing banners.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config     import TEST_SIZE, VAL_SIZE, RANDOM_STATE, DEFAULT_THRESHOLD, MODELS_DIR
from src.preprocess import load_dataset
from src.classifier import train_classifier, fit_temperature, TicketRouter
from src.rag        import build_index


def parse_args():
    p = argparse.ArgumentParser(description="Train the ticket routing model.")
    p.add_argument(
        "--embedding", choices=["hybrid", "tfidf", "minilm"], default="hybrid",
        help=(
            "hybrid (default): TF-IDF classifier + MiniLM retrieval. "
            "tfidf: TF-IDF for both. "
            "minilm: MiniLM for both."
        ),
    )
    p.add_argument("--force-recompute", action="store_true",
                   help="Re-embed even if MiniLM cache exists.")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Confidence threshold τ (default {DEFAULT_THRESHOLD}).")
    return p.parse_args()


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _fit_tfidf(texts: list[str]):
    """Fit TF-IDF vectoriser. Returns (X_sparse, vectoriser)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("[train] Fitting TF-IDF (max_features=8 000, bigrams, sublinear_tf) …")
    vec = TfidfVectorizer(max_features=8_000, ngram_range=(1, 2), sublinear_tf=True)
    X   = vec.fit_transform(texts)
    joblib.dump(vec, MODELS_DIR / "tfidf.pkl")
    print(f"[train] TF-IDF: {X.shape}  nnz={X.nnz:,}")
    return X, vec


def _fit_svd(X_sparse):
    """Reduce sparse TF-IDF to dense 128-dim for FAISS. Returns (X_dense, svd)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    print("[train] Fitting TruncatedSVD (128 components) for FAISS …")
    svd    = TruncatedSVD(n_components=128, random_state=RANDOM_STATE)
    X_dense = normalize(svd.fit_transform(X_sparse).astype(np.float32), norm="l2")
    joblib.dump(svd, MODELS_DIR / "svd.pkl")
    print(f"[train] SVD dense: {X_dense.shape}")
    return X_dense, svd


def _load_minilm(texts: list[str], force: bool = False):
    """Return MiniLM embeddings (uses disk cache after first run)."""
    from src.embeddings import encode_with_cache
    return encode_with_cache(texts, force_recompute=force)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("\n━━  STEP 1 / 5  ─  Loading dataset  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df     = load_dataset(verbose=True)
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # ── 2. Encode ─────────────────────────────────────────────────────────────
    # X      → classifier input  (sparse for tfidf/hybrid, dense for minilm)
    # X_rag  → FAISS input       (always dense float32, L2-normalised)
    print(f"\n━━  STEP 2 / 5  ─  Encoding [{args.embedding}]  ━━━━━━━━━━━━━━━━━━━━━━━")

    if args.embedding == "minilm":
        X     = _load_minilm(texts, force=args.force_recompute)
        X_rag = X       # same dense embeddings for classifier and FAISS

    elif args.embedding == "tfidf":
        X, _     = _fit_tfidf(texts)         # sparse for classifier
        X_rag, _ = _fit_svd(X)               # dense 128-dim for FAISS

    else:  # hybrid — TF-IDF classifier + MiniLM retrieval
        print("[train] Hybrid mode: TF-IDF for classifier, MiniLM for FAISS retrieval.")
        X, _  = _fit_tfidf(texts)            # sparse for classifier
        X_rag = _load_minilm(texts, force=args.force_recompute)   # dense 384-dim for FAISS
        print(f"[train] Classifier input: {X.shape} (sparse TF-IDF)")
        print(f"[train] FAISS input:      {X_rag.shape} (dense MiniLM)")

    # ── 3. Encode labels + split ──────────────────────────────────────────────
    print("\n━━  STEP 3 / 5  ─  Splitting data  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"[train] Classes: {list(le.classes_)}")

    # Split X (classifier) and X_rag (FAISS) with the same indices.
    (X_trainval, X_test,
     X_rag_trainval, X_rag_test,
     y_trainval, y_test) = train_test_split(
        X, X_rag, y,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    (X_train, X_val,
     _,       X_rag_val,
     y_train, y_val) = train_test_split(
        X_trainval, X_rag_trainval, y_trainval,
        test_size=val_frac, stratify=y_trainval, random_state=RANDOM_STATE,
    )
    print(f"[train] Train: {X_train.shape[0]:,}  "
          f"Val: {X_val.shape[0]:,}  "
          f"Test: {X_test.shape[0]:,}")

    joblib.dump({
        "X_test":        X_test,       # classifier features
        "X_test_rag":    X_rag_test,   # FAISS-ready dense features
        "X_val":         X_val,
        "X_val_rag":     X_rag_val,
        "y_test":        y_test,
        "y_val":         y_val,
        "label_encoder": le,
        "embedding_mode": args.embedding,
    }, MODELS_DIR / "splits.pkl")
    joblib.dump({"mode": args.embedding}, MODELS_DIR / "embedding_mode.pkl")
    print(f"[train] Saved splits → models/splits.pkl  (mode={args.embedding})")

    # ── 4. Train classifier ───────────────────────────────────────────────────
    print("\n━━  STEP 4 / 5  ─  Training classifier  ━━━━━━━━━━━━━━━━━━━━━━━━━")
    clf = train_classifier(X_train, y_train)

    # ── 5. Calibrate ──────────────────────────────────────────────────────────
    print("\n━━  STEP 5 / 5  ─  Calibrating confidence scores  ━━━━━━━━━━━━━━━━")
    temperature = fit_temperature(clf, X_val, y_val)

    router = TicketRouter(
        clf=clf, label_encoder=le,
        temperature=temperature, threshold=args.threshold,
    )
    router.save()

    # ── Build FAISS index over train+val set ──────────────────────────────────
    print("\n━━  Building RAG index  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    all_indices  = np.arange(len(texts))
    trainval_idx, _ = train_test_split(
        all_indices, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    rag_embeddings = X_rag_trainval
    build_index(
        embeddings=rag_embeddings,
        metadata=df.iloc[trainval_idx].reset_index(drop=True),
    )
    np.save(MODELS_DIR / "rag_embeddings.npy", rag_embeddings)
    print(f"[train] Saved RAG embeddings → models/rag_embeddings.npy {rag_embeddings.shape}")

    print(f"\n✓  Training complete [{args.embedding}].  Run  python evaluate.py  for metrics.\n")


if __name__ == "__main__":
    main()
