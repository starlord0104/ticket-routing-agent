"""
train.py
────────
End-to-end training pipeline.

Usage:
    python train.py                         # MiniLM (default)
    python train.py --embedding tfidf       # TF-IDF baseline
    python train.py --force-recompute       # re-embed even if cache exists (minilm)
    python train.py --threshold 0.80        # override default τ

Steps:
  1. Load + clean dataset             (preprocess.py)
  2. Encode tickets                   (MiniLM via embeddings.py, or TF-IDF+SVD)
  3. Split train / val / test
  4. Train Logistic Regression        (classifier.py)
  5. Fit temperature scaling on val   (classifier.py)
  6. Save TicketRouter to models/
  7. Build FAISS index on train set   (rag.py)

After this, run:
    python evaluate.py    ← full metrics + plots
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

from src.config    import TEST_SIZE, VAL_SIZE, RANDOM_STATE, DEFAULT_THRESHOLD, MODELS_DIR
from src.preprocess import load_dataset
from src.classifier import train_classifier, fit_temperature, TicketRouter
from src.rag        import build_index


def parse_args():
    p = argparse.ArgumentParser(description="Train the ticket routing model.")
    p.add_argument(
        "--embedding", choices=["tfidf", "minilm"], default="minilm",
        help="Embedding backend: 'minilm' (MiniLM-L6-v2, default) "
             "or 'tfidf' (TF-IDF 8k + bigrams baseline).",
    )
    p.add_argument("--force-recompute", action="store_true",
                   help="Re-embed even if cache exists (minilm only).")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Confidence threshold τ (default {DEFAULT_THRESHOLD}).")
    return p.parse_args()


def _encode_tfidf(texts: list[str]):
    """Fit TF-IDF + SVD on the full corpus.

    Returns
    -------
    X_sparse : scipy sparse (N, 8000)  — used for the classifier
    X_dense  : np.ndarray  (N, 128)   — L2-normalised, used for FAISS
    tfidf    : fitted TfidfVectorizer
    svd      : fitted TruncatedSVD
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    print("[train] Fitting TF-IDF (max_features=8 000, unigrams+bigrams, sublinear_tf) …")
    tfidf = TfidfVectorizer(max_features=8_000, ngram_range=(1, 2), sublinear_tf=True)
    X_sparse = tfidf.fit_transform(texts)
    print(f"[train] TF-IDF matrix: {X_sparse.shape}  nnz={X_sparse.nnz:,}")

    print("[train] Fitting TruncatedSVD (128 components) for FAISS …")
    svd = TruncatedSVD(n_components=128, random_state=RANDOM_STATE)
    X_dense = normalize(svd.fit_transform(X_sparse).astype(np.float32), norm="l2")
    print(f"[train] SVD dense matrix: {X_dense.shape}")

    joblib.dump(tfidf, MODELS_DIR / "tfidf.pkl")
    joblib.dump(svd,   MODELS_DIR / "svd.pkl")
    return X_sparse, X_dense, tfidf, svd


def main():
    args = parse_args()

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("\n━━  STEP 1 / 5  ─  Loading dataset  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df     = load_dataset(verbose=True)
    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # ── 2. Encode ─────────────────────────────────────────────────────────────
    # X      → classifier input  (sparse for tfidf, dense float32 for minilm)
    # X_rag  → FAISS input       (dense float32, L2-normalised, always)
    print(f"\n━━  STEP 2 / 5  ─  Encoding tickets [{args.embedding}]  ━━━━━━━━━━━━━━━━━━")

    if args.embedding == "minilm":
        from src.embeddings import encode_with_cache
        X     = encode_with_cache(texts, force_recompute=args.force_recompute)
        X_rag = X          # same vectors for classifier and FAISS
    else:  # tfidf
        X, X_rag, _, _ = _encode_tfidf(texts)

    # ── 3. Encode labels + split ──────────────────────────────────────────────
    print("\n━━  STEP 3 / 5  ─  Splitting data  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"[train] Classes: {list(le.classes_)}")

    # Split X (classifier) and X_rag (FAISS) together so indices are aligned.
    (X_trainval, X_test,
     X_rag_trainval, X_rag_test,
     y_trainval, y_test) = train_test_split(
        X, X_rag, y,
        test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    (X_train, X_val,
     _, X_rag_val,
     y_train, y_val) = train_test_split(
        X_trainval, X_rag_trainval, y_trainval,
        test_size=val_frac, stratify=y_trainval, random_state=RANDOM_STATE,
    )

    print(f"[train] Train: {X_train.shape[0]:,}  "
          f"Val: {X_val.shape[0]:,}  "
          f"Test: {X_test.shape[0]:,}")

    # Save splits so evaluate.py can load pre-computed features.
    # X_test_rag is always dense float32 — evaluate.py feeds it to FAISS.
    joblib.dump({
        "X_test":     X_test,      # classifier features (sparse for tfidf)
        "X_test_rag": X_rag_test,  # FAISS-ready dense features
        "X_val":      X_val,
        "X_val_rag":  X_rag_val,
        "y_test":     y_test,
        "y_val":      y_val,
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

    # ── Save router ───────────────────────────────────────────────────────────
    router = TicketRouter(
        clf=clf, label_encoder=le,
        temperature=temperature, threshold=args.threshold,
    )
    router.save()

    # ── Build FAISS index over train+val set ──────────────────────────────────
    print("\n━━  Building RAG index  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # Re-derive trainval_idx using the same seed so df metadata aligns.
    all_indices  = np.arange(len(texts))
    trainval_idx, _ = train_test_split(
        all_indices, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    rag_embeddings = X_rag_trainval   # dense, L2-normalised
    build_index(
        embeddings=rag_embeddings,
        metadata=df.iloc[trainval_idx].reset_index(drop=True),
    )
    np.save(MODELS_DIR / "rag_embeddings.npy", rag_embeddings)
    print(f"[train] Saved RAG embeddings → models/rag_embeddings.npy {rag_embeddings.shape}")

    print(f"\n✓  Training complete [{args.embedding}].  Run  python evaluate.py  for metrics.\n")


if __name__ == "__main__":
    main()
