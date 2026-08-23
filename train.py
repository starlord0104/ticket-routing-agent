"""
train.py
────────
End-to-end training pipeline. Run once after placing tickets.csv in data/.

Usage:
    python train.py
    python train.py --force-recompute    # re-embed even if cache exists
    python train.py --threshold 0.80     # override default tau

What this script does (in order):
  1. Load + clean dataset             (preprocess.py)
  2. Encode tickets via MiniLM        (embeddings.py)
  3. Split train / val / test
  4. Train Logistic Regression        (classifier.py)
  5. Fit temperature scaling on val   (classifier.py)
  6. Save TicketRouter to models/
  7. Build FAISS index on train set   (rag.py)
  8. Print a quick training summary

After this, run:
    python evaluate.py    ← full metrics, plots, calibration diagram
    streamlit run app/streamlit_app.py
"""

import argparse
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Windows consoles default to cp1252 and crash on the box-drawing characters
# used in the step banners. Force UTF-8 output where the stream supports it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.preprocessing import LabelEncoder

from src.config    import TEST_SIZE, VAL_SIZE, RANDOM_STATE, DEFAULT_THRESHOLD
from src.preprocess import load_dataset
from src.embeddings import encode_with_cache
from src.classifier import train_classifier, fit_temperature, TicketRouter
from src.rag        import build_index


def parse_args():
    p = argparse.ArgumentParser(description="Train the ticket routing model.")
    p.add_argument("--force-recompute", action="store_true",
                   help="Re-embed even if cache exists")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Confidence threshold τ (default {DEFAULT_THRESHOLD})")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\n━━  STEP 1 / 5  ─  Loading dataset  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    df = load_dataset(verbose=True)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # ── 2. Encode ────────────────────────────────────────────────────────────
    print("\n━━  STEP 2 / 5  ─  Encoding tickets  ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    X = encode_with_cache(texts, force_recompute=args.force_recompute)

    # ── 3. Encode labels + split ──────────────────────────────────────────────
    print("\n━━  STEP 3 / 5  ─  Splitting data  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"[train] Classes: {list(le.classes_)}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Val slice carved from train (for calibration only, not for test)
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    print(f"[train] Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Test: {len(X_test):,}")

    # Save split indices for evaluate.py to use
    import joblib, os
    from src.config import MODELS_DIR
    joblib.dump({
        "X_test": X_test, "y_test": y_test,
        "X_val":  X_val,  "y_val":  y_val,
        "label_encoder": le,
    }, MODELS_DIR / "splits.pkl")
    print(f"[train] Saved splits to models/splits.pkl")

    # ── 4. Train classifier ────────────────────────────────────────────────────
    print("\n━━  STEP 4 / 5  ─  Training classifier  ━━━━━━━━━━━━━━━━━━━━━━━━━")
    clf = train_classifier(X_train, y_train)

    # ── 5. Calibrate ────────────────────────────────────────────────────────────
    print("\n━━  STEP 5 / 5  ─  Calibrating confidence scores  ━━━━━━━━━━━━━━━━")
    temperature = fit_temperature(clf, X_val, y_val)

    # ── Save ─────────────────────────────────────────────────────────────────────
    router = TicketRouter(
        clf=clf,
        label_encoder=le,
        temperature=temperature,
        threshold=args.threshold,
    )
    router.save()

    # ── Build FAISS index over training set ───────────────────────────────────────
    print("\n━━  Building RAG index  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # Use train + val (not test — don't leak)
    train_val_mask = np.ones(len(X), dtype=bool)
    # Get the actual training data rows from df
    n_trainval = len(X_trainval)
    # Rebuild to get the df slice aligned with X_trainval
    df_trainval = df.iloc[:n_trainval].reset_index(drop=True)
    # Actually, let's use a cleaner approach:
    # X_trainval corresponds to the first (1-TEST_SIZE) fraction after stratified split
    # We save the indices to avoid misalignment
    all_indices   = np.arange(len(X))
    trainval_idx, test_idx = train_test_split(
        all_indices, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    rag_embeddings = X[trainval_idx]
    build_index(
        embeddings=rag_embeddings,
        metadata=df.iloc[trainval_idx].reset_index(drop=True),
    )
    # Persist the indexed vectors so offline analyses (e.g. the DBSCAN
    # "Recurring Issues" tab) can reuse them without re-embedding.
    np.save(MODELS_DIR / "rag_embeddings.npy", rag_embeddings)
    print(f"[train] Saved RAG embeddings → models/rag_embeddings.npy "
          f"({rag_embeddings.shape})")

    print("\n✓  Training complete. Run  python evaluate.py  for metrics.\n")


if __name__ == "__main__":
    main()
