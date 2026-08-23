"""
embeddings.py
─────────────
Thin wrapper around sentence-transformers. Handles batching, caching,
and exposes a single encode() function used throughout the project.

Why MiniLM-L6-v2:
  • 384-dim output — fast to train a logistic head on
  • Strong MTEB benchmark performance on classification tasks
  • ~80 MB download — manageable in a student environment
  • Well-known name that reads well on a resume
"""

import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, MAX_SEQ_LENGTH, MODELS_DIR


_CACHE_PATH = MODELS_DIR / "embeddings_cache.pkl"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"[embeddings] Loading {EMBEDDING_MODEL} …")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        _model.max_seq_length = MAX_SEQ_LENGTH
    return _model


def encode(
    texts: list[str],
    batch_size: int = 128,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Encode a list of strings → L2-normalised embedding matrix.

    Args:
        texts:         List of cleaned ticket strings.
        batch_size:    How many to encode at once. 128 is safe on CPU.
        show_progress: tqdm bar during encoding.
        normalize:     L2-normalise vectors (required for FAISS cosine search).

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)


def encode_with_cache(
    texts: list[str],
    cache_path: Path = _CACHE_PATH,
    force_recompute: bool = False,
    **kwargs,
) -> np.ndarray:
    """Encode, but load from disk cache if it exists.

    Embedding the full 47k dataset takes ~3-4 min on CPU the first time;
    subsequent runs are instant. Call with force_recompute=True after
    changing the dataset or cleaning pipeline.
    """
    if cache_path.exists() and not force_recompute:
        print(f"[embeddings] Loading cached embeddings from {cache_path.name}")
        return joblib.load(cache_path)

    print("[embeddings] Computing embeddings (first time — takes ~4 min on CPU) …")
    embs = encode(texts, **kwargs)
    joblib.dump(embs, cache_path)
    print(f"[embeddings] Saved to {cache_path}")
    return embs


if __name__ == "__main__":
    # Quick smoke test
    sample = [
        "VPN keeps disconnecting after Windows update",
        "Cannot access the SQL Server database from application",
        "Request for new user account creation in Active Directory",
    ]
    vecs = encode(sample, show_progress=False)
    print(f"Shape: {vecs.shape}")       # (3, 384)
    print(f"Norms: {np.linalg.norm(vecs, axis=1)}")  # all ≈ 1.0
