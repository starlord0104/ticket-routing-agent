"""
rag.py
FAISS index over resolved tickets for resolution retrieval.

This is NOT generation — it's retrieval-only, which means every suggestion
is a real resolution that worked before. That makes it:
  • Traceable  (link back to source ticket)
  • Explainable (no hallucination; it only shows real past fixes)
  • Evaluable  (precision@k is a clean metric)

Architecture:
  1. At training time: index all training-set tickets with their resolutions.
  2. At inference time: embed new ticket → nearest-neighbour search → top-K results.

Cosine similarity is approximated via inner product on L2-normalised vectors
(which is what encode() returns), so we use faiss.IndexFlatIP.
"""

import numpy as np
import faiss
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import MODELS_DIR, TOP_K


_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
_META_PATH  = MODELS_DIR / "faiss_metadata.pkl"


# building the index

def build_index(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    index_path: Path = _INDEX_PATH,
    meta_path:  Path = _META_PATH,
) -> faiss.IndexFlatIP:
    """Build and save a FAISS flat inner-product index.

    Args:
        embeddings: (N, 384) float32, L2-normalised.
        metadata:   DataFrame with columns [text, label, raw_text, raw_label].
                    Row i corresponds to embeddings[i].
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # exact search, cosine via inner product
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    # Store metadata so we can display resolution text at query time
    joblib.dump(metadata.reset_index(drop=True), meta_path)

    print(f"[rag] Built FAISS index: {index.ntotal:,} vectors, dim={dim}")
    return index


def load_index(
    index_path: Path = _INDEX_PATH,
    meta_path:  Path = _META_PATH,
) -> tuple[faiss.IndexFlatIP, pd.DataFrame]:
    index    = faiss.read_index(str(index_path))
    metadata = joblib.load(meta_path)
    return index, metadata


# retrieval

class ResolutionRetriever:
    """Retrieves the top-K most similar past tickets for a query."""

    def __init__(
        self,
        index:    faiss.IndexFlatIP,
        metadata: pd.DataFrame,
        k:        int = TOP_K,
    ):
        self.index    = index
        self.metadata = metadata
        self.k        = k

    def retrieve(
        self,
        query_embedding: np.ndarray,   # shape (384,) or (1, 384)
        filter_label: Optional[str] = None,
    ) -> list[dict]:
        """Return top-K similar tickets.

        Args:
            query_embedding: Single ticket embedding, L2-normalised.
            filter_label:    If set, only return tickets of this category.
                             Useful when you already know the predicted class.

        Returns list of dicts:
          {
            rank:       int,    # 1-indexed
            similarity: float,  # cosine similarity (0-1)
            category:   str,
            text:       str,    # cleaned ticket text (preview)
            raw_text:   str,    # original ticket text
          }
        """
        q = query_embedding.reshape(1, -1).astype(np.float32)

        # Retrieve more than k if filtering, to ensure we find enough
        fetch_k = self.k * 5 if filter_label else self.k
        fetch_k = min(fetch_k, self.index.ntotal)

        similarities, indices = self.index.search(q, fetch_k)
        sims = similarities[0]
        idxs = indices[0]

        results = []
        for sim, idx in zip(sims, idxs):
            if idx < 0:
                continue
            row = self.metadata.iloc[idx]
            if filter_label and row["label"] != filter_label:
                continue
            results.append({
                "rank":       len(results) + 1,
                "similarity": float(sim),
                "category":   str(row["label"]),
                "text":       str(row["text"])[:300],      # preview
                "raw_text":   str(row["raw_text"])[:500],  # full original
            })
            if len(results) >= self.k:
                break

        return results

    @classmethod
    def load(cls, k: int = TOP_K) -> "ResolutionRetriever":
        index, metadata = load_index()
        return cls(index=index, metadata=metadata, k=k)

    def embed_query(self, text: str) -> np.ndarray:
        """Convert raw text → RAG embedding using saved TF-IDF + SVD pipeline."""
        import joblib
        from sklearn.preprocessing import normalize
        tfidf = joblib.load(MODELS_DIR / "tfidf.pkl")
        svd   = joblib.load(MODELS_DIR / "svd.pkl")
        x_sp  = tfidf.transform([text])
        x_svd = svd.transform(x_sp).astype(np.float32)
        return normalize(x_svd, norm="l2")[0]


# evaluation

def category_match_at_k(
    retriever:        ResolutionRetriever,
    query_embeddings: np.ndarray,
    query_labels:     list[str],
    k:                int = TOP_K,
) -> float:
    """Compute category-match@k: fraction of top-k retrievals that share the
    query ticket's queue label.

    This is NOT a standard IR metric like Precision@k or MRR, which require
    human relevance judgements. It is a proxy — two tickets in the same queue
    are assumed to be relevant to each other. Label it accurately when reporting.

    For each query:
      hits    = retrieved tickets whose label == gold label
      score   = hits / k
    Returns the mean score across all queries.
    """
    scores = []
    for emb, gold_label in zip(query_embeddings, query_labels):
        retrieved = retriever.retrieve(emb, filter_label=None)[:k]
        correct   = sum(1 for h in retrieved if h["category"] == gold_label)
        scores.append(correct / k)
    cm_at_k = float(np.mean(scores))
    print(f"[rag] Category-match@{k}: {cm_at_k:.4f}  "
          f"(label: category-match, not precision — no relevance annotations)")
    return cm_at_k
