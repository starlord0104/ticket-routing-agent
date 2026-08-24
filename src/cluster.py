"""
cluster.py
DBSCAN clustering over embeddings to detect repeated issues.

Use case: when the same underlying problem causes many tickets (e.g. VPN
outage after a patch), the escalated tickets cluster together. Surfacing
these clusters tells ops teams "automate this" before it floods the queue.

Why DBSCAN (not K-Means):
  • No need to pre-specify K
  • Handles noise points (singleton tickets that truly are one-offs)
  • Works with cosine distance on L2-normalised embeddings
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.config import (
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES, AUTOMATION_FLAG_MIN_SIZE
)


def cluster_tickets(
    embeddings: np.ndarray,
    texts:      list[str],
    labels:     list[str] | None = None,
    eps:        float = DBSCAN_EPS,
    min_samples: int  = DBSCAN_MIN_SAMPLES,
) -> pd.DataFrame:
    """Run DBSCAN and return a DataFrame with cluster assignments.

    DBSCAN uses cosine distance: distance = 1 - cosine_similarity.
    Since our embeddings are L2-normalised, cosine_sim = inner_product,
    so cosine_dist = 1 - inner_product.

    Args:
        embeddings:  (N, 384) float32, L2-normalised.
        texts:       Ticket text strings (for display).
        labels:      Predicted category labels (optional, for context).
        eps:         Max cosine distance to be in the same neighbourhood.
        min_samples: Min points to form a core point.

    Returns DataFrame with columns:
        text, label, cluster_id, is_noise
    """
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        n_jobs=-1,
    )
    cluster_ids = db.fit_predict(embeddings)

    df = pd.DataFrame({
        "text":       texts,
        "label":      labels if labels is not None else ["unknown"] * len(texts),
        "cluster_id": cluster_ids,
    })
    df["is_noise"] = df["cluster_id"] == -1

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise    = (cluster_ids == -1).sum()
    print(f"[cluster] Found {n_clusters} clusters, {n_noise} noise points "
          f"out of {len(texts)} tickets.")
    return df


def get_automation_candidates(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Return clusters large enough to be flagged for automation.

    A cluster of >= AUTOMATION_FLAG_MIN_SIZE tickets on the same topic
    suggests a recurring, automatable issue.

    Returns a summary DataFrame sorted by cluster size (descending).
    """
    non_noise = cluster_df[~cluster_df["is_noise"]].copy()
    if non_noise.empty:
        print("[cluster] No non-noise clusters found.")
        return pd.DataFrame()

    summary = (
        non_noise.groupby("cluster_id")
        .agg(
            size       = ("text", "count"),
            top_label  = ("label", lambda x: x.mode()[0]),
            sample_text= ("text", "first"),
        )
        .reset_index()
        .sort_values("size", ascending=False)
    )

    flagged = summary[summary["size"] >= AUTOMATION_FLAG_MIN_SIZE].copy()
    flagged["automation_flag"] = True

    print(f"[cluster] {len(flagged)} clusters flagged for automation "
          f"(size >= {AUTOMATION_FLAG_MIN_SIZE}).")
    return flagged


def describe_clusters(cluster_df: pd.DataFrame, top_n: int = 5) -> None:
    """Print a human-readable summary of the largest clusters."""
    flagged = get_automation_candidates(cluster_df)
    if flagged.empty:
        return
    print("\n[cluster] automation candidates:")
    for _, row in flagged.head(top_n).iterrows():
        print(f"\nCluster {int(row['cluster_id'])}  |  "
              f"Size: {int(row['size'])}  |  "
              f"Top category: {row['top_label']}")
        print(f"  Sample: {row['sample_text'][:120]} …")
    print("")
