"""
preprocess.py
Text cleaning pipeline for IT support tickets.

All transformations are deterministic (no model calls) so this runs
fast over the full dataset. Keep it simple — the embedding model handles
most semantic understanding; over-cleaning hurts more than it helps.
"""

import re
import string
import pandas as pd
from src.config import (
    TEXT_COLUMN, LABEL_COLUMN, CATEGORY_MAP, CATEGORIES, DATA_DIR
)


# cleaning helpers

_HTML_TAG    = re.compile(r"<[^>]+>")
_URL         = re.compile(r"https?://\S+|www\.\S+")
_EMAIL       = re.compile(r"\S+@\S+")
_TICKET_ID   = re.compile(r"\b(INC|REQ|TKT|CHG|RITM|IMP)\d+\b", re.I)
_WHITESPACE  = re.compile(r"\s+")
_NON_ASCII   = re.compile(r"[^\x00-\x7F]+")


def clean_text(text: str) -> str:
    """Apply lightweight normalisation. Returns a clean unicode string."""
    if not isinstance(text, str):
        return ""
    t = _HTML_TAG.sub(" ", text)         # strip HTML tags
    t = _URL.sub(" ", t)                 # remove URLs
    t = _EMAIL.sub(" ", t)              # remove email addresses
    t = _TICKET_ID.sub(" ", t)          # remove ticket IDs (INC0012345 etc.)
    t = _NON_ASCII.sub(" ", t)          # drop non-ASCII characters
    t = t.lower()                       # lowercase
    t = t.translate(                    # remove punctuation (keep space)
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    t = _WHITESPACE.sub(" ", t).strip()
    return t


# label mapping

def map_label(raw: str) -> str | None:
    """Map a raw Topic_group value → one of the 7 spec categories.

    Returns None for rows that cannot be mapped (they are dropped in
    load_dataset so the classifier only sees clean labels).
    """
    raw = str(raw).strip()
    # Direct lookup
    if raw in CATEGORY_MAP:
        return CATEGORY_MAP[raw]
    # Case-insensitive fallback
    raw_lower = raw.lower()
    for key, val in CATEGORY_MAP.items():
        if key.lower() == raw_lower:
            return val
    # Substring match (last resort)
    for key, val in CATEGORY_MAP.items():
        if key.lower() in raw_lower:
            return val
    return None


# dataset loader

def load_dataset(csv_path=None, verbose: bool = True) -> pd.DataFrame:
    """Load, clean, and label-map the raw CSV.

    Returns a DataFrame with columns:
        text  (cleaned ticket text)
        label (one of CATEGORIES)
        raw_text  (original, kept for RAG display)
        raw_label (original Topic_group, for debugging)

    Rows with missing text or unmapped labels are dropped.
    """
    if csv_path is None:
        csv_path = DATA_DIR / "tickets.csv"

    df = pd.read_csv(csv_path)

    if verbose:
        print(f"[preprocess] Loaded {len(df):,} rows from {csv_path.name}")
        print(f"[preprocess] Columns detected: {list(df.columns)}")

    # Validate expected columns exist
    missing = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing!r} not found. "
            f"Available: {list(df.columns)}. "
            f"Update TEXT_COLUMN / LABEL_COLUMN in src/config.py."
        )

    df = df.rename(columns={TEXT_COLUMN: "raw_text", LABEL_COLUMN: "raw_label"})
    df = df[["raw_text", "raw_label"]].copy()

    # Drop rows with null text
    before = len(df)
    df = df.dropna(subset=["raw_text"])
    if verbose and before != len(df):
        print(f"[preprocess] Dropped {before - len(df):,} rows with null text.")

    # Clean text
    df["text"] = df["raw_text"].apply(clean_text)

    # Drop empty text after cleaning
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    # Map labels
    df["label"] = df["raw_label"].apply(map_label)
    unmapped = df["label"].isna().sum()
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    if verbose:
        print(f"[preprocess] Dropped {unmapped:,} rows with unmapped labels.")
        print(f"[preprocess] Final dataset: {len(df):,} rows")
        print("\n[preprocess] Label distribution:")
        counts = df["label"].value_counts()
        for cat in CATEGORIES:
            n = counts.get(cat, 0)
            bar = "█" * (n // max(counts.max() // 40, 1))
            print(f"  {cat:20s} {n:6,}  {bar}")

    return df[["text", "label", "raw_text", "raw_label"]]


if __name__ == "__main__":
    df = load_dataset(verbose=True)
    print(f"\nSample rows:\n{df[['text', 'label']].head(3).to_string()}")
