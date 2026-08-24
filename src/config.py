from pathlib import Path

# paths
BASE_DIR       = Path(__file__).parent.parent
DATA_DIR       = BASE_DIR / "data"
MODELS_DIR     = BASE_DIR / "models"
PLOTS_DIR      = BASE_DIR / "plots"
AUDIT_LOG_PATH = BASE_DIR / "logs" / "audit.jsonl"   # append-only prediction audit log

for _d in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
# logs/ is created on first write by audit.py — no need to mkdir here.

# dataset columns
# These match the Kaggle IT Service Ticket Classification Dataset.
# If you use a different CSV, change these two strings.
TEXT_COLUMN  = "Document"
LABEL_COLUMN = "Topic_group"

# category mapping
# Derived from EDA of the actual dataset (all_tickets_processed_improved_v3.csv).
# Raw labels → 7 routing targets. Zero rows dropped.
#
# Key decisions made from sampling ticket text:
#   Hardware            → Infrastructure   (physical/OS hardware issues)
#   Administrative rights → Access Management (Windows upgrades, Outlook,
#                           hardware swaps — same queue as account access)
#   Access              → Access Management (account permissions, logins)
#   Storage             → Storage          (1-to-1)
#   HR Support          → HR Support       (onboarding, new starters)
#   Purchase            → Procurement      (equipment orders, POs)
#   Internal Project    → Internal Project (task management, pipelines)
#   Miscellaneous       → General IT       (server restarts, misc configs)
CATEGORY_MAP: dict[str, str] = {
    "Hardware":             "Infrastructure",
    "Administrative rights":"Access Management",
    "Access":               "Access Management",
    "Storage":              "Storage",
    "HR Support":           "HR Support",
    "Purchase":             "Procurement",
    "Internal Project":     "Internal Project",
    "Miscellaneous":        "General IT",
}

# The 7 routing targets (8 raw categories; two access-related labels merged into one queue).
CATEGORIES: list[str] = [
    "Infrastructure",
    "Access Management",
    "Storage",
    "HR Support",
    "Procurement",
    "Internal Project",
    "General IT",
]

# embedding model
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, fast, strong on short text
MAX_SEQ_LENGTH   = 256

# training
TEST_SIZE    = 0.2
VAL_SIZE     = 0.1   # carved from train split, used for calibration
RANDOM_STATE = 42

# calibration
TEMPERATURE_INIT = 1.5   # starting value for scipy optimiser

# confidence gate (τ) — tune via the coverage-accuracy curve in evaluate.py
DEFAULT_THRESHOLD = 0.75

# RAG
TOP_K = 3   # number of similar tickets to retrieve

# OOD detection — two signals flag inputs likely outside the 7 queues:
#   • high entropy (near-uniform distribution → model is guessing)
#   • low top-retrieval similarity (nothing in corpus matches)
OOD_ENTROPY_THRESHOLD    = 0.80
OOD_SIMILARITY_THRESHOLD = 0.45

# clustering
DBSCAN_EPS               = 0.4    # cosine-distance neighbourhood radius
DBSCAN_MIN_SAMPLES       = 3
AUTOMATION_FLAG_MIN_SIZE = 5      # flag cluster if it has >= this many tickets


# quick sanity check
if __name__ == "__main__":
    import pandas as pd, sys
    csv_path = DATA_DIR / "tickets.csv"
    if not csv_path.exists():
        print(f"[config] tickets.csv not found at {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"[config] Loaded {len(df):,} rows. Columns: {list(df.columns)}")
    print(f"[config] Unique {LABEL_COLUMN} values:\n")
    for v in sorted(df[LABEL_COLUMN].dropna().unique()):
        mapped = CATEGORY_MAP.get(str(v), "⚠️  UNMAPPED")
        print(f"  {v!r:30s} → {mapped}")
