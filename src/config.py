from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR  = BASE_DIR / "plots"

for _d in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset columns ─────────────────────────────────────────────────────────
# These match the Kaggle IT Service Ticket Classification Dataset.
# If you use a different CSV, change these two strings.
TEXT_COLUMN  = "Document"
LABEL_COLUMN = "Topic_group"

# ── Category mapping ────────────────────────────────────────────────────────
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

# The 7 routing targets — derived from the actual data, not the spec skeleton.
# Interview note: "The dataset had 8 raw categories; I merged two overlapping
# access-related labels into one queue after sampling ticket text, giving 7
# targets with the full 47k rows retained."
CATEGORIES: list[str] = [
    "Infrastructure",
    "Access Management",
    "Storage",
    "HR Support",
    "Procurement",
    "Internal Project",
    "General IT",
]

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, fast, strong on short text
MAX_SEQ_LENGTH   = 256

# ── Training ────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
VAL_SIZE     = 0.1   # carved from train split, used for calibration
RANDOM_STATE = 42

# ── Calibration ─────────────────────────────────────────────────────────────
TEMPERATURE_INIT = 1.5   # starting value for scipy optimiser

# ── Confidence gate ──────────────────────────────────────────────────────────
# This is the τ in the diagram. Tune it via the coverage-accuracy curve;
# a good starting point is 0.75. The sweep in evaluate.py will show the
# full tradeoff so you can justify whatever you pick.
DEFAULT_THRESHOLD = 0.75

# ── RAG ─────────────────────────────────────────────────────────────────────
TOP_K = 3   # number of similar tickets to retrieve

# ── Out-of-distribution (OOD) detection ──────────────────────────────────────
# A closed-world classifier forces every input into one of the 7 queues.
# These two signals flag inputs that likely don't belong to any queue so the
# API can warn instead of silently routing (e.g. a lunch order → Infrastructure).
#   • Normalised prediction entropy above OOD_ENTROPY_THRESHOLD → distribution
#     is near-uniform, the model is guessing.
#   • Top retrieval similarity below OOD_SIMILARITY_THRESHOLD → nothing in the
#     historical corpus looks like this ticket.
OOD_ENTROPY_THRESHOLD    = 0.80
OOD_SIMILARITY_THRESHOLD = 0.45

# ── Clustering ───────────────────────────────────────────────────────────────
DBSCAN_EPS               = 0.4    # cosine-distance neighbourhood radius
DBSCAN_MIN_SAMPLES       = 3
AUTOMATION_FLAG_MIN_SIZE = 5      # flag cluster if it has >= this many tickets


# ── Quick sanity check ───────────────────────────────────────────────────────
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
