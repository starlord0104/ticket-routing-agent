"""
upload_models.py
────────────────
Upload trained model artefacts to HuggingFace Spaces so the live demo
can load them at startup.

Usage:
    python upload_models.py --space starlord0104/ticket-routing-agent

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login        # or set HF_TOKEN env var

What gets uploaded (from models/):
    classifier.pkl      label_encoder.pkl   temperature.pkl
    embedding_mode.pkl  tfidf.pkl           faiss_index.bin
    faiss_metadata.pkl  rag_embeddings.npy

And plots/ (for the System Metrics tab):
    reliability_diagram.png   coverage_accuracy_curve.png   confusion_matrix.png
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


MODELS_DIR = Path(__file__).parent / "models"
PLOTS_DIR  = Path(__file__).parent / "plots"

MODEL_FILES = [
    "classifier.pkl",
    "label_encoder.pkl",
    "temperature.pkl",
    "embedding_mode.pkl",
    "tfidf.pkl",
    "faiss_index.bin",
    "faiss_metadata.pkl",
    "rag_embeddings.npy",
]

PLOT_FILES = [
    "reliability_diagram.png",
    "coverage_accuracy_curve.png",
    "confusion_matrix.png",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--space", required=True,
        help="HuggingFace Space repo id, e.g. starlord0104/ticket-routing-agent",
    )
    args = p.parse_args()

    api    = HfApi()
    repo   = args.space
    rtype  = "space"

    uploaded, skipped = 0, 0

    print(f"\n[upload] Target: https://huggingface.co/spaces/{repo}\n")

    for fname in MODEL_FILES:
        local = MODELS_DIR / fname
        if not local.exists():
            print(f"  SKIP  models/{fname}  (not found — run train.py first)")
            skipped += 1
            continue
        size_mb = local.stat().st_size / 1024 / 1024
        print(f"  UP    models/{fname}  ({size_mb:.1f} MB) …", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"models/{fname}",
            repo_id=repo,
            repo_type=rtype,
        )
        print("done")
        uploaded += 1

    for fname in PLOT_FILES:
        local = PLOTS_DIR / fname
        if not local.exists():
            print(f"  SKIP  plots/{fname}  (run evaluate.py to generate)")
            skipped += 1
            continue
        print(f"  UP    plots/{fname} …", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"plots/{fname}",
            repo_id=repo,
            repo_type=rtype,
        )
        print("done")
        uploaded += 1

    print(f"\n✓  Uploaded {uploaded} files, skipped {skipped}.")
    print(f"   Space: https://huggingface.co/spaces/{repo}\n")


if __name__ == "__main__":
    main()
