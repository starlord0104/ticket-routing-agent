"""
upload_models.py
────────────────
Upload trained model artefacts to a HuggingFace repository so the live
demo (Streamlit Community Cloud) can download them at startup.

Usage — upload to a model repo (free, no compute):
    python upload_models.py --repo starlord0104/ticket-routing-models

Usage — upload to a Space repo (paid HF plan):
    python upload_models.py --repo starlord0104/ticket-routing-agent --repo-type space

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login        # or set HF_TOKEN env var

What gets uploaded (from models/ and plots/):
    classifier.pkl      label_encoder.pkl   temperature.pkl
    embedding_mode.pkl  tfidf.pkl           faiss_index.bin
    faiss_metadata.pkl  rag_embeddings.npy
    reliability_diagram.png  coverage_accuracy_curve.png  confusion_matrix.png
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
    p.add_argument("--repo", required=True,
                   help="HuggingFace repo id, e.g. starlord0104/ticket-routing-models")
    p.add_argument("--repo-type", default="model",
                   choices=["model", "dataset", "space"],
                   help="HuggingFace repo type (default: model)")
    args = p.parse_args()

    api     = HfApi()
    repo    = args.repo
    rtype   = args.repo_type

    # Create the repo if it doesn't exist yet
    api.create_repo(repo_id=repo, repo_type=rtype, exist_ok=True)
    print(f"\n[upload] Target ({rtype}): https://huggingface.co/{rtype}s/{repo}\n")

    uploaded, skipped = 0, 0

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
    hf_url = f"https://huggingface.co/{rtype}s/{repo}"
    print(f"   Repo: {hf_url}\n")
    if rtype in ("model", "dataset"):
        print(
            "   Next: set HF_MODEL_REPO environment variable in Streamlit Community Cloud:\n"
            f"   HF_MODEL_REPO = {repo}\n"
        )


if __name__ == "__main__":
    main()
