"""
evaluate.py
───────────
Comprehensive evaluation — run after train.py.

Produces:
  1. Classification report  (per-class precision / recall / F1)
  2. Confusion matrix        (saved to plots/)
  3. Reliability diagram     (calibration before & after — key for interviews)
  4. Coverage-accuracy curve (τ sweep — your main quantitative result)
  5. RAG precision@k
  6. Cluster analysis on val set

Usage:
    python evaluate.py
    python evaluate.py --threshold 0.80   # evaluate at a specific τ

Everything is printed and saved. Open plots/ after running.
"""

import argparse
import sys
import matplotlib
matplotlib.use("Agg")   # headless — no Tk/display needed for saving plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Windows consoles default to cp1252 and crash on the box-drawing characters
# used in the section banners. Force UTF-8 output where the stream supports it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

from src.config     import CATEGORIES, MODELS_DIR, PLOTS_DIR, DEFAULT_THRESHOLD
from src.classifier import TicketRouter, get_logits, predict_proba_calibrated
from src.rag        import ResolutionRetriever, category_match_at_k
from src.cluster    import cluster_tickets, describe_clusters
from src.embeddings import encode


plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family":     "DejaVu Sans",
})
_PALETTE = "#4C72B0"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def reliability_diagram(
    probs_raw: np.ndarray,
    probs_cal: np.ndarray,
    y_true:    np.ndarray,
    n_bins:    int = 10,
    save_path: Path = PLOTS_DIR / "reliability_diagram.png",
):
    """Plot calibration curves for raw vs temperature-scaled probabilities.

    The key interview visual: show that after calibration, a 0.9 confidence
    prediction is correct ~90% of the time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle("Reliability Diagram  (closer to diagonal = better calibration)",
                 fontsize=12, y=1.02)

    for ax, probs, title in zip(
        axes,
        [probs_raw, probs_cal],
        ["Before calibration (raw softmax)", f"After temperature scaling"],
    ):
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct     = (predictions == y_true)

        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

        # ECE
        total  = sum(bin_counts)
        ece    = sum(
            count * abs(acc - conf)
            for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
        ) / total

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.bar(bin_confs, bin_accs, width=0.07, alpha=0.6,
               color=_PALETTE, label=f"ECE = {ece:.4f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Reliability diagram saved → {save_path}")


def coverage_accuracy_curve(
    probs_cal: np.ndarray,
    y_true:    np.ndarray,
    save_path: Path = PLOTS_DIR / "coverage_accuracy_curve.png",
) -> float:
    """Sweep τ and plot (coverage, accuracy) — your core result.

    Pick the operating point and report:
    'At τ=X, Y% of tickets auto-route at Z% routing precision.'
    Returns the AUC of the coverage-accuracy curve.
    """
    taus = np.linspace(0.50, 0.99, 100)
    coverages, accuracies = [], []

    confs       = probs_cal.max(axis=1)
    predictions = probs_cal.argmax(axis=1)
    correct     = (predictions == y_true)

    for tau in taus:
        auto_mask = confs >= tau
        coverage  = auto_mask.mean()
        if coverage == 0:
            break
        accuracy = correct[auto_mask].mean()
        coverages.append(coverage)
        accuracies.append(accuracy)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(coverages, accuracies, lw=2, color=_PALETTE)
    ax.fill_between(coverages, accuracies, alpha=0.1, color=_PALETTE)

    # Annotate the DEFAULT_THRESHOLD operating point
    default_mask = confs >= DEFAULT_THRESHOLD
    if default_mask.any():
        def_cov = default_mask.mean()
        def_acc = correct[default_mask].mean()
        ax.scatter([def_cov], [def_acc], s=80, color="red", zorder=5)
        ax.annotate(
            f"τ={DEFAULT_THRESHOLD}  \ncov={def_cov:.2f}, acc={def_acc:.2f}",
            xy=(def_cov, def_acc), xytext=(def_cov - 0.15, def_acc - 0.07),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red",
        )

    ax.set_xlabel("Coverage  (fraction auto-routed)")
    ax.set_ylabel("Routing accuracy on auto-routed tickets")
    ax.set_title("Coverage–Accuracy Curve  (confidence threshold sweep)")
    ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.01)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Coverage-accuracy curve saved → {save_path}")

    # Return AUC. Coverage decreases as τ rises, so sort by coverage ascending
    # before integrating — otherwise np.trapz returns a negative area.
    if len(coverages) > 1:
        order = np.argsort(coverages)
        auc = float(np.trapz(np.asarray(accuracies)[order], np.asarray(coverages)[order]))
    else:
        auc = 0.0
    return auc


def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    classes:   list[str],
    save_path: Path = PLOTS_DIR / "confusion_matrix.png",
):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Normalised Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load model + splits
    router  = TicketRouter.load()
    router.set_threshold(args.threshold)
    splits  = joblib.load(MODELS_DIR / "splits.pkl")
    le      = splits["label_encoder"]
    X_test  = splits["X_test"]      # classifier features (sparse for tfidf)
    y_test  = splits["y_test"]
    X_val   = splits["X_val"]
    y_val   = splits["y_val"]
    # Dense features for FAISS retrieval eval — always float32, L2-normalised.
    # Falls back to X_test for MiniLM runs where X_test is already dense.
    X_test_rag = splits.get("X_test_rag", X_test)
    embedding_mode = splits.get("embedding_mode", "minilm")
    print(f"[evaluate] Embedding mode: {embedding_mode}")

    classes = list(le.classes_)

    print("\n━━  1 / 5  ─  Classification report  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    # Raw probs (no calibration — for comparison)
    from scipy.special import softmax as sp_softmax
    raw_probs_test = sp_softmax(
        router.clf.decision_function(X_test) / 1.0, axis=1
    )
    cal_probs_test = predict_proba_calibrated(
        router.clf, X_test, router.temperature
    )

    y_pred = cal_probs_test.argmax(axis=1)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(classification_report(y_test, y_pred, target_names=classes))
    print(f"Macro-F1: {macro_f1:.4f}")

    print("\n━━  2 / 5  ─  Confusion matrix  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    plot_confusion_matrix(y_test, y_pred, classes)

    print("\n━━  3 / 5  ─  Reliability diagram  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    raw_probs_val = sp_softmax(
        router.clf.decision_function(X_val) / 1.0, axis=1
    )
    cal_probs_val = predict_proba_calibrated(
        router.clf, X_val, router.temperature
    )
    reliability_diagram(raw_probs_val, cal_probs_val, y_val)

    print("\n━━  4 / 5  ─  Coverage-accuracy curve  ━━━━━━━━━━━━━━━━━━━━━━━━━━")
    auc = coverage_accuracy_curve(cal_probs_test, y_test)
    print(f"[evaluate] Coverage-accuracy AUC: {auc:.4f}")

    # Escalation stats at current threshold
    auto_mask = cal_probs_test.max(axis=1) >= args.threshold
    auto_acc  = (y_pred[auto_mask] == y_test[auto_mask]).mean() if auto_mask.any() else 0
    print(f"\n[evaluate] At τ={args.threshold}:")
    print(f"  Auto-routed : {auto_mask.sum():,} / {len(y_test):,} "
          f"({auto_mask.mean()*100:.1f}%)")
    print(f"  Routing accuracy on auto-routed: {auto_acc:.4f}")
    print(f"  Escalated   : {(~auto_mask).sum():,} tickets")

    print("\n━━  5 / 5  ─  Historical retrieval: category-match@k  ━━━━━━━━━━━━━━")
    print("[evaluate] Note: metric is category-match@k (proxy — same queue = assumed relevant).")
    print("[evaluate] Not standard IR Precision@k — no human relevance annotations exist.")
    try:
        retriever   = ResolutionRetriever.load()
        gold_labels = [classes[i] for i in y_test[:500]]   # sample for speed
        # Use dense RAG embeddings — FAISS needs float32 arrays, not sparse matrices.
        cm3 = category_match_at_k(retriever, X_test_rag[:500], gold_labels)
        print(f"[evaluate] Category-match@3: {cm3:.4f}")
    except FileNotFoundError:
        print("[evaluate] FAISS index not found — skipping retrieval eval.")

    print("\n━━  Done  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"\n  Macro-F1          : {macro_f1:.4f}")
    print(f"  Coverage @ τ={args.threshold} : {auto_mask.mean()*100:.1f}%")
    print(f"  Acc @ coverage    : {auto_acc:.4f}")
    print(f"\n  Plots saved to: {PLOTS_DIR}/\n")


if __name__ == "__main__":
    main()
