"""
finetune_colab.py
─────────────────
Fine-tune all-MiniLM-L6-v2 as a 7-class IT ticket classifier.
Run this in Google Colab (Runtime → T4 GPU).

Open directly:
  https://colab.research.google.com/github/starlord0104/ticket-routing-agent/blob/master/finetune_colab.py

Or paste cell-by-cell into a new Colab notebook.
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install
# ════════════════════════════════════════════════════════════════════════════
# @title Install dependencies
# !pip install -q transformers datasets scikit-learn huggingface_hub \
#              pandas numpy joblib accelerate evaluate

# ════════════════════════════════════════════════════════════════════════════
# CELL 2 — Mount Drive and upload tickets.csv
# ════════════════════════════════════════════════════════════════════════════
# @title Mount Google Drive (tickets.csv must be at MyDrive/tickets.csv)
"""
from google.colab import drive
drive.mount('/content/drive')

import shutil, os
os.makedirs('/content/data', exist_ok=True)
shutil.copy('/content/drive/MyDrive/tickets.csv', '/content/data/tickets.csv')
print("✓ tickets.csv copied")
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 3 — Preprocess
# ════════════════════════════════════════════════════════════════════════════
# @title Load and preprocess tickets
import pandas as pd
import numpy as np
from pathlib import Path

# ── category map (mirrors src/preprocess.py) ─────────────────────────────
CATEGORY_MAP = {
    "Hardware":               "Infrastructure",
    "Administrative rights":  "Access Management",
    "Access":                 "Access Management",
    "Storage":                "Storage",
    "HR Support":             "HR Support",
    "Purchase":               "Procurement",
    "Internal Project":       "Internal Project",
    "Miscellaneous":          "General IT",
}
CATEGORIES = sorted(set(CATEGORY_MAP.values()))
LABEL2ID   = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL   = {i: c for c, i in LABEL2ID.items()}

def load_data(csv_path: str = "/content/data/tickets.csv"):
    df = pd.read_csv(csv_path)
    # normalise column names
    df.columns = [c.strip() for c in df.columns]
    text_col  = next(c for c in df.columns if "document" in c.lower() or "text" in c.lower())
    label_col = next(c for c in df.columns if "topic" in c.lower() or "label" in c.lower())
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["raw_label"].map(CATEGORY_MAP)
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"].str.len() > 5]
    df["label_id"] = df["label"].map(LABEL2ID)
    print(f"  {len(df):,} tickets  ·  {df['label'].nunique()} classes")
    print(df["label"].value_counts().to_string())
    return df

# df = load_data()   # uncomment when running in Colab


# ════════════════════════════════════════════════════════════════════════════
# CELL 4 — Build HuggingFace Dataset + train/val/test split
# ════════════════════════════════════════════════════════════════════════════
# @title Split into train / val / test (80 / 10 / 10)
from sklearn.model_selection import train_test_split

def make_splits(df, seed=42):
    train_df, tmp = train_test_split(df, test_size=0.20, stratify=df["label_id"], random_state=seed)
    val_df, test_df = train_test_split(tmp, test_size=0.50, stratify=tmp["label_id"], random_state=seed)
    print(f"  train {len(train_df):,}  val {len(val_df):,}  test {len(test_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# train_df, val_df, test_df = make_splits(df)


# ════════════════════════════════════════════════════════════════════════════
# CELL 5 — Tokenise
# ════════════════════════════════════════════════════════════════════════════
# @title Tokenise with MiniLM tokenizer
from transformers import AutoTokenizer
from datasets import Dataset

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def tokenise_split(df, tokenizer, max_length=128):
    ds = Dataset.from_dict({"text": df["text"].tolist(), "labels": df["label_id"].tolist()})
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    return ds.map(tok, batched=True, batch_size=512)

# tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)
# train_ds   = tokenise_split(train_df, tokenizer)
# val_ds     = tokenise_split(val_df,   tokenizer)
# test_ds    = tokenise_split(test_df,  tokenizer)


# ════════════════════════════════════════════════════════════════════════════
# CELL 6 — Fine-tune
# ════════════════════════════════════════════════════════════════════════════
# @title Fine-tune all-MiniLM-L6-v2 for sequence classification
import evaluate as hf_evaluate
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

def make_compute_metrics(metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels, average="macro")
    return compute_metrics

def fine_tune(train_ds, val_ds, tokenizer, output_dir="/content/finetuned_minilm"):
    metric = hf_evaluate.load("f1")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=True,                    # T4 supports fp16
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Training …")
    trainer.train()
    print(f"\nBest val macro-F1: {trainer.state.best_metric:.4f}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")
    return trainer

# trainer = fine_tune(train_ds, val_ds, tokenizer)


# ════════════════════════════════════════════════════════════════════════════
# CELL 7 — Evaluate on held-out test set
# ════════════════════════════════════════════════════════════════════════════
# @title Full classification report on test set
from sklearn.metrics import classification_report

def evaluate_model(trainer, test_ds):
    preds_out = trainer.predict(test_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    labels    = preds_out.label_ids
    print(classification_report(labels, preds, target_names=CATEGORIES))

# evaluate_model(trainer, test_ds)


# ════════════════════════════════════════════════════════════════════════════
# CELL 8 — Upload to HuggingFace
# ════════════════════════════════════════════════════════════════════════════
# @title Push fine-tuned model to HuggingFace Hub
"""
Steps:
  1. Log in:   !huggingface-cli login
  2. Run this cell

The model will be at:
  https://huggingface.co/starlord0104/ticket-routing-minilm-finetuned
"""

HF_FINETUNED_REPO = "starlord0104/ticket-routing-minilm-finetuned"

def push_to_hub(output_dir="/content/finetuned_minilm"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model     = AutoModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.push_to_hub(HF_FINETUNED_REPO, private=False)
    tokenizer.push_to_hub(HF_FINETUNED_REPO, private=False)
    print(f"\n✓ Model live at https://huggingface.co/{HF_FINETUNED_REPO}")

# !huggingface-cli login
# push_to_hub()


# ════════════════════════════════════════════════════════════════════════════
# CELL 9 — Quick sanity check (run after uploading)
# ════════════════════════════════════════════════════════════════════════════
# @title Sanity-check the uploaded model
"""
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="starlord0104/ticket-routing-minilm-finetuned",
    top_k=3,
)

tests = [
    "I need to purchase a new laptop, mine is 5 years old.",
    "My laptop screen is black and won't turn on.",
    "New hire starting Monday — set up email and system access.",
    "I am locked out of my admin account.",
    "Drive is full, cannot save any files.",
]

for t in tests:
    top = clf(t)[0]
    print(f"{top['label']:20s}  {top['score']:.2%}  |  {t[:60]}")
"""
