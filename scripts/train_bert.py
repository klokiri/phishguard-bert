"""
Phase 4b: fine-tune Kasim's own BERT model on the training split.

Run:
  .venv/bin/python scripts/train_bert.py --benchmark       time a few steps, write nothing
  .venv/bin/python scripts/train_bert.py                   full training (resumes if interrupted)

What "fine-tuning" means here: start from bert-base-uncased, a general English
language model (110M parameters) that knows nothing about phishing, add a
2-class output layer, and train the whole network on our labelled training
split.

Choices and why:
  - bert-base-uncased (not bert-large like the author): ~3x fewer parameters,
    so training fits the time available on an Apple M4 Pro.
  - Learning rate 2e-5, batch size 16, weight decay 0.01: the author's values
    (standard for BERT fine-tuning).
  - max_length: chosen from results/token_length_stats.json (see --max-length).
  - group_by_length: batches contain texts of similar length, so short URLs
    are not padded up to max_length. Big speed-up on this dataset.
  - The validation set (from train only) is evaluated every 1,000 steps; the
    checkpoint with the best validation F1 is kept, and training stops early
    if F1 does not improve for 3 evaluations in a row (~1 epoch). The test
    set is NOT touched until training is finished.
  - Checkpoints are saved every 1,000 steps to checkpoints/ (git-ignored), so
    an interrupted run continues where it stopped when the command is re-run.
    Run it under `caffeinate -i` with the Mac plugged in and the lid open:
    macOS sleeps on lid close and the training pauses.
  - Seed 42 everywhere for reproducibility.

Outputs (full run):
  models/phishguard-bert-base/            final model (git-ignored)
  results/bert_base_history.json          loss/metrics per logging step and epoch
  results/bert_base.json (+ predictions)  test-set evaluation via src/evaluate.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil  # installed with accelerate; reports this process's memory use
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import evaluate, load_test, print_summary  # noqa: E402
from src.models.bert import BertClassifier  # noqa: E402

BASE_MODEL = "bert-base-uncased"
SEED = 42
SPLITS = REPO_ROOT / "data" / "splits"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "phishguard-bert-base"
FINAL_DIR = REPO_ROOT / "models" / "phishguard-bert-base"
ID2LABEL = {0: "benign", 1: "phishing"}
EVAL_EVERY_STEPS = 1000
# Early stopping: stop if validation F1 has not improved for 3 evaluations
# in a row (3,000 steps, roughly one epoch of 3,479 steps).
EARLY_STOPPING_PATIENCE = 3


# Pad each batch up to a multiple of 64 tokens (64, 128, ..., 512). Why: on
# Apple's MPS backend every new tensor shape gets its own cached GPU buffers,
# and with free-length padding there are hundreds of shapes. The first
# restarted run grew to 22 GB and swapped. With multiples of 64 there are at
# most 8 shapes; the extra padding is masked out, so results are unaffected.
PAD_MULTIPLE = 64
MEMORY_CHECK_EVERY = 50


def make_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=PAD_MULTIPLE)


class MPSMemoryCallback(TrainerCallback):
    """Every 50 steps: release cached MPS memory and log memory use.

    The log line makes a memory leak visible within minutes instead of hours.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % MEMORY_CHECK_EVERY or not torch.backends.mps.is_available():
            return
        torch.mps.empty_cache()
        gpu_gb = torch.mps.driver_allocated_memory() / 1e9
        rss_gb = psutil.Process().memory_info().rss / 1e9
        print(f"[memory] step {state.global_step}: MPS driver {gpu_gb:.1f} GB, process RSS {rss_gb:.1f} GB",
              flush=True)


class TokenizedDataset(torch.utils.data.Dataset):
    """Holds pre-tokenised texts and labels; the Trainer reads items from it."""

    def __init__(self, encodings, labels):
        self.input_ids = encodings["input_ids"]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # Padding happens later, per batch, in DataCollatorWithPadding.
        return {"input_ids": self.input_ids[i], "labels": int(self.labels[i])}


def make_dataset(tokenizer, df, max_length):
    enc = tokenizer(list(df["text"]), truncation=True, max_length=max_length)
    return TokenizedDataset(enc, df["label"])


def compute_metrics(eval_pred):
    """Validation metrics after each epoch (binary, phishing = positive)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def build_args(args, output_dir, max_steps=-1):
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
        weight_decay=0.01,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        train_sampling_strategy="group_by_length",  # transformers 5.x name for group_by_length
        # Validate and checkpoint every 1,000 steps (~1/3.5 of an epoch) rather
        # than once per epoch: the first run was lost when the Mac slept at
        # step 407 with no checkpoint saved. Now an interruption costs at most
        # 1,000 steps, and re-running the command resumes.
        eval_strategy="steps",
        eval_steps=EVAL_EVERY_STEPS,
        save_strategy="steps",
        save_steps=EVAL_EVERY_STEPS,
        save_total_limit=2,  # keeps the latest and the best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
        dataloader_num_workers=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--benchmark", action="store_true", help="time a few training steps, write nothing")
    parser.add_argument("--benchmark-steps", type=int, default=60)
    args = parser.parse_args()

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    )
    train_df = pd.read_parquet(SPLITS / "train.parquet")
    val_df = pd.read_parquet(SPLITS / "val.parquet")

    if args.benchmark:
        # A random sample keeps the mix of URLs, texts and HTML representative.
        n_steps = args.benchmark_steps
        sample = train_df.sample(n_steps * 16, random_state=SEED)
        ds = make_dataset(tokenizer, sample, args.max_length)
        bench_args = build_args(args, REPO_ROOT / "checkpoints" / "benchmark", max_steps=n_steps)
        bench_args.eval_strategy = "no"
        bench_args.save_strategy = "no"
        bench_args.load_best_model_at_end = False
        trainer = Trainer(model=model, args=bench_args, train_dataset=ds,
                          data_collator=make_collator(tokenizer), callbacks=[MPSMemoryCallback()])
        t0 = time.perf_counter()
        out = trainer.train()
        sec_per_step = (time.perf_counter() - t0) / n_steps
        steps_per_epoch = -(-len(train_df) // 16)
        print(f"\nBENCHMARK max_length={args.max_length} precision={args.precision}: "
              f"{sec_per_step:.3f} s/step, loss {out.training_loss:.4f}")
        print(f"  -> about {sec_per_step * steps_per_epoch / 60:.0f} min per epoch "
              f"({steps_per_epoch} steps), plus validation")
        return

    train_ds = make_dataset(tokenizer, train_df, args.max_length)
    val_ds = make_dataset(tokenizer, val_df, args.max_length)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=build_args(args, CHECKPOINT_DIR),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=make_collator(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE), MPSMemoryCallback()],
    )
    last = get_last_checkpoint(str(CHECKPOINT_DIR))
    if last:
        print(f"Resuming from {last}")
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=last)
    train_seconds = time.perf_counter() - t0

    # Save the best model (load_best_model_at_end=True already loaded it).
    trainer.save_model(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    history = {
        "base_model": BASE_MODEL,
        "settings": {"max_length": args.max_length, "epochs_max": args.epochs, "precision": args.precision,
                     "learning_rate": 2e-5, "batch_size": 16, "weight_decay": 0.01, "seed": SEED,
                     "early_stopping": f"validation F1 every {EVAL_EVERY_STEPS} steps, patience {EARLY_STOPPING_PATIENCE} evaluations", "group_by_length": True},
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_f1": trainer.state.best_metric,
        "train_seconds_this_session": round(train_seconds, 1),
        "resumed_from": last,
        "log_history": trainer.state.log_history,
        "command": f".venv/bin/python scripts/train_bert.py --max-length {args.max_length} "
                   f"--epochs {args.epochs} --precision {args.precision}",
    }
    (REPO_ROOT / "results" / "bert_base_history.json").write_text(json.dumps(history, indent=2) + "\n")

    # Final, one-time evaluation on the test set through the shared evaluator.
    del trainer, model
    clf = BertClassifier(str(FINAL_DIR), half=True, max_length=args.max_length)
    test = load_test()
    t0 = time.perf_counter()
    scores = clf.predict_proba(test["text"], batch_size=32)
    elapsed = time.perf_counter() - t0
    result = evaluate(
        "bert_base", test, scores, latency_seconds_total=elapsed,
        details={
            "description": f"{BASE_MODEL} fine-tuned by us on data/splits/train.parquet",
            "training": history["settings"],
            "best_validation_f1": history["best_validation_f1"],
            "train_seconds": history["train_seconds_this_session"],
            "inference": f"device {clf.device.type}, float16={clf.half}, batch 32, "
                         f"truncation to {args.max_length} tokens",
            "saved_model": "models/phishguard-bert-base/ (git-ignored; re-create with the command)",
            "command": history["command"],
        },
    )
    print_summary(result)


if __name__ == "__main__":
    main()
