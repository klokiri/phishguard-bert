"""
Write the COMPLETE training configuration of the fine-tuned BERT model to a
committed JSON file, for reproducibility.

Run:  .venv/bin/python scripts/export_training_config.py
Writes results/bert_base_training_config.json.

Why a separate script: results/bert_base_history.json lists the main settings,
but not every Trainer argument (optimiser, learning-rate schedule, warm-up,
etc.). The Trainer saves all of them in models/phishguard-bert-base/
training_args.bin; this script reads that file, so the record comes from what
was actually used, not from what we think was used.
"""

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_bert  # noqa: E402  (constants used by the run: padding, eval interval, patience)

MODEL_DIR = REPO_ROOT / "models" / "phishguard-bert-base"
LIBS = ["torch", "transformers", "accelerate", "tokenizers", "scikit-learn", "numpy", "pandas"]


def main() -> None:
    # weights_only=False is needed to unpickle a TrainingArguments object.
    # Safe here: the file was written by our own training run, on this machine.
    training_args = torch.load(MODEL_DIR / "training_args.bin", weights_only=False)
    model_config = json.loads((MODEL_DIR / "config.json").read_text())
    manifest = json.loads((REPO_ROOT / "results" / "split_manifest.json").read_text())

    report = {
        "model": "bert_base",
        "base_model": train_bert.BASE_MODEL,
        "training_arguments": json.loads(json.dumps(training_args.to_dict(), default=str)),
        "script_constants": {
            "pad_to_multiple_of": train_bert.PAD_MULTIPLE,
            "eval_and_save_every_steps": train_bert.EVAL_EVERY_STEPS,
            "early_stopping_patience_evaluations": train_bert.EARLY_STOPPING_PATIENCE,
            "mps_cache_cleared_every_steps": train_bert.MEMORY_CHECK_EVERY,
            "seed": train_bert.SEED,
        },
        "model_config": {k: model_config.get(k) for k in
                         ["architectures", "id2label", "label2id", "num_hidden_layers", "hidden_size",
                          "max_position_embeddings", "vocab_size"]},
        "data": {
            "dataset_revision": manifest["source"]["revision"],
            "train_content_sha256": manifest["splits"]["train"]["content_sha256"],
            "val_content_sha256": manifest["splits"]["val"]["content_sha256"],
            "test_used_for_training_or_selection": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": "mps (Apple M4 Pro)" if torch.backends.mps.is_available() else "cpu",
            "libraries": {lib: importlib.metadata.version(lib) for lib in LIBS},
        },
        "reproducibility_note": "Seed 42 fixes data order and initialisation, but GPU kernels on Apple MPS "
                                "are not guaranteed bit-for-bit deterministic, so a re-run should give very "
                                "close but not necessarily identical numbers.",
        "command": ".venv/bin/python scripts/export_training_config.py",
    }
    out = REPO_ROOT / "results" / "bert_base_training_config.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    ta = report["training_arguments"]
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    for k in ["learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio", "optim", "weight_decay",
              "per_device_train_batch_size", "num_train_epochs", "fp16", "seed", "data_seed",
              "metric_for_best_model", "eval_strategy", "eval_steps", "train_sampling_strategy"]:
        print(f"  {k}: {ta.get(k)}")


if __name__ == "__main__":
    main()
