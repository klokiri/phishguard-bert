"""
Data preparation for PhishGuard BERT.

Run from the repo root:   .venv/bin/python -m src.data

Which data and why (decided by Kasim on 2026-11-02, see PROGRESS.md Phase 2):
  The published model ealvaradob/bert-finetuned-phishing was trained on the
  train.json / test.json files of the Hugging Face dataset
  ealvaradob/phishing-dataset at revision 23b8c17f (2023-12-27). We know this
  because train.json has 61,849 rows = exactly the 3,866 steps per epoch (batch
  size 16) on the model card. A month later the author replaced these files
  with combined_reduced.json, which re-sampled the URLs; splitting that file
  would put ~38% of the model's own training data into our test set.
  So we use the ORIGINAL split. That gives the pre-trained model a genuinely
  held-out test set, and every model in this project is tested on it.

What the script does, step by step:
  1. Downloads train.json and test.json at that pinned revision (~551 MB in
     total; skipped if already present).
  2. Checks each file's SHA-256 against the hash Hugging Face publishes.
  3. Loads them with pandas, keeping the published row order.
  4. test  = test.json, unchanged (15,466 rows).
     train / val = train.json split 90/10, stratified by label, seed 42.
     Validation comes from the training data only, so the test set is never
     used to tune or choose anything.
  5. Saves data/splits/{train,val,test}.parquet (git-ignored, large) and two
     small committed evidence files:
         results/split_manifest.json  - hashes proving which rows are in each split
         results/data_summary.json    - sample counts, class balance, checks
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# ── Configuration ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_REPO = "ealvaradob/phishing-dataset"
# Exact dataset commit on the Hub. Pinning it means the data can never
# silently change under us, even if the author uploads new versions.
DATASET_REVISION = "23b8c17f4c9438c3e5d7729be349db049614a266"
# SHA-256 that the Hub publishes for each file (its Git LFS object id).
FILES = {
    "train": ("train.json", "e2e959e3962766c8f917a9fdf76156d6e3b41ce9458eef56deed7830212de15b"),
    "test": ("test.json", "1f5e51ea099f3d2a24754243e08cbdf023b72d77e1eaa2217d2cc1f0ebc590f9"),
}

SEED = 42        # the author's random_state; reused everywhere for consistency
VAL_SIZE = 0.1   # our choice: fraction of train.json held out for validation

RAW_DIR = REPO_ROOT / "data" / "raw" / f"rev_{DATASET_REVISION[:12]}"
SPLIT_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"

# From the model card: 3,866 optimisation steps per epoch at batch size 16.
# Used to confirm this train.json is the one the published model trained on.
MODEL_CARD_STEPS_PER_EPOCH = 3_866
MODEL_CARD_BATCH_SIZE = 16


def sha256_of_file(path: Path) -> str:
    """Hash a file in 1 MB chunks so large files don't need to fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_rows(df: pd.DataFrame) -> str:
    """Hash the CONTENT of a split: source, row ids, labels and texts, in order.

    A parquet file's hash can change if a different pyarrow version writes the
    same data slightly differently. This content hash only changes if the
    actual rows change, so it is the stronger reproducibility check.
    """
    h = hashlib.sha256()
    for src, row_id, text, label in zip(df["source_file"], df["row_id"], df["text"], df["label"]):
        h.update(f"{src}\t{row_id}\t{label}\t".encode("utf-8"))
        h.update(text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def download(filename: str, expected_sha256: str) -> Path:
    """Download one dataset file (once) and verify its hash."""
    path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=RAW_DIR,
        )
    )
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}:\n  expected {expected_sha256}\n  got      {actual}"
        )
    print(f"[1] {path.relative_to(REPO_ROOT)}: {path.stat().st_size:,} bytes, SHA-256 verified")
    return path


def load(path: Path) -> pd.DataFrame:
    """Load a JSON list of {"text", "label"} records, keeping file order."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame.from_records(records, columns=["text", "label"])
    # source_file + row_id = where the sample sits in the published files,
    # so anyone can trace a sample in our splits back to the original data.
    df.insert(0, "source_file", path.name)
    df.insert(1, "row_id", range(len(df)))
    df["label"] = df["label"].astype("int64")

    # Sanity checks: fail loudly rather than train on broken data.
    assert df["text"].map(lambda t: isinstance(t, str)).all(), "non-string text found"
    assert set(df["label"].unique()) <= {0, 1}, "labels must be 0 (benign) or 1 (phishing)"
    print(f"[2] {path.name}: {len(df):,} rows")
    return df


def guess_type(text: str) -> str:
    """Rough guess of what kind of sample this is: url, html or text.

    The dataset does not record the source type, but it matters for later error
    analysis (e.g. "the model struggles on HTML"). This is a heuristic, and is
    labelled as such in the summary:
      - html : contains an HTML tag such as <html, <div, <!doctype ...
      - url  : a single token (no whitespace) that contains a dot
      - text : everything else (emails and SMS messages)
    """
    stripped = text.strip()
    if re.search(r"<\s*(!doctype|html|head|body|div|script|meta|title|a\s)", stripped[:2000], re.I):
        return "html"
    if " " not in stripped and "\n" not in stripped and "." in stripped:
        return "url"
    return "text"


def normalise(text: str) -> str:
    """Lower-case and collapse whitespace, to catch near-identical duplicates."""
    return re.sub(r"\s+", " ", text).strip().lower()


def class_balance(d: pd.DataFrame) -> dict:
    n = len(d)
    phishing = int((d["label"] == 1).sum())
    return {
        "rows": n,
        "phishing": phishing,
        "benign": n - phishing,
        "phishing_fraction": round(phishing / n, 4),
    }


def main() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_full = load(download(*FILES["train"]))
    test = load(download(*FILES["test"]))

    # Validation comes from train.json only (stratified so both parts keep
    # the same phishing/benign ratio).
    train, val = train_test_split(
        train_full, test_size=VAL_SIZE, stratify=train_full["label"], random_state=SEED
    )
    print(f"[3] split: train {len(train):,} / val {len(val):,} / test {len(test):,}")

    parts = {"train": train, "val": val, "test": test}
    for part in parts.values():
        part["type_guess"] = part["text"].map(guess_type)

    # ── Save splits ──
    manifest_splits = {}
    for name, part in parts.items():
        out = SPLIT_DIR / f"{name}.parquet"
        part.to_parquet(out, index=False)
        manifest_splits[name] = {
            "file": str(out.relative_to(REPO_ROOT)),
            "rows": len(part),
            "file_sha256": sha256_of_file(out),
            "content_sha256": sha256_of_rows(part),
        }
        print(f"[4] wrote {out.relative_to(REPO_ROOT)}")

    # ── Is this the published model's training file? ──
    steps_per_epoch = -(-len(train_full) // MODEL_CARD_BATCH_SIZE)  # ceiling division
    provenance_check = {
        "train_json_rows": len(train_full),
        "implied_steps_per_epoch_batch16": steps_per_epoch,
        "model_card_steps_per_epoch": MODEL_CARD_STEPS_PER_EPOCH,
        "matches_model_card": steps_per_epoch == MODEL_CARD_STEPS_PER_EPOCH,
    }

    # ── Data-quality checks ──
    # A test text that also appears in training data lets a model "remember"
    # the answer instead of generalising, which inflates test scores.
    train_full_texts = set(train_full["text"])
    train_full_norm = set(train_full["text"].map(normalise))
    quality = {
        "test_rows_exact_text_in_train_json": int(test["text"].isin(train_full_texts).sum()),
        "test_rows_normalised_text_in_train_json": int(
            test["text"].map(normalise).isin(train_full_norm).sum()
        ),
        "duplicate_texts_within_train_json": int(train_full["text"].duplicated().sum()),
        "duplicate_texts_within_test_json": int(test["text"].duplicated().sum()),
        "note": "Overlapping test rows are kept so the test set stays identical to the "
                "author's; the count is reported as a limitation.",
    }

    manifest = {
        "source": {
            "dataset": DATASET_REPO,
            "revision": DATASET_REVISION,
            "revision_date": "2023-12-27 (dataset licence at this revision: MIT)",
            "files": {name: {"file": fn, "sha256": sha} for name, (fn, sha) in FILES.items()},
            "why_this_revision": "train.json is the published model's training set "
                                 "(row count matches model-card steps/epoch); test.json is "
                                 "its held-out set. Later combined_reduced.json re-sampled URLs.",
        },
        "procedure": {
            "test": "test.json unchanged, in published order",
            "train_val": f"train_test_split(train.json, test_size={VAL_SIZE}, stratify=label, "
                         f"random_state={SEED})",
            "content_sha256": "sha256 over 'source_file<TAB>row_id<TAB>label<TAB>text<NEWLINE>' "
                              "for each row in split order",
        },
        "splits": manifest_splits,
        "provenance_check": provenance_check,
    }

    everything = pd.concat(parts.values())
    summary = {
        "total": class_balance(everything),
        "splits": {name: class_balance(p) for name, p in parts.items()},
        "type_guess_note": "heuristic from src/data.py:guess_type (html tag / single-token URL / other text)",
        "type_guess_counts": {
            name: {k: int(v) for k, v in p["type_guess"].value_counts().sort_index().items()}
            for name, p in {"total": everything, **parts}.items()
        },
        "text_length_chars": {
            k: round(float(v), 1)
            for k, v in everything["text"].str.len().describe(percentiles=[0.5, 0.9, 0.99]).items()
        },
        "quality_checks": quality,
    }

    (RESULTS_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[5] wrote results/split_manifest.json and results/data_summary.json")
    print("\nProvenance check:")
    for k, v in provenance_check.items():
        print(f"  {k}: {v}")
    print("\nQuality checks:")
    for k, v in quality.items():
        if k != "note":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
