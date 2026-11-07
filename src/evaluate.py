"""
Shared evaluation for every PhishGuard model.

Every model (heuristic, classical, BERT) is scored by THIS function on the SAME
test set, so their numbers are directly comparable. Nothing here is model-
specific: a model only has to hand over, for each test row, a phishing score
between 0 and 1 (and optionally its own yes/no decisions).

Outputs, per model:
  results/<model_name>.json                 - all metrics (committed)
  results/predictions/<model_name>.csv.gz   - per-row scores (committed; used
                                              later for ROC curves and error analysis)

Metric definitions (decided in Phase 3, used everywhere):
  - Positive class = phishing (label 1).
  - precision, recall and F1 are BINARY, for the phishing class. This is the
    standard choice for detection tasks and matches the author's precision and
    recall. A weighted-average F1 (both classes, weighted by size) is also
    stored as "f1_weighted", clearly labelled, because some papers report it.
  - false positive rate (FPR) = FP / (FP + TN): the share of legitimate items
    wrongly flagged as phishing. This is what annoys real users.
  - ROC-AUC uses the continuous score, so it measures ranking quality
    independent of any threshold.
  - latency = wall-clock prediction time for the whole test set divided by the
    number of rows (mean milliseconds per sample), measured by the caller.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
TEST_PATH = REPO_ROOT / "data" / "splits" / "test.parquet"


def load_test() -> pd.DataFrame:
    """The one shared test set (created by src/data.py)."""
    return pd.read_parquet(TEST_PATH)


def git_commit() -> str:
    """Current commit hash, with '+dirty' if there are uncommitted changes.

    Stored in every results file so a number can be traced to the exact code.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=REPO_ROOT) != 0
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """All headline metrics for one set of predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0: if a model never predicts phishing, precision is 0
        # rather than a crash or warning.
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        # ROC-AUC needs both classes present; otherwise it is undefined.
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }
    # Round floats for readability; 4 decimals is plenty for 15k samples.
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in metrics.items()}


def evaluate(
    model_name: str,
    test_df: pd.DataFrame,
    scores,
    latency_seconds_total: float,
    preds=None,
    threshold: float = 0.5,
    details: dict | None = None,
    write: bool = True,
) -> dict:
    """Score one model on the test set and (by default) write its results files.

    model_name            short id used for file names, e.g. "tfidf_logreg"
    test_df               the shared test set (needs label, row_id, type_guess)
    scores                phishing score per row, in [0, 1], same order as test_df
    latency_seconds_total wall-clock seconds the model took to predict all rows
    preds                 the model's own 0/1 decisions; if None, score >= threshold
    details               free-form description: parameters, command, notes
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(test_df):
        raise ValueError(f"{len(scores)} scores for {len(test_df)} test rows")
    if np.isnan(scores).any() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("scores must be finite and within [0, 1]")
    preds = (scores >= threshold).astype(int) if preds is None else np.asarray(preds, dtype=int)

    y_true = test_df["label"].to_numpy()
    result = {
        "model": model_name,
        "test_set": "data/splits/test.parquet (see results/split_manifest.json)",
        "decision_rule": f"phishing if score >= {threshold}" if threshold is not None else "model-specific",
        **compute_metrics(y_true, preds, scores),
        "latency_ms_per_sample": round(1000 * latency_seconds_total / len(test_df), 4),
        # Same metrics broken down by sample type (heuristic guess from src/data.py),
        # e.g. to see whether a model struggles on HTML pages.
        "by_type": {
            t: compute_metrics(y_true[mask], preds[mask], scores[mask])
            for t in sorted(test_df["type_guess"].unique())
            for mask in [(test_df["type_guess"] == t).to_numpy()]
        },
        "git_commit": git_commit(),
        "details": details or {},
    }

    if write:
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{model_name}.json").write_text(json.dumps(result, indent=2) + "\n")
        pred_dir = RESULTS_DIR / "predictions"
        pred_dir.mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "row_id": test_df["row_id"].to_numpy(),
                "label": y_true,
                "type_guess": test_df["type_guess"].to_numpy(),
                "score": np.round(scores, 6),
                "pred": preds,
            }
        ).to_csv(pred_dir / f"{model_name}.csv.gz", index=False)
    return result


def print_summary(result: dict) -> None:
    """One-line-per-metric summary for the terminal."""
    keys = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "latency_ms_per_sample"]
    print(f"\n{result['model']}  (n={result['n']})")
    for k in keys:
        print(f"  {k:<22} {result[k]}")
    print(f"  confusion_matrix       {result['confusion_matrix']}")
