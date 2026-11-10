"""
Evaluate one model on the out-of-distribution email set (scripts/build_ood.py).

Run (one model per process: xgboost and torch must not share a process):
  .venv/bin/python scripts/eval_ood.py <model>
  models: heuristic tfidf_logreg tfidf_xgboost features_rf bert_base bert_pretrained
Writes results/ood_<model>.json and results/predictions/ood_<model>.csv.gz
(scores and labels only, no email text).

Nothing is trained or tuned here: every model is used exactly as it was
evaluated on the main test set, with the same 0.5 decision threshold.

How to read the result: all phishing emails come from one source (Nazario)
and all legitimate ones from another (SpamAssassin, ~20 years older). So the
two numbers to report are
  recall = share of the 776 phishing emails caught,
  fpr    = share of the 2,720 legitimate emails wrongly flagged,
and "by_type" gives the same per source (easy vs hard ham, 2023 vs 2024).
Mixed metrics (accuracy, F1, ROC-AUC) are in the file too, but they depend on
the arbitrary phishing/legitimate mix of this set and on era cues.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import guess_type  # noqa: E402
from src.evaluate import evaluate, print_summary  # noqa: E402

OOD = REPO_ROOT / "data" / "ood" / "ood.parquet"


def scorer(model: str):
    if model == "heuristic":
        from src.models.heuristic import analyse
        return lambda texts: np.array([analyse(t, "email")["normalised"] for t in texts])
    if model in ("tfidf_logreg", "tfidf_xgboost"):
        import joblib
        pipe = joblib.load(REPO_ROOT / "models" / "classical" / f"{model}.joblib")
        return lambda texts: pipe.predict_proba(list(texts))[:, 1]
    if model == "features_rf":
        import joblib
        from src.models.classical import feature_table
        rf = joblib.load(REPO_ROOT / "models" / "classical" / "features_rf.joblib")
        return lambda texts: rf.predict_proba(feature_table(texts, [guess_type(t) for t in texts]))[:, 1]
    if model in ("bert_base", "bert_pretrained"):
        from src.models.bert import BertClassifier
        path = {"bert_base": "models/phishguard-bert-base",
                "bert_pretrained": "models/pretrained/bert-finetuned-phishing"}[model]
        clf = BertClassifier(str(REPO_ROOT / path), half=True, max_length=512)
        return lambda texts: clf.predict_proba(texts, batch_size=32)
    raise SystemExit(f"unknown model {model!r}")


def main(model: str) -> None:
    ood = pd.read_parquet(OOD)
    score = scorer(model)
    t0 = time.perf_counter()
    scores = score(ood["text"].tolist())
    elapsed = time.perf_counter() - t0
    result = evaluate(
        f"ood_{model}", ood, np.clip(scores, 0, 1), latency_seconds_total=elapsed,
        details={
            "description": f"Model '{model}' (unchanged) on the out-of-distribution email set",
            "dataset": "data/ood/ood.parquet, see results/ood_manifest.json",
            "by_type_means": "per email source (nazario_2023/2024 = phishing, sa_easy_ham/sa_hard_ham = legitimate)",
            "how_to_read": "Report recall (phishing caught) and fpr (legitimate flagged) separately; mixed "
                           "metrics depend on the class mix and on the ~20-year era gap between sources.",
            "command": f".venv/bin/python scripts/eval_ood.py {model}",
        },
    )
    print_summary(result)
    for src, m in result["by_type"].items():
        kind = "recall" if src.startswith("nazario") else "fpr"
        print(f"  {src:14} n={m['n']:5}  {kind} {m[kind]}")


if __name__ == "__main__":
    main(sys.argv[1])
