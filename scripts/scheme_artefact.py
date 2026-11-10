"""
Counterfactual test of the URL-scheme dataset artefact (PROGRESS.md, Phase 5).

Run (one model per process: xgboost and torch must not share a process):
  .venv/bin/python scripts/scheme_artefact.py <model>
  models: heuristic tfidf_logreg tfidf_xgboost features_rf bert_base bert_pretrained
Each run adds/updates one entry in results/scheme_artefact.json.

Background: in the training data only 10.2% of benign URLs start with
"http(s)://" versus 45.1% of phishing URLs. A model may therefore have learned
"has a scheme -> phishing", which would hurt real users, who paste full URLs.

Experiment: take every TEST url that has NO scheme, and score it twice:
  (a) as it is, e.g.       example.com/page
  (b) with https:// added  https://example.com/page
Nothing else changes, so any difference is caused by the scheme alone.
  - For benign URLs, the share flagged as phishing is the false positive
    rate (FPR); if (b) is much higher than (a), the model penalises https://.
  - For phishing URLs, the share flagged is the recall.
The test set is only used for this measurement; nothing is trained or tuned.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import DECISION_THRESHOLD  # noqa: E402
from src.evaluate import load_test  # noqa: E402

OUT = REPO_ROOT / "results" / "scheme_artefact.json"
HAS_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def scorer(model: str):
    """Return a function texts -> phishing scores for the named model."""
    if model == "heuristic":
        from src.models.heuristic import analyse
        return lambda texts: np.array([analyse(t, "url")["normalised"] for t in texts])
    if model in ("tfidf_logreg", "tfidf_xgboost"):
        import joblib
        pipe = joblib.load(REPO_ROOT / "models" / "classical" / f"{model}.joblib")
        return lambda texts: pipe.predict_proba(list(texts))[:, 1]
    if model == "features_rf":
        import joblib
        from src.models.classical import feature_table
        rf = joblib.load(REPO_ROOT / "models" / "classical" / "features_rf.joblib")
        return lambda texts: rf.predict_proba(feature_table(texts, ["url"] * len(texts)))[:, 1]
    if model in ("bert_base", "bert_pretrained"):
        from src.models.bert import BertClassifier
        path = {"bert_base": "models/phishguard-bert-base",
                "bert_pretrained": "models/pretrained/bert-finetuned-phishing"}[model]
        clf = BertClassifier(str(REPO_ROOT / path), half=True, max_length=512)
        return lambda texts: clf.predict_proba(texts, batch_size=64)
    raise SystemExit(f"unknown model {model!r}")


def main(model: str) -> None:
    test = load_test()
    urls = test[(test["type_guess"] == "url") & ~test["text"].str.match(HAS_SCHEME)]
    score = scorer(model)
    report = {}
    for label, name in ((0, "benign"), (1, "phishing")):
        part = urls[urls["label"] == label]
        plain = score(part["text"].tolist())
        https = score(("https://" + part["text"]).tolist())
        flag_plain = plain >= DECISION_THRESHOLD
        flag_https = https >= DECISION_THRESHOLD
        report[name] = {
            "urls": len(part),
            "flagged_as_phishing_plain": round(float(flag_plain.mean()), 4),
            "flagged_as_phishing_with_https": round(float(flag_https.mean()), 4),
            "flipped_to_phishing": int((~flag_plain & flag_https).sum()),
            "flipped_to_benign": int((flag_plain & ~flag_https).sum()),
            "mean_score_plain": round(float(plain.mean()), 4),
            "mean_score_with_https": round(float(https.mean()), 4),
        }
        print(f"{model:16} {name:9} n={len(part):5}  flagged plain {flag_plain.mean():.4f} "
              f"-> with https:// {flag_https.mean():.4f}")

    all_results = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_results["_about"] = ("Test URLs without a scheme, scored as-is and with 'https://' prepended. "
                             "For benign URLs, 'flagged' = false positive rate; for phishing URLs, = recall. "
                             f"Decision threshold {DECISION_THRESHOLD}. Command: "
                             ".venv/bin/python scripts/scheme_artefact.py <model>")
    all_results[model] = report
    OUT.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
