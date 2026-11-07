"""
Evaluate Baseline 0 (the web app's regex heuristic) on the shared test set.

Run:  .venv/bin/python scripts/run_heuristic.py
Writes results/heuristic.json and results/predictions/heuristic.csv.gz.

Decision rule: the heuristic has three verdicts. For binary metrics we count
only "phishing" (normalised > 0.55) as a phishing prediction, i.e. exactly
what the app labels as phishing. As a secondary view, details also report
the metrics if "suspicious" were counted as phishing too.
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import compute_metrics, evaluate, load_test, print_summary  # noqa: E402
from src.models.heuristic import analyse  # noqa: E402


def main() -> None:
    test = load_test()
    input_types = np.where(test["type_guess"] == "url", "url", "email")

    start = time.perf_counter()
    outputs = [analyse(text, t) for text, t in zip(test["text"], input_types)]
    elapsed = time.perf_counter() - start

    scores = np.array([o["normalised"] for o in outputs])
    verdicts = np.array([o["verdict"] for o in outputs])
    preds = (verdicts == "phishing").astype(int)
    preds_incl_suspicious = (verdicts != "legitimate").astype(int)

    result = evaluate(
        "heuristic",
        test,
        scores,
        latency_seconds_total=elapsed,
        preds=preds,
        threshold=None,
        details={
            "description": "Python port of simulateAnalysis() from frontend/PhishGuard_AI.html; "
                           "parity with the JS verified by scripts/check_heuristic_parity.py",
            "decision_rule": "phishing if verdict == 'phishing' (normalised score > 0.55)",
            "input_type_mapping": "type_guess 'url' -> url mode; text and html -> email mode (no headers)",
            "verdict_counts": {v: int((verdicts == v).sum()) for v in ("phishing", "suspicious", "legitimate")},
            "secondary_suspicious_counted_as_phishing": compute_metrics(
                test["label"].to_numpy(), preds_incl_suspicious, scores
            ),
            "command": ".venv/bin/python scripts/run_heuristic.py",
        },
    )
    print_summary(result)
    print("  verdict counts        ", result["details"]["verdict_counts"])


if __name__ == "__main__":
    main()
