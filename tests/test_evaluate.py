"""Tests for src/evaluate.py using tiny hand-checked examples.

Run:  .venv/bin/python -m pytest tests/test_evaluate.py
"""

import pandas as pd
import pytest

from src.evaluate import compute_metrics, evaluate


def test_metrics_hand_checked():
    # 4 phishing (1) and 4 legitimate (0).
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0]  # TP=3, FN=1, FP=1, TN=3
    y_score = [0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1]
    m = compute_metrics(y_true, y_pred, y_score)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}
    assert m["accuracy"] == 0.75          # 6 correct / 8
    assert m["precision"] == 0.75         # 3 / (3 + 1)
    assert m["recall"] == 0.75            # 3 / (3 + 1)
    assert m["f1"] == 0.75
    assert m["fpr"] == 0.25               # 1 / (1 + 3)
    # Ranking: only the pair (phish 0.4, legit 0.6) is out of order -> 15/16.
    assert m["roc_auc"] == 0.9375


def test_never_predicts_phishing_does_not_crash():
    m = compute_metrics([1, 0, 0], [0, 0, 0], [0.1, 0.1, 0.1])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["fpr"] == 0.0


def _toy_test_df():
    return pd.DataFrame(
        {"row_id": [0, 1, 2, 3], "label": [1, 0, 1, 0], "type_guess": ["url", "url", "text", "text"]}
    )


def test_evaluate_threshold_and_by_type():
    r = evaluate("toy", _toy_test_df(), [0.9, 0.2, 0.4, 0.6], latency_seconds_total=0.004, write=False)
    # threshold 0.5 -> preds [1, 0, 0, 1]
    assert r["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert r["latency_ms_per_sample"] == 1.0
    assert r["by_type"]["url"]["accuracy"] == 1.0
    assert r["by_type"]["text"]["accuracy"] == 0.0


def test_evaluate_rejects_bad_scores():
    with pytest.raises(ValueError):
        evaluate("toy", _toy_test_df(), [0.5, 0.5, 0.5], 1.0, write=False)  # wrong length
    with pytest.raises(ValueError):
        evaluate("toy", _toy_test_df(), [1.5, 0, 0, 0], 1.0, write=False)  # out of range
