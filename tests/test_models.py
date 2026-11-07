"""Tests for the heuristic port and the hand-crafted feature extractor.

Run:  .venv/bin/python -m pytest tests/test_models.py
(Full JS-vs-Python parity on the whole test set: scripts/check_heuristic_parity.py)
"""

from src.models.classical import FEATURE_NAMES, extract_features
from src.models.heuristic import analyse


def test_heuristic_is_deterministic():
    text = "URGENT: verify now or your account will be suspended. Enter your password."
    assert analyse(text, "email") == analyse(text, "email")


def test_heuristic_rules_that_did_not_fire_score_zero():
    r = analyse("hello team, lunch at noon?", "email")
    risk_names = {"Urgency Language", "Sensitive Data Request", "Brand Spoofing"}
    for ind in r["indicators"]:
        if ind["name"] in risk_names:
            assert ind["score"] == 0 and ind["direction"] == "safe"


def test_heuristic_url_score_by_hand():
    # Fires: Brand Impersonation 92, Suspicious TLD 78, No HTTPS 65,
    # Suspicious Path Keywords 70 -> (92+78+65+70)*0.9 = 274.5 -> 274.5/500 = 0.549
    r = analyse("http://paypa1-login.tk/x", "url")
    assert abs(r["normalised"] - 0.549) < 1e-9
    assert r["verdict"] == "suspicious"  # 0.549 is just below the 0.55 phishing cut-off


def test_feature_row_has_fixed_length():
    for text, kind in [("http://a.b/c", "url"), ("hi there", "text"), ("<html><body>x</body></html>", "html")]:
        assert len(extract_features(text, kind)) == len(FEATURE_NAMES)


def test_html_features_zero_for_non_html():
    row = dict(zip(FEATURE_NAMES, extract_features("plain email text", "text")))
    assert all(v == 0 for k, v in row.items() if k.startswith("html_") and k != "html_parse_failed")
    assert row["is_text"] == 1.0
