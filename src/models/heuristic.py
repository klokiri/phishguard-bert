"""
Baseline 0: Python port of the regex heuristic simulateAnalysis() from
frontend/PhishGuard_AI.html (lines 1765-1870 at the time of porting).

Why port it: the web app currently falls back to this heuristic, so it is the
"before" in our before/after story. Measuring it on the same test set as the
ML models shows how much the ML actually adds.

The port reproduces the JavaScript's scoring exactly (checked against the
original code running in Node.js by scripts/check_heuristic_parity.py), with
two deliberate differences, both about the indicator list, not the score:
  1. The JS gives indicators that did NOT fire a random score
     (Math.random() * 30). A measurement must be repeatable, so here they get
     score 0. This does not affect the phishing score or the verdict.
  2. The artificial 1.8-2.6 second delay (setTimeout) is not reproduced.

How the score works (same as the JS):
  - Each risk rule that fires adds  rule_score * weight  (0.9 for URLs, 0.85
    for emails); each "safe" rule that fires subtracts  score * 0.6 / 0.5.
  - The total is clipped to [0, 500] and divided by 500 -> "normalised" (0..1).
  - verdict: phishing if normalised > 0.55, suspicious if > 0.25, else legitimate.
"""

import re

# re.search(pattern, s) is the Python equivalent of JavaScript's /pattern/.test(s).


def _url_rules(content: str, lc: str):
    """(fired?, name, score) for each URL risk rule, in the JS order."""
    return [
        (bool(re.search(r"paypa1|amaz0n|g00gle|apple-id|secure-login|verify-account|account-confirm", lc)),
         "Brand Impersonation", 92),
        (bool(re.search(r"\.ru|\.tk|\.xyz|\.top|\.cc|\.biz", lc)) and not re.search(r"\.co\.uk", lc),
         "Suspicious TLD", 78),
        (bool(re.search(r"http://(?!localhost)", lc)) and not re.search(r"^https", lc),
         "No HTTPS Encryption", 65),
        # [0-9] not \d: in JavaScript \d is ASCII-only, in Python it matches any Unicode digit.
        (bool(re.search(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", lc)), "IP Address in Host", 88),
        (_js_length(content) > 100, "Excessive URL Length", 55),
        (len(re.findall(r"[0-9]", content)) > 8, "High Digit Density", 60),
        (bool(re.search(r"login|verify|secure|account|update|confirm|restore", lc)),
         "Suspicious Path Keywords", 70),
        (bool(re.search(r"\?.*token|redirect|ref=", lc)), "Suspicious Query Parameters", 62),
    ]


def _url_safe_rules(content: str, lc: str):
    return [
        (bool(re.search(r"^https://", content)), "HTTPS Protocol", 85),
        (bool(re.search(r"\.(gov|edu|org|co\.uk)", lc)), "Trusted Domain Extension", 80),
    ]


def _email_rules(lc: str, hc: str):
    return [
        (bool(re.search(r"urgent|immediately|suspended|limited|verify now|act now|24 hours|click here|click immediately", lc)),
         "Urgency Language", 85),
        (bool(re.search(r"password|credit card|ssn|social security|cvv|bank account|billing", lc)),
         "Sensitive Data Request", 92),
        (bool(re.search(r"paypa1|amaz0n|g00gle|microsofft|appie|bankofamerica", lc + hc)),
         "Brand Spoofing", 95),
        (bool(re.search(r"\.ru|\.tk|\.xyz|@.*-.*\.com", hc)), "Suspicious Sender Domain", 88),
        (bool(re.search(r"reply-to.*(?!@.*\.com)", re.sub(r"\s", "", hc))) and "reply-to" in hc,
         "Reply-To Mismatch", 82),
        (bool(re.search(r"permanently|legal action|account closure|law enforcement", lc)),
         "Threat Language", 80),
        (bool(re.search(r"http://[^\s]+\.tk|\.ru|\.xyz", lc)), "Malicious Embedded URLs", 90),
        (bool(re.search(r"dear (valued|customer|user|member)", lc)), "Generic Salutation", 55),
    ]


def _email_safe_rules(lc: str, hc: str):
    return [
        (bool(re.search(r"github\.com|microsoft\.com|google\.com|amazon\.com", hc)), "Known Trusted Sender", 90),
        (not re.search(r"(urgent|immediately|suspended)", lc, re.I), "No Urgency Indicators", 75),
    ]


def _js_length(s: str) -> int:
    """JavaScript's string.length counts UTF-16 code units: characters outside
    the Basic Multilingual Plane (e.g. emoji) count as 2. Python's len() counts
    them as 1, so we count the extra units to match the JS exactly."""
    return len(s) + sum(1 for ch in s if ord(ch) > 0xFFFF)


def _js_lower(s: str) -> str:
    """Stands in for JavaScript's toLowerCase(). Python's lower() may differ on
    a few rare Unicode characters; scripts/check_heuristic_parity.py measures
    whether that ever changes a result on the test set."""
    return s.lower()


def analyse(content: str, input_type: str, headers: str = "") -> dict:
    """Run the heuristic on one input. input_type is "url" or "email"."""
    lc = _js_lower(content)
    hc = _js_lower(headers)
    phish_score = 0.0
    indicators = []

    if input_type == "url":
        risk, safe, risk_w, safe_w = _url_rules(content, lc), _url_safe_rules(content, lc), 0.9, 0.6
    else:
        risk, safe, risk_w, safe_w = _email_rules(lc, hc), _email_safe_rules(lc, hc), 0.85, 0.5

    for fired, name, score in risk:
        phish_score += score * risk_w if fired else 0
        # Deterministic: a rule that did not fire scores 0 (the JS used a random number).
        indicators.append({"name": name, "score": score if fired else 0, "direction": "risk" if fired else "safe"})
    for fired, name, score in safe:
        phish_score -= score * safe_w if fired else 0
        if fired:
            indicators.append({"name": name, "score": score, "direction": "safe"})

    phish_score = max(0.0, min(500.0, phish_score))
    normalised = phish_score / 500

    if normalised > 0.55:
        verdict = "phishing"
    elif normalised > 0.25:
        verdict = "suspicious"
    else:
        verdict = "legitimate"

    indicators.sort(key=lambda i: i["score"], reverse=True)
    return {"normalised": normalised, "verdict": verdict, "indicators": indicators[:7]}
