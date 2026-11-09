"""
The ONE place for settings shared by evaluation and the backend.

Every value can be overridden with an environment variable (shown in
brackets), so the same code runs with a different model or threshold without
editing files, e.g.

    PHISHGUARD_MODEL=heuristic .venv/bin/python -m backend.app

Thresholds, and why they are what they are:
  DECISION_THRESHOLD = 0.5   A text is classified as phishing when the model's
                             phishing probability is >= 0.5. This is the rule
                             every result in results/*.json was measured with,
                             so the app uses the same rule.
  SUSPICIOUS_LOW/HIGH        Display-only caution band: 0.4 <= p < 0.6 is shown
                             as "suspicious" so users see when the model is
                             unsure. It does not change the binary decision
                             (is_phishing) and was not used in evaluation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    """Read an environment variable, converted to the type of the default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return type(default)(raw)


@dataclass
class Config:
    # Which model the backend serves: "bert" | "classical" | "heuristic" [PHISHGUARD_MODEL]
    model_kind: str = field(default_factory=lambda: _env("PHISHGUARD_MODEL", "bert"))
    # Fine-tuned BERT folder (Kasim's model by default) [PHISHGUARD_BERT_PATH]
    bert_path: str = field(default_factory=lambda: _env(
        "PHISHGUARD_BERT_PATH", str(REPO_ROOT / "models" / "phishguard-bert-base")))
    # Saved scikit-learn pipeline for the classical option [PHISHGUARD_CLASSICAL_PATH]
    classical_path: str = field(default_factory=lambda: _env(
        "PHISHGUARD_CLASSICAL_PATH", str(REPO_ROOT / "models" / "classical" / "tfidf_logreg.joblib")))
    # BERT input limit in tokens; 512 is what the model was trained with [PHISHGUARD_MAX_LENGTH]
    max_length: int = field(default_factory=lambda: _env("PHISHGUARD_MAX_LENGTH", 512))

    decision_threshold: float = field(default_factory=lambda: _env("PHISHGUARD_DECISION_THRESHOLD", 0.5))
    suspicious_low: float = field(default_factory=lambda: _env("PHISHGUARD_SUSPICIOUS_LOW", 0.4))
    suspicious_high: float = field(default_factory=lambda: _env("PHISHGUARD_SUSPICIOUS_HIGH", 0.6))

    # Requests per hour per client IP (NFR-11 asks for 100/hour per user) [PHISHGUARD_RATE_LIMIT]
    rate_limit_per_hour: int = field(default_factory=lambda: _env("PHISHGUARD_RATE_LIMIT", 100))
    # Largest accepted input, in characters (protects the server from huge uploads)
    max_content_chars: int = field(default_factory=lambda: _env("PHISHGUARD_MAX_CHARS", 100_000))
    # Leave-one-word-out attribution looks at this many words at most (cost grows per word)
    attribution_max_words: int = field(default_factory=lambda: _env("PHISHGUARD_ATTRIBUTION_WORDS", 60))
    attribution_top_k: int = 5

    # SQLite file for the analysis log [PHISHGUARD_DB]
    db_path: str = field(default_factory=lambda: _env("PHISHGUARD_DB", str(REPO_ROOT / "data" / "analyses.db")))

    def verdict(self, phish_prob: float) -> str:
        """Display verdict from the phishing probability (see module docstring)."""
        if self.suspicious_low <= phish_prob < self.suspicious_high:
            return "suspicious"
        return "phishing" if phish_prob >= self.decision_threshold else "legitimate"

    def thresholds(self) -> dict:
        return {"decision": self.decision_threshold,
                "suspicious_band": [self.suspicious_low, self.suspicious_high]}


# Used by src/evaluate.py so offline evaluation and the app share one rule.
DECISION_THRESHOLD = Config().decision_threshold
