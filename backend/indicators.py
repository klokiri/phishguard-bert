"""
Deterministic, rule-based phishing indicators.

What an indicator is: a fixed, human-readable rule ("the link's host is a raw
IP address", "the text contains urgency words") checked against the input.
It is shown ONLY when the rule fires, together with its evidence (the matched
words or the measured value). Nothing is random and nothing is shown for a
rule that did not fire.

Where the 0-100 score comes from (not chosen by hand):
  scripts/build_indicator_stats.py runs every rule over the TRAINING split and
  records, for the samples where the rule fires, what fraction were phishing.
  That is saved in results/indicator_stats.json. At run time:
    - if that phishing rate is above the overall phishing rate of the training
      data, the indicator is a "risk" and score = phishing rate x 100;
    - otherwise it is "safe" and score = legitimate rate x 100.
  So "Raw IP address as host - 100" means: of the training URLs where this rule
  fired, (rounded) 100% were phishing. Not shown: rules that fired on fewer
  than MIN_SUPPORT training samples (too little evidence), rules within
  MIN_LIFT of the base rate (uninformative), and EXCLUDED_FROM_DISPLAY rules
  (dataset artefacts that would mislead users).

What indicators are NOT: they are not an explanation of how BERT decided.
They are independent checks shown next to the model's score. The model-derived
explanation is the word attribution in backend/attribution.py.

URL rules use the reference project's URLFeatures (src/features.py) where it
has a matching feature; email rules use TextFeatures plus keyword lists below.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from src.config import REPO_ROOT
from src.features import TextFeatures, URLFeatures

STATS_PATH = REPO_ROOT / "results" / "indicator_stats.json"
MIN_SUPPORT = 30
# A rule whose phishing rate is within 10 percentage points of the overall
# rate carries almost no information, so it is not shown.
MIN_LIFT = 0.10
# Measured but never shown, with the reason (see PROGRESS.md, Phase 5):
EXCLUDED_FROM_DISPLAY = {
    "url_https": "dataset artefact: only 10% of benign training URLs include a scheme vs 45% of phishing "
                 "URLs, so 'https://' looks like a phishing sign in this data; showing it would mislead users",
    "url_explicit_http": "same scheme artefact as url_https",
}


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    description: str  # the exact rule, shown in the API response
    check: Callable[[str], str | None]  # returns evidence text if the rule fires, else None


# ── URL rules ────────────────────────────────────────────────────────────────
SUSPICIOUS_TLDS = ("tk", "ml", "ga", "cf", "gq", "xyz", "top", "ru", "cn", "biz", "info", "click", "zip")
BRAND_LOOKALIKES = r"paypa1|amaz0n|g00gle|micr0soft|microsofft|app1e|faceb00k|netf1ix|1nstagram|linkedln"


def _url_features(url: str):
    # urlparse only finds the host if there is a scheme ("http://"); many URLs
    # in the dataset have none, so add one for parsing (the original string is
    # still used for length and character counts).
    parse_target = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url
    feats = URLFeatures(parse_target)
    return feats, urlparse(parse_target)


def _url_rule(fn):
    """Wrap a check that receives (features, parsed, original_url)."""
    def check(url):
        feats, parsed = _url_features(url)
        return fn(feats, parsed, url)
    return check


IPV4 = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

URL_RULES = [
    # Checks the host name WITHOUT the port: the reference URLFeatures.host_is_ip()
    # tests "host:port" and so misses e.g. http://10.0.0.1:8080/ (found by a unit test).
    Rule("url_ip_host", "Raw IP address as host", "The host is an IP address instead of a domain name.",
         _url_rule(lambda f, p, u: p.hostname if IPV4.match(p.hostname or "") else None)),
    Rule("url_at_sign", "'@' symbol in URL", "The URL contains '@' (browsers ignore everything before it).",
         _url_rule(lambda f, p, u: "@" if f.has_haveat_sign() else None)),
    Rule("url_shortener", "URL shortening service", "The URL uses a known link shortener (bit.ly, tinyurl, ...).",
         _url_rule(lambda f, p, u: p.hostname if f.use_shortening_services() else None)),
    Rule("url_hyphen_domain", "Hyphen in domain name", "The host name contains '-'.",
         _url_rule(lambda f, p, u: p.hostname if f.prefix_suffix_presence() else None)),
    Rule("url_double_slash", "Redirect-style '//' in path", "'//' appears after the scheme part of the URL.",
         _url_rule(lambda f, p, u: "//" if f.has_double_slash_in_wrong_position() else None)),
    Rule("url_port", "Explicit port number", "The host includes a port number (e.g. :8080).",
         _url_rule(lambda f, p, u: f"port {p.port}" if f.has_port() else None)),
    Rule("url_login_word", "'login' in URL", "The URL contains the word 'login'.",
         _url_rule(lambda f, p, u: "login" if f.has_login_in_string() else None)),
    Rule("url_admin_word", "'admin' in URL", "The URL contains the word 'admin'.",
         _url_rule(lambda f, p, u: "admin" if f.has_admin_in_string() else None)),
    Rule("url_encoded", "Percent-encoded characters", "The URL contains '%' encoding.",
         _url_rule(lambda f, p, u: f"{f.num_encoded_char()} encoded" if f.is_encoded() else None)),
    Rule("url_long", "Long URL (over 75 characters)", "The URL is longer than 75 characters.",
         _url_rule(lambda f, p, u: f"{len(u)} characters" if len(u) > 75 else None)),
    Rule("url_many_subdomains", "Many subdomains (3+ dots in host)", "The host name has 3 or more dots.",
         _url_rule(lambda f, p, u: p.hostname if (p.hostname or "").count(".") >= 3 else None)),
    Rule("url_many_digits", "Many digits (more than 5)", "The URL contains more than 5 digits.",
         _url_rule(lambda f, p, u: f"{f.number_of_digits()} digits" if f.number_of_digits() > 5 else None)),
    Rule("url_suspicious_tld", "Frequently abused top-level domain",
         f"The host ends in one of: {', '.join('.' + t for t in SUSPICIOUS_TLDS)}.",
         _url_rule(lambda f, p, u: p.hostname if (p.hostname or "").rsplit(".", 1)[-1] in SUSPICIOUS_TLDS else None)),
    Rule("url_brand_lookalike", "Brand look-alike spelling", "The URL contains a look-alike brand spelling "
         "(e.g. 'paypa1', 'g00gle').",
         _url_rule(lambda f, p, u: (m.group(0) if (m := re.search(BRAND_LOOKALIKES, u.lower())) else None))),
    Rule("url_explicit_http", "Unencrypted http://", "The URL explicitly starts with http:// (no TLS).",
         lambda u: "http://" if u.lower().startswith("http://") else None),
    Rule("url_https", "Uses https://", "The URL starts with https://.",
         lambda u: "https://" if u.lower().startswith("https://") else None),
]


# ── Email / text rules ───────────────────────────────────────────────────────
KEYWORDS = {
    "text_urgency": ("Urgency language", ["urgent", "immediately", "act now", "within 24 hours", "expire",
                                          "suspended", "verify now", "as soon as possible", "final notice",
                                          "last chance"]),
    "text_credentials": ("Request for credentials or payment details",
                         ["password", "credit card", "card number", "cvv", "social security", "ssn",
                          "bank account", "login details", "pin number", "verify your account",
                          "confirm your identity"]),
    "text_threat": ("Threat or consequence language", ["account will be closed", "legal action", "permanently",
                                                       "terminated", "law enforcement", "locked", "unauthorized"]),
    "text_prize": ("Prize or money lure", ["winner", "you have won", "prize", "claim", "congratulations",
                                           "free gift", "cash", "lottery", "reward"]),
    "text_generic_greeting": ("Generic greeting", ["dear customer", "dear user", "dear valued", "dear member",
                                                   "dear account holder", "dear sir/madam"]),
}


def _keyword_rule(rule_id, name, words):
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)

    def check(text):
        found = sorted({m.group(0).lower() for m in pattern.finditer(text)})
        return ", ".join(found) if found else None

    return Rule(rule_id, name, f"The text contains any of: {', '.join(words)}.", check)


IP_LINK = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

EMAIL_RULES = [_keyword_rule(rid, name, words) for rid, (name, words) in KEYWORDS.items()] + [
    Rule("text_has_url", "Contains a link", "The text contains a URL (reference TextFeatures.has_url).",
         lambda t: "link found" if TextFeatures(t).has_url() else None),
    Rule("text_ip_link", "Link to a raw IP address", "The text contains a link whose host is an IP address.",
         lambda t: (m.group(0) if (m := IP_LINK.search(t)) else None)),
    Rule("text_has_email", "Contains an email address", "The text contains an email address "
         "(reference TextFeatures.has_email_adress).",
         lambda t: "address found" if TextFeatures(t).has_email_adress() else None),
    Rule("text_brand_lookalike", "Brand look-alike spelling", "The text contains a look-alike brand spelling.",
         lambda t: (m.group(0) if (m := re.search(BRAND_LOOKALIKES, t.lower())) else None)),
]

RULES = {"url": URL_RULES, "email": EMAIL_RULES}


def fired_rules(text: str, input_type: str):
    """[(rule, evidence)] for every rule that fires on this text."""
    out = []
    for rule in RULES[input_type]:
        evidence = rule.check(text)
        if evidence is not None:
            out.append((rule, evidence))
    return out


def load_stats(path: Path = STATS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: .venv/bin/python scripts/build_indicator_stats.py")
    return json.loads(path.read_text())


def build_indicators(text: str, input_type: str, stats: dict) -> list[dict]:
    """Indicators for one input, strongest first. See module docstring for the scoring rule."""
    base = stats["base_phishing_rate"][input_type]
    indicators = []
    for rule, evidence in fired_rules(text, input_type):
        s = stats["rules"].get(rule.id)
        if rule.id in EXCLUDED_FROM_DISPLAY or not s or s["fired"] < MIN_SUPPORT:
            continue  # excluded, or not enough training evidence to give this rule a score
        if abs(s["phishing_rate"] - base) < MIN_LIFT:
            continue  # fires about as often on phishing as on anything else: uninformative
        risk = s["phishing_rate"] > base
        rate = s["phishing_rate"] if risk else 1 - s["phishing_rate"]
        indicators.append({
            "name": rule.name,
            "score": round(100 * rate),
            "direction": "risk" if risk else "safe",
            "evidence": evidence[:120],
            "rule": rule.description,
            "basis": f"{'phishing' if risk else 'legitimate'} in {100 * rate:.0f}% of {s['fired']:,} "
                     f"training samples where this rule fired",
        })
    indicators.sort(key=lambda i: (i["direction"] != "risk", -i["score"]))
    return indicators
