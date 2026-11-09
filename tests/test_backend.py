"""
Backend API tests.

Run:  .venv/bin/python -m pytest tests/test_backend.py
Coverage (NFR-09):  .venv/bin/python scripts/coverage_report.py

The model is replaced by FakeService, a tiny deterministic stand-in, so the
tests are fast and test the API rather than BERT. Every API test also runs
against the real HeuristicService WITHOUT any change to the test code: that
is the NFR-12 check that the API layer does not depend on the model.
"""

from pathlib import Path

import numpy as np
import pytest

from backend import attribution, indicators
from backend.app import RateLimiter, create_app
from backend.model_service import ClassicalService, HeuristicService, ModelService, create_service
from src.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
PHISHY = "URGENT: verify your password immediately at http://192.168.0.1/login"
BENIGN = "Hi team, the standup moves to 10am tomorrow. Thanks, Sam"


class FakeService(ModelService):
    """Deterministic stand-in: 0.9 if the text contains 'verify', else 0.1."""

    name, kind, device = "fake-model", "fake", "cpu"

    def predict_proba(self, texts, input_type):
        return np.array([0.9 if "verify" in t.lower() else 0.1 for t in texts])


class BrokenService(FakeService):
    def predict_proba(self, texts, input_type):
        raise RuntimeError("model crashed")


def make_client(tmp_path, service, rate_limit=100):
    cfg = Config()
    cfg.db_path = str(tmp_path / "test.db")
    cfg.rate_limit_per_hour = rate_limit
    return create_app(cfg, service).test_client()


@pytest.fixture(params=[FakeService, HeuristicService], ids=["fake", "heuristic"])
def client(request, tmp_path):
    return make_client(tmp_path, request.param())


# ── Required cases ───────────────────────────────────────────────────────────
def test_valid_request_returns_200_with_expected_shape(client):
    r = client.post("/api/v1/analyse", json={"content": PHISHY, "input_type": "email"})
    assert r.status_code == 200
    body = r.get_json()
    for key in ("verdict", "confidence", "phish_prob", "legit_prob", "indicators", "top_tokens",
                "explanation", "model", "latency_ms", "attribution_method"):
        assert key in body
    assert body["verdict"] in ("phishing", "suspicious", "legitimate")
    assert 0 <= body["phish_prob"] <= 1 and abs(body["phish_prob"] + body["legit_prob"] - 1) < 1e-3
    for ind in body["indicators"]:
        assert set(("name", "score", "direction")) <= set(ind) and 0 <= ind["score"] <= 100


@pytest.mark.parametrize("content", ["", "   ", None, 123])
def test_empty_or_non_string_content_returns_422(client, content):
    r = client.post("/api/v1/analyse", json={"content": content, "input_type": "email"})
    assert r.status_code == 422


def test_missing_content_returns_422(client):
    assert client.post("/api/v1/analyse", json={"input_type": "url"}).status_code == 422


@pytest.mark.parametrize("input_type", ["sms", "", "URL", 5])
def test_invalid_input_type_returns_422(client, input_type):
    r = client.post("/api/v1/analyse", json={"content": PHISHY, "input_type": input_type})
    assert r.status_code == 422


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    c = make_client(tmp_path, FakeService(), rate_limit=3)
    for _ in range(3):
        assert c.post("/api/v1/analyse", json={"content": BENIGN}).status_code == 200
    r = c.post("/api/v1/analyse", json={"content": BENIGN})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


# ── More validation and endpoints ────────────────────────────────────────────
def test_non_json_body_returns_400(client):
    r = client.post("/api/v1/analyse", data="not json", content_type="text/plain")
    assert r.status_code == 400


def test_body_must_be_object(client):
    assert client.post("/api/v1/analyse", json=["a list"]).status_code == 422


def test_too_long_content_returns_422(tmp_path):
    c = make_client(tmp_path, FakeService())
    assert c.post("/api/v1/analyse", json={"content": "a" * 100_001}).status_code == 422


def test_headers_must_be_string(client):
    r = client.post("/api/v1/analyse", json={"content": PHISHY, "headers": ["From: x"]})
    assert r.status_code == 422


def test_headers_are_read_with_the_body(tmp_path):
    c = make_client(tmp_path, FakeService())
    r = c.post("/api/v1/analyse", json={"content": BENIGN, "headers": "Subject: please verify"})
    assert r.get_json()["phish_prob"] == 0.9  # 'verify' only appears in the headers


def test_model_failure_returns_500(tmp_path):
    c = make_client(tmp_path, BrokenService())
    r = c.post("/api/v1/analyse", json={"content": PHISHY})
    assert r.status_code == 500 and "RuntimeError" in r.get_json()["error"]


def test_health_and_model_info(client):
    h = client.get("/api/v1/health").get_json()
    assert h["status"] == "ok" and "model" in h and "device" in h
    info = client.get("/api/v1/model/info").get_json()
    assert info["thresholds"] == {"decision": 0.5, "suspicious_band": [0.4, 0.6]}


def test_history_stores_hash_not_content_and_filters(client):
    client.post("/api/v1/analyse", json={"content": PHISHY})
    client.post("/api/v1/analyse", json={"content": BENIGN})
    rows = client.get("/api/v1/history").get_json()["analyses"]
    assert len(rows) == 2
    assert all(len(r["content_hash"]) == 64 for r in rows)
    assert not any(PHISHY in str(v) or BENIGN in str(v) for r in rows for v in r.values())
    verdict = rows[0]["verdict"]
    assert all(r["verdict"] == verdict for r in client.get(f"/api/v1/history?verdict={verdict}").get_json()["analyses"])


def test_history_sql_injection_is_harmless(client):
    client.post("/api/v1/analyse", json={"content": PHISHY})
    r = client.get("/api/v1/history", query_string={"verdict": "phishing' OR '1'='1"})
    assert r.status_code == 200 and r.get_json()["count"] == 0   # treated as a plain value
    assert client.get("/api/v1/history").get_json()["count"] == 1  # table intact


def test_history_limit_validation(client):
    assert client.get("/api/v1/history?limit=abc").status_code == 422
    assert client.get("/api/v1/history?limit=0").status_code == 200


def test_stats_counts(tmp_path):
    c = make_client(tmp_path, FakeService())
    c.post("/api/v1/analyse", json={"content": PHISHY})
    c.post("/api/v1/analyse", json={"content": BENIGN})
    s = c.get("/api/v1/stats").get_json()
    assert s["total"] == 2 and s["phishing"] == 1 and s["legitimate"] == 1


def test_batch(client):
    ok = client.post("/api/v1/batch", json={"items": [{"content": PHISHY}, {"content": BENIGN, "input_type": "email"}]})
    assert ok.status_code == 200 and ok.get_json()["count"] == 2
    assert client.post("/api/v1/batch", json={"items": []}).status_code == 422
    assert client.post("/api/v1/batch", json={"items": [{"content": "x"}] * 21}).status_code == 422
    assert client.post("/api/v1/batch", json={"items": [{"content": ""}]}).status_code == 422


def test_batch_rate_limited(tmp_path):
    c = make_client(tmp_path, FakeService(), rate_limit=1)
    assert c.post("/api/v1/batch", json={"items": [{"content": BENIGN}]}).status_code == 200
    assert c.post("/api/v1/batch", json={"items": [{"content": BENIGN}]}).status_code == 429


def test_cors_allows_local_pages_only(tmp_path):
    c = make_client(tmp_path, FakeService())
    for origin, allowed in [("null", True), ("http://localhost:8000", True),
                            ("http://127.0.0.1:5500", True), ("https://evil.example", False)]:
        r = c.get("/api/v1/health", headers={"Origin": origin})
        assert (r.headers.get("Access-Control-Allow-Origin") == origin) is allowed


# ── Units: config, limiter, services, indicators, attribution ────────────────
def test_verdict_bands():
    cfg = Config()
    assert cfg.verdict(0.39) == "legitimate"
    assert cfg.verdict(0.40) == "suspicious" and cfg.verdict(0.59) == "suspicious"
    assert cfg.verdict(0.60) == "phishing"


def test_rate_limiter_window_expires():
    rl = RateLimiter(limit=2, window=10)
    assert rl.check("ip", now=0)[0] and rl.check("ip", now=1)[0]
    allowed, retry = rl.check("ip", now=2)
    assert not allowed and retry == 9
    assert rl.check("ip", now=10.5)[0]  # first request has left the window
    assert rl.check("other-ip", now=2)[0]  # limits are per client


def test_create_service():
    cfg = Config()
    cfg.model_kind = "heuristic"
    assert isinstance(create_service(cfg), HeuristicService)
    cfg.model_kind = "nonsense"
    with pytest.raises(ValueError):
        create_service(cfg)


CLASSICAL = REPO_ROOT / "models" / "classical" / "tfidf_logreg.joblib"


@pytest.mark.skipif(not CLASSICAL.exists(), reason="run scripts/train_classical.py first")
def test_classical_service_through_api(tmp_path):
    cfg = Config()
    cfg.model_kind, cfg.classical_path = "classical", str(CLASSICAL)
    c = make_client(tmp_path, create_service(cfg))
    body = c.post("/api/v1/analyse", json={"content": PHISHY}).get_json()
    assert body["model"] == "tfidf_logreg" and 0 <= body["phish_prob"] <= 1


BERT_DIR = REPO_ROOT / "models" / "phishguard-bert-base"


@pytest.mark.skipif(not (BERT_DIR / "config.json").exists(), reason="run scripts/train_bert.py first")
def test_bert_service_through_api(tmp_path):
    cfg = Config()
    cfg.model_kind, cfg.bert_path = "bert", str(BERT_DIR)
    c = make_client(tmp_path, create_service(cfg))
    phish = c.post("/api/v1/analyse", json={"content": PHISHY}).get_json()
    benign = c.post("/api/v1/analyse", json={"content": BENIGN}).get_json()
    assert phish["model"] == "phishguard-bert-base" and phish["phish_prob"] > benign["phish_prob"]


def test_indicators_are_deterministic_and_only_fired_rules():
    stats = indicators.load_stats()
    a = indicators.build_indicators(PHISHY, "email", stats)
    assert a == indicators.build_indicators(PHISHY, "email", stats)
    names = {i["name"] for i in a}
    assert "Link to a raw IP address" in names and "Urgency language" in names
    assert "Prize or money lure" not in names  # did not fire
    assert indicators.build_indicators(BENIGN, "email", stats) == []


def test_indicator_score_follows_training_stats():
    stats = {"base_phishing_rate": {"url": 0.5},
             "rules": {"url_ip_host": {"fired": 100, "phishing_rate": 0.97},
                       "url_encoded": {"fired": 100, "phishing_rate": 0.2},
                       "url_port": {"fired": 5, "phishing_rate": 0.9}}}
    inds = {i["name"]: i for i in indicators.build_indicators("http://10.0.0.1:8080/a%20b", "url", stats)}
    assert inds["Raw IP address as host"]["score"] == 97 and inds["Raw IP address as host"]["direction"] == "risk"
    assert inds["Percent-encoded characters"]["score"] == 80 and inds["Percent-encoded characters"]["direction"] == "safe"
    assert "Explicit port number" not in inds  # only 5 training hits < MIN_SUPPORT


def test_scheme_artefact_rules_are_never_shown():
    shown = {i["name"] for i in indicators.build_indicators("https://example.com", "url", indicators.load_stats())}
    assert "Uses https://" not in shown


def test_occlusion_finds_the_word_that_matters():
    tokens = attribution.occlusion(FakeService(), "please verify this now", "email", 0.9, max_words=60, top_k=5)
    assert tokens[0]["token"] == "verify" and tokens[0]["direction"] == "phishing"
    assert all(t["token"] != "please" for t in tokens)  # no effect -> not listed
    assert attribution.occlusion(FakeService(), "!!!", "email", 0.1, 60, 5) == []


def test_explanation_mentions_only_what_fired(tmp_path):
    c = make_client(tmp_path, FakeService())
    benign = c.post("/api/v1/analyse", json={"content": BENIGN}).get_json()["explanation"]
    assert "None of the rule-based warning-sign checks fired" in benign and "Urgency" not in benign
    phish = c.post("/api/v1/analyse", json={"content": PHISHY}).get_json()["explanation"]
    assert "Urgency language" in phish and "'verify'" in phish
