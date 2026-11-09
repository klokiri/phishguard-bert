"""
PhishGuard backend: Flask REST API around a swappable ModelService.

Run from the repo root:
    .venv/bin/python -m backend.app                      # Kasim's fine-tuned BERT
    PHISHGUARD_MODEL=heuristic .venv/bin/python -m backend.app
Then open http://127.0.0.1:5000/api/v1/health

Endpoints
    GET  /api/v1/health      status + which model is loaded (polled by the frontend)
    GET  /api/v1/model/info  model, thresholds, limits
    POST /api/v1/analyse     {"content": str, "input_type": "email"|"url", "headers": str (optional)}
    POST /api/v1/batch       {"items": [{"content": ..., "input_type": ...}, ...]}  (max 20; scores only)
    GET  /api/v1/history     recent analyses (hash, verdict, scores; never the content itself)
    GET  /api/v1/stats       totals for the dashboard

Design notes for the report:
  - All settings come from src/config.py (one place, overridable by env vars).
  - The model sits behind backend/model_service.py:ModelService, so the API
    code is the same for heuristic, classical and BERT models (NFR-12).
  - Every database query is parameterised ("?" placeholders): user input is
    never pasted into SQL text (NFR-03).
  - Only a SHA-256 hash of the content is stored, not the text, so the log
    holds no copy of anyone's emails.
  - Rate limit per client IP, sliding one-hour window, 429 + Retry-After (NFR-11).
"""

import hashlib
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, g, jsonify, request
from flask_cors import CORS

from backend import explain
from backend.model_service import ModelService, create_service
from src.config import Config

API = "/api/v1"
VALID_INPUT_TYPES = ("email", "url")


# ── Rate limiting ────────────────────────────────────────────────────────────
class RateLimiter:
    """Sliding-window limiter: at most `limit` requests per `window` seconds per key."""

    def __init__(self, limit: int, window: float = 3600.0):
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)  # key -> timestamps of recent requests
        self.lock = threading.Lock()    # Flask may serve requests on several threads

    def check(self, key: str, now: float | None = None):
        """Return (allowed, retry_after_seconds)."""
        now = time.time() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= now - self.window:  # forget requests older than the window
                q.popleft()
            if len(q) >= self.limit:
                return False, int(q[0] + self.window - now) + 1
            q.append(now)
            return True, 0


# ── Database (SQLite) ────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    input_type    TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    phish_prob    REAL NOT NULL,
    confidence    REAL NOT NULL,
    model         TEXT NOT NULL,
    latency_ms    REAL NOT NULL
)"""


def db_connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str) -> None:
    with db_connect(path) as conn:
        conn.execute(SCHEMA)


# ── Validation ───────────────────────────────────────────────────────────────
def validate_item(data, config: Config):
    """Return (content, input_type, headers) or raise ValueError(message)."""
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    content = data.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("'content' is required and must be a non-empty string.")
    content = content.strip()
    if len(content) > config.max_content_chars:
        raise ValueError(f"'content' is longer than {config.max_content_chars} characters.")
    input_type = data.get("input_type", "email")
    if input_type not in VALID_INPUT_TYPES:
        raise ValueError(f"'input_type' must be one of {list(VALID_INPUT_TYPES)}.")
    headers = data.get("headers", "") or ""
    if not isinstance(headers, str):
        raise ValueError("'headers' must be a string if given.")
    return content, input_type, headers.strip()


def error(message: str, status: int):
    return jsonify({"error": message}), status


# ── App factory ──────────────────────────────────────────────────────────────
def create_app(config: Config | None = None, service: ModelService | None = None) -> Flask:
    """Build the Flask app. Tests pass a fake `service` and a temporary db_path."""
    config = config or Config()
    service = service or create_service(config)
    app = Flask(__name__)
    # Allow the frontend when opened from a local web server or straight from
    # disk (file:// pages send the origin "null"). The API is meant for local use.
    CORS(app, resources={rf"{API}/*": {"origins": [r"^http://(localhost|127\.0\.0\.1)(:\d+)?$", "null"]}})
    init_db(config.db_path)
    limiter = RateLimiter(config.rate_limit_per_hour)
    app.config.update(PHISHGUARD_CONFIG=config, PHISHGUARD_SERVICE=service)

    def rate_limited():
        allowed, retry_after = limiter.check(request.remote_addr or "unknown")
        if allowed:
            return None
        resp, status = error(f"Rate limit exceeded: {config.rate_limit_per_hour} requests per hour.", 429)
        resp.headers["Retry-After"] = str(retry_after)
        return resp, status

    def db():
        if "db" not in g:
            g.db = db_connect(config.db_path)
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    @app.get(f"{API}/health")
    def health():
        info = service.info()
        return jsonify({"status": "ok", "model": info["name"], "model_kind": info["kind"],
                        "device": info["device"]})

    @app.get(f"{API}/model/info")
    def model_info():
        return jsonify({"model": service.info(), "thresholds": config.thresholds(),
                        "max_length_tokens": config.max_length,
                        "max_content_chars": config.max_content_chars,
                        "rate_limit": f"{config.rate_limit_per_hour} requests/hour per IP",
                        "attribution_method": explain.ATTRIBUTION_METHOD})

    @app.post(f"{API}/analyse")
    def analyse():
        limited = rate_limited()
        if limited:
            return limited
        data = request.get_json(silent=True)
        if data is None:
            return error("Request body must be JSON.", 400)
        try:
            content, input_type, headers = validate_item(data, config)
        except ValueError as exc:
            return error(str(exc), 422)

        # Headers (From:, Reply-To: ...) are part of what a reader sees, so the
        # model reads them together with the body.
        text = f"{headers}\n{content}" if headers else content
        start = time.perf_counter()
        try:
            phish_prob = float(service.predict_proba([text], input_type)[0])
        except Exception as exc:  # report model failures without crashing the server
            return error(f"Model inference failed: {type(exc).__name__}", 500)
        model_ms = 1000 * (time.perf_counter() - start)

        details = explain.explain(service, config, text, content, input_type, headers, phish_prob)
        total_ms = 1000 * (time.perf_counter() - start)
        verdict = config.verdict(phish_prob)
        result = {
            "id": uuid.uuid4().hex,
            "verdict": verdict,
            "is_phishing": phish_prob >= config.decision_threshold,
            "confidence": round(100 * max(phish_prob, 1 - phish_prob), 1),
            "phish_prob": round(phish_prob, 4),
            "legit_prob": round(1 - phish_prob, 4),
            **details,
            "model": service.info()["name"],
            "model_kind": service.kind,
            "input_type": input_type,
            "thresholds": config.thresholds(),
            "latency_ms": round(total_ms, 1),
            "model_latency_ms": round(model_ms, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            db().execute(
                "INSERT INTO analyses (id, created_at, input_type, content_hash, verdict, phish_prob, "
                "confidence, model, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (result["id"], result["timestamp"], input_type, hashlib.sha256(text.encode()).hexdigest(),
                 verdict, result["phish_prob"], result["confidence"], result["model"], result["latency_ms"]),
            )
            db().commit()
        except sqlite3.Error:
            app.logger.exception("could not log analysis")  # logging must not break analysis
        return jsonify(result), 200

    @app.post(f"{API}/batch")
    def batch():
        limited = rate_limited()
        if limited:
            return limited
        data = request.get_json(silent=True)
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            return error("'items' must be a non-empty list.", 422)
        if len(items) > 20:
            return error("At most 20 items per batch.", 422)
        try:
            parsed = [validate_item(item, config) for item in items]
        except ValueError as exc:
            return error(str(exc), 422)
        results = []
        for content, input_type, headers in parsed:
            text = f"{headers}\n{content}" if headers else content
            p = float(service.predict_proba([text], input_type)[0])
            results.append({"verdict": config.verdict(p), "phish_prob": round(p, 4), "input_type": input_type})
        return jsonify({"results": results, "count": len(results), "model": service.info()["name"]}), 200

    @app.get(f"{API}/history")
    def history():
        try:
            limit = max(1, min(int(request.args.get("limit", 50)), 200))
        except ValueError:
            return error("'limit' must be an integer.", 422)
        verdict = request.args.get("verdict")
        if verdict:
            rows = db().execute("SELECT * FROM analyses WHERE verdict = ? ORDER BY created_at DESC LIMIT ?",
                                (verdict, limit)).fetchall()
        else:
            rows = db().execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return jsonify({"analyses": [dict(r) for r in rows], "count": len(rows)})

    @app.get(f"{API}/stats")
    def stats():
        row = db().execute(
            "SELECT COUNT(*) AS total, "
            "SUM(verdict = 'phishing') AS phishing, SUM(verdict = 'legitimate') AS legitimate, "
            "SUM(verdict = 'suspicious') AS suspicious, AVG(confidence) AS avg_conf, "
            "AVG(latency_ms) AS avg_latency FROM analyses").fetchone()
        total = row["total"] or 0
        return jsonify({"total": total, "phishing": row["phishing"] or 0, "legitimate": row["legitimate"] or 0,
                        "suspicious": row["suspicious"] or 0,
                        "avg_confidence": round(row["avg_conf"] or 0, 1),
                        "avg_latency_ms": round(row["avg_latency"] or 0, 1)})

    return app


if __name__ == "__main__":
    cfg = Config()
    print(f"Loading model ({cfg.model_kind}) ...")
    application = create_app(cfg)
    print(f"Serving {application.config['PHISHGUARD_SERVICE'].info()} on http://127.0.0.1:5000{API}/health")
    # 127.0.0.1 = reachable only from this computer, not from the network.
    application.run(host="127.0.0.1", port=5000, debug=False)
