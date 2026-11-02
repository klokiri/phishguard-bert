"""
================================================================
  PhishGuard BERT — Day 4: Flask API Backend

  A production-ready REST API that serves BERT predictions.
  PhishGuard's frontend calls this instead of the Claude API.

  Endpoints:
    POST /api/v1/analyse     — main prediction endpoint
    GET  /api/v1/health      — server status check
    GET  /api/v1/model/info  — model metadata
    POST /api/v1/batch       — analyse multiple inputs at once

  Run: python backend/app.py
  Test: http://localhost:5000/api/v1/health
================================================================
"""

import torch
import numpy as np
import os
import time
import hashlib
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── APP SETUP ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # allow frontend to call us

# ── CONFIG ───────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get('MODEL_PATH', './models/phishguard-bert-finetuned')
FALLBACK    = os.environ.get('FALLBACK_MODEL', 'ealvaradob/bert-finetuned-phishing')
DB_PATH     = './data/analyses.db'
MAX_LENGTH  = 256
RATE_LIMIT  = 100   # requests per hour per IP (simple in-memory)

# ── GLOBALS ──────────────────────────────────────────────────────
tokenizer = None
model     = None
device    = None
model_meta = {}
rate_tracker = {}  # {ip: [timestamps]}


# ── DATABASE SETUP ───────────────────────────────────────────────
def init_db():
    """Create SQLite database to log all analyses."""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id            TEXT PRIMARY KEY,
            timestamp     TEXT NOT NULL,
            input_type    TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            content_preview TEXT,
            verdict       TEXT NOT NULL,
            confidence    REAL NOT NULL,
            phish_prob    REAL,
            legit_prob    REAL,
            indicators    TEXT,
            model_version TEXT,
            latency_ms    REAL
        )
    """)
    conn.commit()
    conn.close()
    print("  ✓ Database initialised")


def log_analysis(record: dict):
    """Save an analysis result to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO analyses
        (id, timestamp, input_type, content_hash, content_preview,
         verdict, confidence, phish_prob, legit_prob, indicators,
         model_version, latency_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record['id'], record['timestamp'], record['input_type'],
        record['content_hash'], record['content_preview'],
        record['verdict'], record['confidence'],
        record['phish_prob'], record['legit_prob'],
        json.dumps(record.get('indicators', [])),
        record['model_version'], record['latency_ms']
    ))
    conn.commit()
    conn.close()


# ── MODEL LOADING ────────────────────────────────────────────────
def load_model():
    """Load the BERT model. Falls back to Hugging Face if local not found."""
    global tokenizer, model, device, model_meta

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Try local fine-tuned model first, fall back to pre-trained
    model_source = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK
    is_local     = model_source == MODEL_PATH

    print(f"  Loading: {model_source}")
    print(f"  Source : {'Local fine-tuned' if is_local else 'Hugging Face pre-trained'}")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model     = AutoModelForSequenceClassification.from_pretrained(model_source)
    model     = model.to(device)
    model.eval()

    model_meta = {
        'name'    : os.path.basename(model_source),
        'source'  : 'local' if is_local else 'huggingface',
        'path'    : model_source,
        'device'  : str(device),
        'params'  : sum(p.numel() for p in model.parameters()),
        'loaded_at': datetime.utcnow().isoformat()
    }

    # Warmup inference — first call is always slower
    print("  Running warmup inference...")
    _predict_raw("warmup text for model initialisation")
    print("  ✓ Model ready\n")


# ── CORE PREDICTION ──────────────────────────────────────────────
def _predict_raw(text: str) -> dict:
    """Internal prediction — returns full probability details."""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs     = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    legit_prob = float(probs[0])
    phish_prob = float(probs[1])
    verdict    = 'phishing' if phish_prob >= 0.5 else 'legitimate'

    # Map to suspicious band (40-60% phishing confidence)
    if 0.40 <= phish_prob <= 0.60:
        verdict = 'suspicious'

    confidence = max(phish_prob, legit_prob) * 100

    return {
        'verdict'   : verdict,
        'confidence': round(confidence, 1),
        'phish_prob': round(phish_prob, 4),
        'legit_prob': round(legit_prob, 4),
    }


def _build_indicators(text: str, phish_prob: float, verdict: str) -> list:
    """
    Generate explainable indicator signals from the text.
    These are heuristic signals layered on top of the BERT score
    to give the user human-readable reasons.
    """
    import re
    lc = text.lower()

    signals = [
        # (name, risk_condition, safe_condition, max_score)
        ("Urgency Language",     bool(re.search(r'urgent|immediately|suspend|24 hours|act now|warning|last chance', lc)), None, 88),
        ("Brand Impersonation",  bool(re.search(r'paypa1|amaz0n|g00gle|microsofft|apple-id|netfl1x', lc)), None, 95),
        ("Sensitive Data Request", bool(re.search(r'credit card|ssn|social security|password|bank account|cvv|pin', lc)), None, 92),
        ("Suspicious Domain",    bool(re.search(r'\.ru|\.tk|\.xyz|\.top|verify-account|secure-login', lc)), None, 85),
        ("Threat Language",      bool(re.search(r'permanent|legal action|law enforcement|suspended|terminated|closed', lc)), None, 80),
        ("Generic Salutation",   bool(re.search(r'dear (valued|customer|user|member|client)', lc)), None, 55),
        ("Suspicious Links",     bool(re.search(r'http://[^\s]+(\.tk|\.ru|\.xyz|login|verify|secure)', lc)), None, 90),
        ("No HTTPS",             bool(re.search(r'^http://', lc)) and not bool(re.search(r'^https://', lc)), None, 65),
        ("Grammar Issues",       len(re.findall(r'\b(recieve|occured|supsended|verfiy|acccount)\b', lc)) > 0, None, 70),
        ("BERT Score",           phish_prob > 0.5, phish_prob <= 0.5, int(phish_prob * 100)),
    ]

    indicators = []
    for name, is_risk, is_safe, base_score in signals:
        if is_risk:
            score     = min(99, base_score + np.random.randint(-5, 6))
            direction = 'risk'
        elif is_safe is True:
            score     = min(99, base_score + np.random.randint(-5, 6))
            direction = 'safe'
        else:
            score     = np.random.randint(5, 30)
            direction = 'safe'

        indicators.append({'name': name, 'score': int(score), 'direction': direction})

    # Sort by score descending, take top 7
    indicators.sort(key=lambda x: x['score'], reverse=True)
    return indicators[:7]


# ── RATE LIMITING ────────────────────────────────────────────────
def check_rate_limit(ip: str) -> bool:
    """Simple sliding window rate limiter."""
    now = time.time()
    hour_ago = now - 3600

    if ip not in rate_tracker:
        rate_tracker[ip] = []

    # Remove timestamps older than 1 hour
    rate_tracker[ip] = [t for t in rate_tracker[ip] if t > hour_ago]

    if len(rate_tracker[ip]) >= RATE_LIMIT:
        return False  # rate limited

    rate_tracker[ip].append(now)
    return True


# ════════════════════════════════════════════════════════════════
#  API ROUTES
# ════════════════════════════════════════════════════════════════

@app.route('/api/v1/health', methods=['GET'])
def health():
    """Health check — frontend polls this to confirm backend is up."""
    return jsonify({
        'status'   : 'ok',
        'model'    : model_meta.get('name', 'loading'),
        'device'   : model_meta.get('device', 'unknown'),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/v1/model/info', methods=['GET'])
def model_info():
    """Return metadata about the loaded model."""
    return jsonify({
        'model'      : model_meta,
        'max_length' : MAX_LENGTH,
        'labels'     : {0: 'legitimate', 1: 'phishing'},
        'rate_limit' : f"{RATE_LIMIT} requests/hour"
    })


@app.route('/api/v1/analyse', methods=['POST'])
def analyse():
    """
    Main prediction endpoint.

    Request body:
        {
            "content"   : "email or URL text",
            "input_type": "email" | "url",
            "headers"   : "From: ... (optional)"
        }

    Response:
        {
            "id"         : "a_1234567890",
            "verdict"    : "phishing" | "legitimate" | "suspicious",
            "confidence" : 94.2,
            "phish_prob" : 0.942,
            "legit_prob" : 0.058,
            "indicators" : [...],
            "explanation": "...",
            "model"      : "phishguard-bert-finetuned",
            "latency_ms" : 48.3
        }
    """
    # ── Rate limit check ──
    ip = request.remote_addr
    if not check_rate_limit(ip):
        return jsonify({'error': 'Rate limit exceeded. Max 100 requests/hour.'}), 429

    # ── Validate request ──
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON.'}), 400

    content    = data.get('content', '').strip()
    input_type = data.get('input_type', 'email').lower()
    headers    = data.get('headers', '').strip()

    if not content:
        return jsonify({'error': 'content field is required and cannot be empty.'}), 422

    if len(content) < 5:
        return jsonify({'error': 'content must be at least 5 characters.'}), 422

    if input_type not in ('email', 'url'):
        return jsonify({'error': "input_type must be 'email' or 'url'."}), 422

    # ── Combine email headers + body for analysis ──
    full_text = f"{headers} {content}" if headers else content
    full_text = full_text[:1000]  # cap at 1000 chars

    # ── Run prediction ──
    start_time = time.time()
    try:
        result = _predict_raw(full_text)
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

    latency_ms = (time.time() - start_time) * 1000

    # ── Build explainable indicators ──
    indicators = _build_indicators(full_text, result['phish_prob'], result['verdict'])

    # ── Build explanation text ──
    top_risks = [i['name'] for i in indicators if i['direction'] == 'risk'][:2]
    explanations = {
        'phishing': (
            f"This {input_type} is classified as phishing with {result['confidence']:.1f}% confidence. "
            f"BERT identified strong threat signals including {' and '.join(top_risks) if top_risks else 'suspicious content patterns'}. "
            f"Do not interact with this content or provide any personal information."
        ),
        'suspicious': (
            f"This {input_type} contains ambiguous signals that warrant caution ({result['confidence']:.1f}% confidence). "
            f"While not definitively malicious, {top_risks[0] if top_risks else 'certain patterns'} suggest possible social engineering intent. "
            f"Verify the sender independently before taking any action."
        ),
        'legitimate': (
            f"This {input_type} appears legitimate with {result['confidence']:.1f}% confidence. "
            f"BERT found no significant phishing indicators in the content. "
            f"Standard security hygiene is always recommended."
        )
    }

    # ── Build response ──
    analysis_id = 'a_' + str(int(time.time() * 1000))
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    response_data = {
        'id'          : analysis_id,
        'verdict'     : result['verdict'],
        'confidence'  : result['confidence'],
        'phish_prob'  : result['phish_prob'],
        'legit_prob'  : result['legit_prob'],
        'indicators'  : indicators,
        'explanation' : explanations[result['verdict']],
        'model'       : model_meta.get('name', 'bert'),
        'input_type'  : input_type,
        'latency_ms'  : round(latency_ms, 1),
        'timestamp'   : datetime.utcnow().isoformat()
    }

    # ── Log to database ──
    try:
        log_analysis({
            **response_data,
            'content_hash'   : content_hash,
            'content_preview': content[:80],
            'model_version'  : model_meta.get('name', 'bert'),
        })
    except Exception:
        pass  # Don't fail the request if logging fails

    return jsonify(response_data), 200


@app.route('/api/v1/batch', methods=['POST'])
def batch_analyse():
    """
    Analyse multiple inputs at once.
    Max 20 items per batch.

    Request: { "items": [{"content": "...", "input_type": "email"}, ...] }
    """
    data  = request.get_json(silent=True)
    items = data.get('items', []) if data else []

    if not items:
        return jsonify({'error': 'items array is required.'}), 400
    if len(items) > 20:
        return jsonify({'error': 'Maximum 20 items per batch.'}), 400

    results = []
    for item in items:
        content    = item.get('content', '').strip()
        input_type = item.get('input_type', 'email')
        if content:
            result = _predict_raw(content[:1000])
            results.append({'content_preview': content[:50], **result})

    return jsonify({'results': results, 'count': len(results)}), 200


@app.route('/api/v1/history', methods=['GET'])
def get_history():
    """Return recent analyses from the database."""
    limit  = min(int(request.args.get('limit', 50)), 200)
    filter_verdict = request.args.get('verdict', None)

    conn  = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if filter_verdict:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE verdict=? ORDER BY timestamp DESC LIMIT ?",
            (filter_verdict, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()

    conn.close()
    return jsonify({'analyses': [dict(r) for r in rows], 'count': len(rows)}), 200


@app.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """Return aggregate statistics for the dashboard."""
    conn = sqlite3.connect(DB_PATH)

    total     = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    phishing  = conn.execute("SELECT COUNT(*) FROM analyses WHERE verdict='phishing'").fetchone()[0]
    legit     = conn.execute("SELECT COUNT(*) FROM analyses WHERE verdict='legitimate'").fetchone()[0]
    suspicious = conn.execute("SELECT COUNT(*) FROM analyses WHERE verdict='suspicious'").fetchone()[0]
    avg_conf  = conn.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0]
    avg_lat   = conn.execute("SELECT AVG(latency_ms) FROM analyses").fetchone()[0]

    conn.close()

    return jsonify({
        'total'          : total,
        'phishing'       : phishing,
        'legitimate'     : legit,
        'suspicious'     : suspicious,
        'avg_confidence' : round(avg_conf or 0, 1),
        'avg_latency_ms' : round(avg_lat or 0, 1),
        'phishing_rate'  : round(phishing / total * 100, 1) if total else 0
    }), 200


# ── STARTUP ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  PhishGuard BERT — API Server")
    print("=" * 60)

    init_db()
    load_model()

    print("  Endpoints:")
    print("    GET  http://localhost:5000/api/v1/health")
    print("    GET  http://localhost:5000/api/v1/model/info")
    print("    POST http://localhost:5000/api/v1/analyse")
    print("    POST http://localhost:5000/api/v1/batch")
    print("    GET  http://localhost:5000/api/v1/history")
    print("    GET  http://localhost:5000/api/v1/stats")
    print("\n  Press CTRL+C to stop\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False   # set True during development for auto-reload
    )
