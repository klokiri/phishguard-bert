/*
 * PhishGuard <-> BERT backend connector.
 *
 * Loaded by PhishGuard_AI.html before its main script. It only DEFINES
 * functions; it does not replace any handler. runAnalysis() in the HTML
 * decides which engine to use:
 *   1. the local BERT backend (backend/app.py) when it is reachable,
 *   2. otherwise the app's existing path (Claude API if a key is set,
 *      else the in-browser regex heuristic).
 *
 * Works when the HTML is opened from a local web server (python -m
 * http.server) and straight from disk (file://): the backend allows both
 * origins (see backend/app.py, CORS).
 *
 * Security: everything that comes from the backend is written into the page
 * with textContent (never innerHTML), so text from a model or an analysed
 * email cannot inject HTML or scripts.
 */

const BERT_API = {
  // 127.0.0.1 rather than "localhost": the backend only listens on IPv4
  // 127.0.0.1, and "localhost" may resolve to IPv6 (::1) first.
  baseURL: 'http://127.0.0.1:5000/api/v1',
  online: false,
  model: '',
  device: '',
  healthTimeoutMs: 3000,
  analyseTimeoutMs: 20000,
};

/** Ask the backend whether it is up; update the status badge. Returns true/false. */
async function checkBertBackend() {
  try {
    const resp = await fetch(`${BERT_API.baseURL}/health`, { signal: AbortSignal.timeout(BERT_API.healthTimeoutMs) });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    BERT_API.online = data.status === 'ok';
    BERT_API.model = String(data.model || '');
    BERT_API.device = String(data.device || '');
  } catch (e) {
    BERT_API.online = false;
  }
  renderBackendStatus();
  return BERT_API.online;
}

/** Status badge on the Analyse page (id="engine-status"). */
function renderBackendStatus() {
  const el = document.getElementById('engine-status');
  if (!el) return;
  el.classList.toggle('online', BERT_API.online);
  el.classList.toggle('offline', !BERT_API.online);
  const text = el.querySelector('.engine-status-text');
  text.textContent = BERT_API.online
    ? `BERT backend online · ${BERT_API.model} · ${BERT_API.device.toUpperCase()}`
    : 'BERT backend offline · using fallback engine';
}

/** POST one item to the backend. Throws on any failure so the caller can fall back. */
async function callBertAPI(content, inputType, headers = '') {
  const resp = await fetch(`${BERT_API.baseURL}/analyse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, input_type: inputType, headers }),
    signal: AbortSignal.timeout(BERT_API.analyseTimeoutMs),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return {
    // The fields the existing UI already understands:
    verdict: data.verdict,
    confidence: data.confidence,
    indicators: Array.isArray(data.indicators) ? data.indicators : [],
    explanation: data.explanation || '',
    // Extra fields shown in the "Model details" block:
    engine: 'bert',
    model: data.model,
    phishProb: data.phish_prob,
    latencyMs: data.latency_ms,
    topTokens: Array.isArray(data.top_tokens) ? data.top_tokens : [],
    attributionMethod: data.attribution_method || '',
  };
}

const ENGINE_LABELS = {
  bert: 'Local BERT backend',
  claude: 'Claude API',
  heuristic: 'In-browser heuristic (fallback)',
};

/** Fill the "Model details" block (id="engine-details") for any engine. */
function renderEngineDetails(result) {
  const box = document.getElementById('engine-details');
  if (!box) return;
  const set = (id, value) => { document.getElementById(id).textContent = value; };

  set('engine-name', ENGINE_LABELS[result.engine] || 'Unknown engine');
  set('engine-model', result.model || '—');
  set('engine-prob', typeof result.phishProb === 'number' ? formatProbability(result.phishProb) : 'not available');
  set('engine-latency', typeof result.latencyMs === 'number' ? `${result.latencyMs} ms (server)` : '—');
  set('engine-note', result.fallbackReason || '');

  const list = document.getElementById('engine-tokens');
  list.replaceChildren();  // clear old chips
  const tokens = result.topTokens || [];
  document.getElementById('engine-tokens-block').style.display = tokens.length ? 'block' : 'none';
  tokens.forEach(t => {
    const chip = document.createElement('span');
    chip.className = 'token-chip ' + (t.direction === 'phishing' ? 'towards-phishing' : 'towards-legit');
    const sign = t.effect > 0 ? '+' : '';
    chip.textContent = `${t.token} ${sign}${Number(t.effect).toFixed(2)}`;
    chip.title = t.direction === 'phishing'
      ? 'Removing this word lowered the phishing score (log-odds change shown)'
      : 'Removing this word raised the phishing score (log-odds change shown)';
    list.appendChild(chip);
  });
  set('engine-tokens-method', result.attributionMethod ? `Method: ${result.attributionMethod}` : '');
  box.style.display = 'block';
}

/** '>99.9%' instead of a rounded '100.0%' (same rule as the backend). */
function formatProbability(p) {
  if (p > 0.999) return '>99.9%';
  if (p < 0.001) return '<0.1%';
  return `${(100 * p).toFixed(1)}%`;
}
