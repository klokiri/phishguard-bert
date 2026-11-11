/*
 * PhishGuard AI for Gmail: background service worker.
 *
 * This is the only part of the extension that uses the network, and it talks
 * to exactly one place: the PhishGuard backend on this computer (BACKEND).
 * The Gmail content script cannot call http://127.0.0.1 itself (Gmail is an
 * HTTPS page, so Chrome blocks the request as mixed content), so it sends the
 * email here and this worker forwards it. The address is fixed in code; no
 * web page or email can change where the text is sent.
 *
 * Messages it accepts (only from this extension's own scripts):
 *   {type: 'health'}                      -> {ok, model, device} or {ok: false, kind, message}
 *   {type: 'analyse', content, headers}   -> {ok, result} or {ok: false, kind, message, retryAfter}
 */

const BACKEND = 'http://127.0.0.1:5000/api/v1';
const MAX_CONTENT_CHARS = 100000; // the backend's limit (max_content_chars in src/config.py)
const MAX_HEADER_CHARS = 2000;
const HEALTH_TIMEOUT_MS = 3000;
const ANALYSE_TIMEOUT_MS = 30000;

function networkError(err) {
  return err && err.name === 'TimeoutError'
    ? { ok: false, kind: 'timeout', message: 'The PhishGuard backend took too long to answer.' }
    : { ok: false, kind: 'offline', message: 'The PhishGuard backend is not reachable at 127.0.0.1:5000.' };
}

async function health() {
  try {
    const resp = await fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok || data.status !== 'ok') return { ok: false, kind: 'http', message: `HTTP ${resp.status}` };
    return { ok: true, model: String(data.model || ''), device: String(data.device || '') };
  } catch (err) {
    return networkError(err);
  }
}

/** Keep only the fields the banner shows, with the expected types. */
function pick(data) {
  const list = v => (Array.isArray(v) ? v : []);
  return {
    verdict: String(data.verdict || ''),
    phishProb: Number(data.phish_prob),
    confidence: Number(data.confidence),
    indicators: list(data.indicators).map(i => ({
      name: String(i.name || ''), evidence: String(i.evidence || ''), score: Number(i.score) || 0,
      direction: i.direction === 'safe' ? 'safe' : 'risk', basis: String(i.basis || ''),
    })),
    topTokens: list(data.top_tokens).map(t => ({
      token: String(t.token || ''), effect: Number(t.effect) || 0,
      direction: t.direction === 'phishing' ? 'phishing' : 'legitimate',
    })),
    explanation: String(data.explanation || ''),
    model: String(data.model || ''),
    latencyMs: Number(data.latency_ms),
    attributionMethod: String(data.attribution_method || ''),
  };
}

async function analyse(content, headers) {
  const body = JSON.stringify({
    content: String(content || '').slice(0, MAX_CONTENT_CHARS),
    input_type: 'email',
    headers: String(headers || '').slice(0, MAX_HEADER_CHARS),
  });
  let resp;
  try {
    resp = await fetch(`${BACKEND}/analyse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: AbortSignal.timeout(ANALYSE_TIMEOUT_MS),
    });
  } catch (err) {
    return networkError(err);
  }
  const data = await resp.json().catch(() => ({}));
  if (resp.status === 429) {
    return { ok: false, kind: 'rate_limited', message: String(data.error || 'Rate limit reached.'),
             retryAfter: Number(resp.headers.get('Retry-After')) || null };
  }
  if (!resp.ok) return { ok: false, kind: 'http', message: String(data.error || `HTTP ${resp.status}`) };
  return { ok: true, result: pick(data) };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Only this extension's own content script and popup may use the worker.
  if (sender.id !== chrome.runtime.id || !msg) return false;
  if (msg.type === 'health') {
    health().then(sendResponse);
    return true; // the answer is sent asynchronously
  }
  if (msg.type === 'analyse') {
    analyse(msg.content, msg.headers).then(sendResponse);
    return true;
  }
  return false;
});
