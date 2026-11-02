/**
 * ================================================================
 *  PhishGuard BERT — Day 5: Frontend Connector
 *
 *  Drop this into your PhishGuard HTML file to replace the
 *  Claude API call with your local BERT backend.
 *
 *  HOW TO USE:
 *  1. Open your PhishGuard_AI.html file
 *  2. Find the line:  async function callClaudeAPI(content, inputType, headers) {
 *  3. Replace that entire function with the code in this file
 *  4. Also update runAnalysis() to use callBERTAPI instead of callClaudeAPI
 * ================================================================
 */

// ── CONFIG ───────────────────────────────────────────────────────
const BERT_API = {
    baseURL   : 'http://localhost:5000/api/v1',
    timeout   : 10000,   // 10 second timeout
    isOnline  : false,   // updated by health check
};

// ── HEALTH CHECK ─────────────────────────────────────────────────
// Call this on page load to check if backend is running
async function checkBERTBackend() {
    try {
        const resp = await fetch(`${BERT_API.baseURL}/health`, {
            signal: AbortSignal.timeout(3000)
        });
        if (resp.ok) {
            const data = await resp.json();
            BERT_API.isOnline = true;
            BERT_API.modelName = data.model;
            BERT_API.device    = data.device;
            console.log(`✓ BERT backend online | Model: ${data.model} | Device: ${data.device}`);
            updateBackendStatus(true, data.model, data.device);
            return true;
        }
    } catch (e) {
        console.warn('BERT backend offline — using demo mode');
        BERT_API.isOnline = false;
        updateBackendStatus(false);
        return false;
    }
}

// Update the UI to show backend status in the sidebar
function updateBackendStatus(online, modelName = '', device = '') {
    // Update the active model display in settings
    const modelEl = document.getElementById('active-model-name');
    const metaEl  = document.getElementById('active-model-meta');

    if (modelEl) {
        modelEl.textContent = online
            ? `PhishGuard BERT (${modelName})`
            : 'Demo Mode (BERT offline)';
    }
    if (metaEl) {
        metaEl.textContent = online
            ? `Local BERT model · ${device.includes('cuda') ? 'GPU' : 'CPU'} · localhost:5000`
            : 'Start backend: python backend/app.py';
    }

    // Add a status indicator to the page
    const existingStatus = document.getElementById('bert-status-bar');
    if (!existingStatus) {
        const bar = document.createElement('div');
        bar.id = 'bert-status-bar';
        bar.style.cssText = `
            position: fixed; bottom: 0; left: 240px; right: 0;
            padding: 6px 24px; font-family: var(--mono); font-size: 11px;
            display: flex; align-items: center; gap: 8px;
            background: var(--bg2); border-top: 1px solid var(--border2);
            z-index: 50;
        `;
        bar.innerHTML = online
            ? `<span style="color:#00e887">●</span> BERT backend connected · ${modelName} · ${device}`
            : `<span style="color:#ff3b5c">●</span> BERT backend offline · <span style="color:var(--text3)">Run: python backend/app.py</span>`;
        document.body.appendChild(bar);
    } else {
        existingStatus.innerHTML = online
            ? `<span style="color:#00e887">●</span> BERT backend connected · ${modelName} · ${device}`
            : `<span style="color:#ff3b5c">●</span> BERT backend offline · <span style="color:var(--text3)">Run: python backend/app.py</span>`;
    }
}

// ── MAIN API CALL ────────────────────────────────────────────────
/**
 * Replace callClaudeAPI() with this function.
 * Same signature, same return shape — drop-in replacement.
 */
async function callBERTAPI(content, inputType, headers = '') {
    if (!BERT_API.isOnline) {
        throw new Error('BERT backend is offline');
    }

    const response = await fetch(`${BERT_API.baseURL}/analyse`, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({
            content,
            input_type: inputType,
            headers
        }),
        signal: AbortSignal.timeout(BERT_API.timeout)
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${response.status}`);
    }

    const data = await response.json();

    // The BERT API returns the same shape PhishGuard already expects
    // so no mapping needed — just return it
    return {
        verdict    : data.verdict,
        confidence : data.confidence,
        indicators : data.indicators,
        explanation: data.explanation,
        // Extra BERT-specific fields (displayed in results)
        phish_prob : data.phish_prob,
        legit_prob : data.legit_prob,
        latency_ms : data.latency_ms,
        model      : data.model,
    };
}

// ── UPDATED runAnalysis() ────────────────────────────────────────
/**
 * Replace the existing runAnalysis() function in PhishGuard
 * with this version. The only change is the decision tree
 * now routes to callBERTAPI instead of callClaudeAPI.
 */
async function runAnalysis_BERT() {
    const btn = document.getElementById('analyse-btn');
    btn.disabled = true;

    document.getElementById('result-empty').style.display   = 'none';
    document.getElementById('result-content').classList.remove('visible');
    document.getElementById('result-loading').style.display = 'block';

    // Animated loading lines
    ['loading-line-2','loading-line-3','loading-line-4'].forEach(id =>
        document.getElementById(id).style.opacity = '0'
    );
    setTimeout(() => document.getElementById('loading-line-2').style.opacity = '1', 300);
    setTimeout(() => document.getElementById('loading-line-3').style.opacity = '1', 700);
    setTimeout(() => document.getElementById('loading-line-4').style.opacity = '1', 1200);

    let content = '', headers = '';
    if (STATE.inputType === 'url') {
        content = document.getElementById('input-url').value.trim();
    } else {
        headers = document.getElementById('input-email-headers').value.trim();
        content = document.getElementById('input-email-body').value.trim();
    }

    let result;
    try {
        if (BERT_API.isOnline) {
            // ✅ Use local BERT model (best path)
            result = await callBERTAPI(content, STATE.inputType, headers);
            showToast('✓ Analysed with BERT model', 'success');
        } else if (STATE.apiKey) {
            // Fallback 1: Claude API (if key is set)
            result = await callClaudeAPI(content, STATE.inputType, headers);
            showToast('⚡ BERT offline — used Claude API', 'success');
        } else {
            // Fallback 2: Heuristic simulation
            result = await simulateAnalysis(content, STATE.inputType, headers);
            showToast('Demo mode — start backend for BERT', 'success');
        }
    } catch (err) {
        console.error('Analysis error:', err);
        result = await simulateAnalysis(content, STATE.inputType, headers);
        showToast('Inference error — using heuristic fallback', 'error');
    }

    document.getElementById('result-loading').style.display = 'none';
    renderResult(result);

    // Save to history
    const record = {
        id        : 'a_' + Date.now(),
        timestamp : new Date().toISOString(),
        inputType : STATE.inputType,
        content   : content.slice(0, 80) + (content.length > 80 ? '…' : ''),
        verdict   : result.verdict,
        confidence: result.confidence,
        indicators: result.indicators,
        explanation: result.explanation
    };
    STATE.analyses.unshift(record);
    persist();
    updateDashboard();

    btn.disabled = false;
}

// ── BATCH ANALYSIS ───────────────────────────────────────────────
async function batchAnalyse(items) {
    /**
     * Analyse multiple emails/URLs at once.
     * items: [{content: '...', input_type: 'email'}, ...]
     * Returns array of results.
     */
    if (!BERT_API.isOnline) {
        throw new Error('BERT backend required for batch analysis');
    }

    const response = await fetch(`${BERT_API.baseURL}/batch`, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ items })
    });

    if (!response.ok) throw new Error(`Batch failed: HTTP ${response.status}`);

    const data = await response.json();
    return data.results;
}

// ── AUTO-INIT ────────────────────────────────────────────────────
// Check backend status every 30 seconds
document.addEventListener('DOMContentLoaded', () => {
    checkBERTBackend();
    setInterval(checkBERTBackend, 30000);
});
