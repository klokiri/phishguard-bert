/*
 * PhishGuard AI for Gmail: content script (runs inside the Gmail tab).
 *
 * What it does:
 *   1. Watches the Gmail page (MutationObserver) for an email being opened.
 *   2. Reads that message's sender, subject, body text and link targets.
 *   3. If automatic checking is on (the default; it can be switched off in the
 *      toolbar popup) it sends them to the background worker, which forwards
 *      them to the PhishGuard backend on this computer. If it is off, the
 *      banner shows a "Check this email" button instead.
 *   4. Shows the backend's answer in a banner above the message.
 *
 * Gmail's page structure is not a public API. The class names in GMAIL are the
 * ones Gmail uses at the time of writing (Nov 2026) and are the ones commonly
 * used by Gmail extensions. If Gmail renames them, the extension stops finding
 * emails (nothing breaks in Gmail itself); update GMAIL to fix it.
 *
 * Security: text from the email or the backend is only ever written with
 * textContent, never as HTML. The banner lives in its own shadow DOM, so
 * Gmail's styles do not change it.
 */
(() => {
  'use strict';

  const GMAIL = {
    body: 'div.a3s',          // body of an expanded (open) message
    message: 'div.adn',       // one expanded message in a conversation
    sender: 'span.gD[email]', // sender; attributes "name" and "email"
    subject: 'h2.hP',         // conversation subject
  };
  const MAX_CHARS = 100000;   // the backend's input limit
  const MAX_LINKS = 20;       // link targets appended to the text
  const SCAN_DELAY_MS = 300;  // wait for Gmail to finish rendering

  const handled = new WeakSet(); // message bodies that already have a banner
  const cache = new Map();       // hash of the email text -> Promise of the backend answer
  let autoCheck = true;

  // ── Reading an email from Gmail's page ──────────────────────────────────────
  function isVisible(el) {
    return el.getClientRects().length > 0;
  }

  /**
   * Text sent to the model. headers: "From: Name <address>" and "Subject: ...".
   * content: the body as displayed, followed by the targets of its links, one
   * per line. In an HTML email the visible link text ("Verify now") hides the
   * real address; the address is what matters, so it is added (like the
   * plain-text version of an email shows it). Links whose address is already
   * in the text are not repeated.
   */
  function readEmail(body) {
    const message = body.closest(GMAIL.message);
    const sender = message && message.querySelector(GMAIL.sender);
    const subjectEl = document.querySelector(GMAIL.subject);
    const headers = [];
    if (sender) {
      const name = (sender.getAttribute('name') || sender.textContent || '').trim();
      headers.push(`From: ${name} <${sender.getAttribute('email')}>`);
    }
    if (subjectEl && subjectEl.textContent.trim()) headers.push(`Subject: ${subjectEl.textContent.trim()}`);

    let content = (body.innerText || '').trim();
    const links = [];
    for (const a of body.querySelectorAll('a[href]')) {
      const href = a.href;
      if (!/^https?:\/\//i.test(href) || content.includes(href) || links.includes(href)) continue;
      links.push(href);
      if (links.length >= MAX_LINKS) break;
    }
    if (links.length) content += '\n' + links.join('\n');
    return { headers: headers.join('\n'), content: content.slice(0, MAX_CHARS) };
  }

  async function sha256(text) {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
  }

  // ── Talking to the background worker ───────────────────────────────────────
  function ask(email) {
    try {
      return chrome.runtime.sendMessage({ type: 'analyse', content: email.content, headers: email.headers })
        .then(res => res || { ok: false, kind: 'extension', message: 'No answer from the extension.' })
        .catch(err => ({ ok: false, kind: 'extension', message: String((err && err.message) || err) }));
    } catch (err) { // thrown at once if the extension was reloaded since this tab was opened
      return Promise.resolve({ ok: false, kind: 'extension', message: String((err && err.message) || err) });
    }
  }

  async function check(banner, email) {
    banner.loading();
    const key = await sha256(email.headers + '\n' + email.content);
    // The same email opened again (or re-rendered by Gmail) reuses the answer,
    // so it is sent to the backend only once per page load.
    if (!cache.has(key)) cache.set(key, ask(email));
    const res = await cache.get(key);
    if (!res.ok) cache.delete(key); // errors are not remembered: "Retry" asks again
    if (res.ok) banner.result(res.result);
    else banner.error(res, () => check(banner, email));
  }

  // ── The banner ──────────────────────────────────────────────────────────────
  const STYLE = `
    :host { all: initial; display: block; margin: 0 0 12px; }
    .card { font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #202124;
            background: #ffffff; border: 1px solid #dadce0; border-left: 4px solid #80868b;
            border-radius: 8px; padding: 10px 12px; }
    .card.phishing   { border-left-color: #b3261e; }
    .card.suspicious { border-left-color: #b06000; }
    .card.legitimate { border-left-color: #137333; }
    .row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .brand { font-weight: 600; color: #3c4043; }
    .pill { font-weight: 700; font-size: 12px; letter-spacing: .04em; padding: 2px 8px; border-radius: 999px;
            background: #f1f3f4; color: #3c4043; }
    .pill.phishing   { background: #fce8e6; color: #a50e0e; }
    .pill.suspicious { background: #fef7e0; color: #8a4b00; }
    .pill.legitimate { background: #e6f4ea; color: #0d652d; }
    .muted { color: #5f6368; }
    .spacer { flex: 1; }
    .advice { margin-top: 4px; }
    .section { margin-top: 6px; }
    .label { color: #5f6368; margin-right: 4px; }
    .chip { display: inline-block; font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
            padding: 0 6px; margin: 2px 4px 0 0; border-radius: 4px; border: 1px solid #dadce0; }
    .chip.phishing   { border-color: #f4b8b3; background: #fdf1f0; }
    .chip.legitimate { border-color: #a8dab5; background: #f0f9f2; }
    button { font: inherit; cursor: pointer; border-radius: 4px; }
    .action { border: 1px solid #1a73e8; background: #1a73e8; color: #fff; padding: 3px 10px; }
    .close { border: none; background: none; color: #5f6368; font-size: 16px; line-height: 1; padding: 2px 4px; }
    details { margin-top: 6px; }
    summary { cursor: pointer; color: #1a73e8; }
    details p { margin: 6px 0 0; }
  `;

  const VERDICT = {
    phishing: ['PHISHING', 'Likely phishing. Do not click links, open attachments or reply until you have checked the sender another way.'],
    suspicious: ['SUSPICIOUS', 'The model is unsure. Be careful with links and requests for information.'],
    legitimate: ['NO PHISHING DETECTED', 'The model did not find this email phishing-like. This is an estimate, not a guarantee.'],
  };

  function formatProb(p) {
    if (!Number.isFinite(p)) return 'not available';
    if (p > 0.999) return '>99.9%';
    if (p < 0.001) return '<0.1%';
    return `${(100 * p).toFixed(1)}%`;
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function createBanner(body) {
    const host = document.createElement('div');
    host.setAttribute('data-phishguard', 'banner');
    const root = host.attachShadow({ mode: 'open' });
    root.appendChild(el('style', null, STYLE));
    const card = el('div', 'card');
    card.setAttribute('role', 'status');
    card.setAttribute('aria-live', 'polite');
    root.appendChild(card);
    body.parentNode.insertBefore(host, body);

    function header(pillText, pillClass) {
      const row = el('div', 'row');
      row.appendChild(el('span', 'brand', 'PhishGuard AI'));
      if (pillText) row.appendChild(el('span', `pill ${pillClass || ''}`, pillText));
      return row;
    }
    function closeButton(row) {
      row.appendChild(el('span', 'spacer'));
      const close = el('button', 'close', '×');
      close.title = 'Hide for this email';
      close.setAttribute('aria-label', 'Hide PhishGuard banner');
      close.addEventListener('click', () => { host.hidden = true; });
      row.appendChild(close);
    }
    function setState(state, verdict) {
      host.setAttribute('data-state', state);
      if (verdict) host.setAttribute('data-verdict', verdict); else host.removeAttribute('data-verdict');
      card.className = `card ${verdict || ''}`;
      card.replaceChildren();
    }

    return {
      idle(onCheck) {
        setState('idle');
        const row = header();
        const btn = el('button', 'action', 'Check this email');
        btn.addEventListener('click', onCheck);
        row.appendChild(btn);
        row.appendChild(el('span', 'muted', 'Automatic checking is off (toolbar popup).'));
        closeButton(row);
        card.appendChild(row);
      },
      loading() {
        setState('loading');
        const row = header();
        row.appendChild(el('span', 'muted', 'Checking this email with the local BERT model…'));
        card.appendChild(row);
      },
      result(r) {
        const verdict = VERDICT[r.verdict] ? r.verdict : 'suspicious';
        setState('done', verdict);
        const row = header(VERDICT[verdict][0], verdict);
        row.appendChild(el('span', 'muted', `Phishing probability ${formatProb(r.phishProb)} · model ${r.model}`));
        closeButton(row);
        card.appendChild(row);
        card.appendChild(el('div', 'advice', VERDICT[verdict][1]));

        const risks = r.indicators.filter(i => i.direction === 'risk').slice(0, 3);
        if (risks.length) {
          const sec = el('div', 'section');
          sec.appendChild(el('span', 'label', 'Warning signs (rule checks):'));
          sec.appendChild(el('span', null, risks.map(i => (i.evidence ? `${i.name} (${i.evidence})` : i.name)).join('; ')));
          card.appendChild(sec);
        }
        if (r.topTokens.length) {
          const sec = el('div', 'section');
          sec.appendChild(el('span', 'label', 'Words that moved the model:'));
          r.topTokens.forEach(t => {
            const chip = el('span', `chip ${t.direction}`, `${t.token} ${t.effect > 0 ? '+' : ''}${t.effect.toFixed(2)}`);
            chip.title = t.direction === 'phishing'
              ? 'Removing this word lowered the phishing score (change in log-odds)'
              : 'Removing this word raised the phishing score (change in log-odds)';
            sec.appendChild(chip);
          });
          card.appendChild(sec);
        }
        const details = el('details');
        details.appendChild(el('summary', null, 'Why?'));
        details.appendChild(el('p', null, r.explanation));
        const ms = Number.isFinite(r.latencyMs) ? ` in ${Math.round(r.latencyMs)} ms` : '';
        details.appendChild(el('p', 'muted',
          `Checked on this computer by the PhishGuard backend${ms}. Word method: ${r.attributionMethod}. ` +
          'The email text was sent only to 127.0.0.1; the backend keeps a hash of it, not the text.'));
        card.appendChild(details);
      },
      error(res, onRetry) {
        setState('error');
        const row = header('NOT CHECKED');
        const messages = {
          offline: 'The PhishGuard backend is not running. Start it with: .venv/bin/python -m backend.app',
          timeout: 'The PhishGuard backend took too long to answer.',
          rate_limited: `Rate limit reached (the backend allows 100 checks per hour).${
            res.retryAfter ? ` Try again in ${Math.ceil(res.retryAfter / 60)} min.` : ''}`,
          extension: 'PhishGuard was updated or reloaded. Reload this Gmail tab.',
        };
        row.appendChild(el('span', 'muted', messages[res.kind] || `Backend error: ${res.message}`));
        if (res.kind !== 'extension') {
          const retry = el('button', 'action', 'Retry');
          retry.addEventListener('click', onRetry);
          row.appendChild(retry);
        }
        closeButton(row);
        card.appendChild(row);
      },
    };
  }

  // ── Finding opened emails ───────────────────────────────────────────────────
  function scan() {
    for (const body of document.querySelectorAll(GMAIL.body)) {
      if (handled.has(body) || !isVisible(body) || !body.parentNode) continue;
      const email = readEmail(body);
      if (!email.content) continue; // still loading: try again on the next change
      handled.add(body);
      const banner = createBanner(body);
      if (autoCheck) check(banner, email);
      else banner.idle(() => check(banner, email));
    }
  }

  let timer = null;
  const observer = new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(scan, SCAN_DELAY_MS);
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.autoCheck) autoCheck = changes.autoCheck.newValue !== false;
  });
  chrome.storage.local.get({ autoCheck: true }).then(({ autoCheck: value }) => {
    autoCheck = value !== false;
    observer.observe(document.body, { childList: true, subtree: true });
    scan();
  });
})();
