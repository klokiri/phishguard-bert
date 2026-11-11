// Browser test of the Gmail extension (extension/) in a real, headless Chrome.
//
// Chrome only runs the content script on pages that match the manifest
// (https://mail.google.com/*), and a test cannot sign in to Gmail. So this
// script loads a COPY of the extension whose manifest also matches the local
// test page tests/extension/gmail_fixture.html, which copies the parts of
// Gmail's page structure that content.js reads. All code is unchanged; only
// that one match pattern is added. A live Gmail check is done by hand (README).
//
// Run (needs Node 22+ and Google Chrome; uses a throw-away Chrome profile):
//   .venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory tests/extension &
//   start the backend (PHISHGUARD_DB=<scratch file> keeps test checks out of
//     data/analyses.db), or leave it stopped for the offline scenario, then
//   node scripts/extension_test.mjs online    # or: offline
// Prints JSON with what the banner and the popup showed, writes it to
// results/extension_test_<scenario>.json, and exits 1 if a check fails.
import { execSync, spawn } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scenario = process.argv[2] || 'online';
const REPO = join(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const FIXTURE_ORIGIN = 'http://127.0.0.1:8765';
const BACKEND = 'http://127.0.0.1:5000/api/v1';
const SCREENSHOT = process.env.SCREENSHOT || '';

// 1. Copy the extension and add the fixture's address to the content-script matches.
const work = mkdtempSync(join(tmpdir(), 'pg-ext-test-'));
const extDir = join(work, 'extension');
cpSync(join(REPO, 'extension'), extDir, { recursive: true });
const manifest = JSON.parse(readFileSync(join(extDir, 'manifest.json'), 'utf8'));
manifest.content_scripts[0].matches.push(`${FIXTURE_ORIGIN}/*`);
writeFileSync(join(extDir, 'manifest.json'), JSON.stringify(manifest, null, 2));

// 2. Start Chrome with the DevTools protocol on a pipe (needed to load an unpacked extension).
const chrome = spawn(CHROME, [
  '--headless=new', '--remote-debugging-pipe', '--enable-unsafe-extension-debugging',
  `--user-data-dir=${join(work, 'profile')}`, '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });
const toChrome = chrome.stdio[3];
const fromChrome = chrome.stdio[4];
fromChrome.setEncoding('utf8');
let id = 0;
let buffer = '';
const pending = new Map();
fromChrome.on('data', chunk => {
  buffer += chunk;
  let end;
  while ((end = buffer.indexOf('\0')) >= 0) {
    const msg = JSON.parse(buffer.slice(0, end));
    buffer = buffer.slice(end + 1);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  }
});
const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
  const i = ++id;
  pending.set(i, msg => (msg.error ? reject(new Error(`${method}: ${msg.error.message}`)) : resolve(msg.result)));
  toChrome.write(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }) + '\0');
});
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function openTab(url) {
  const { targetId } = await send('Target.createTarget', { url });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Runtime.enable', {}, sessionId);
  const evaluate = async expr => {
    const res = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }, sessionId);
    if (res.exceptionDetails) throw new Error(JSON.stringify(res.exceptionDetails));
    return res.result.value;
  };
  return { sessionId, evaluate };
}

async function waitFor(evaluate, expr, timeoutMs = 45000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const value = await evaluate(expr);
    if (value) return value;
    await sleep(250);
  }
  return null;
}

async function stats() {
  try { return await (await fetch(`${BACKEND}/stats`)).json(); } catch { return null; }
}

// What the visible banner shows, once it has left the "loading" state.
const BANNER = `(() => {
  const host = document.querySelector('#view [data-phishguard]');
  if (!host || host.getAttribute('data-state') === 'loading') return null;
  const card = host.shadowRoot.querySelector('.card');
  return { state: host.getAttribute('data-state'), verdict: host.getAttribute('data-verdict'),
           shown: card.innerText.split('\\n').filter(Boolean),
           explanation: (host.shadowRoot.querySelector('details p') || {}).textContent || null,
           placedAboveBody: !!(host.nextElementSibling && host.nextElementSibling.classList.contains('a3s')) };
})()`;

const checks = [];
const expect = (name, ok, detail) => checks.push({ check: name, pass: !!ok, ...(detail !== undefined ? { detail } : {}) });
const out = { scenario };

try {
  out.chrome = (await send('Browser.getVersion')).product;
  // The real, unmodified extension/ folder must load too (its manifest is the one users get).
  const { id: realId } = await send('Extensions.loadUnpacked', { path: join(REPO, 'extension') });
  expect('unmodified extension/ loads in Chrome', !!realId);
  const { id: extensionId } = await send('Extensions.loadUnpacked', { path: extDir });
  out.extensionLoaded = !!extensionId;

  const page = await openTab(`${FIXTURE_ORIGIN}/gmail_fixture.html`);
  await waitFor(page.evaluate, `document.readyState === 'complete'`);
  await sleep(1500);
  expect('no banner before an email is opened', (await page.evaluate(`document.querySelectorAll('[data-phishguard]').length`)) === 0);

  if (scenario === 'offline') {
    await page.evaluate('openEmail(1)');
    out.email1 = await waitFor(page.evaluate, BANNER);
    expect('offline: banner says the backend is not running', out.email1 && out.email1.state === 'error'
      && out.email1.shown.join(' ').includes('backend is not running'));
  } else {
    const s0 = await stats();
    await page.evaluate('openEmail(1)');
    out.email1 = await waitFor(page.evaluate, BANNER);
    if (SCREENSHOT) {
      const shot = await send('Page.captureScreenshot', { format: 'png' }, page.sessionId);
      writeFileSync(SCREENSHOT, Buffer.from(shot.data, 'base64'));
    }
    await page.evaluate('openEmail(2)');
    out.email2 = await waitFor(page.evaluate, BANNER);
    const s1 = await stats();
    await page.evaluate('openEmail(1)');
    out.email1Again = await waitFor(page.evaluate, BANNER);
    await sleep(1000);
    const s2 = await stats();
    out.backendChecks = { firstTwoEmails: s1.total - s0.total, reopeningEmail1: s2.total - s1.total };

    expect('email 1: result shown above the message body', out.email1 && out.email1.state === 'done' && out.email1.placedAboveBody);
    expect('email 2: result shown', out.email2 && out.email2.state === 'done');
    expect('reopened email shows the same verdict', out.email1Again && out.email1Again.verdict === out.email1.verdict);
    expect('each email sent to the backend once; reopening uses the saved answer',
      out.backendChecks.firstTwoEmails === 2 && out.backendChecks.reopeningEmail1 === 0, out.backendChecks);
  }
  expect('hidden message body is ignored', (await page.evaluate(
    `!document.getElementById('hidden-body').previousElementSibling`)) === true);

  // The toolbar popup: backend status, then switch automatic checking off.
  const popup = await openTab(`chrome-extension://${extensionId}/popup.html`);
  out.popupStatus = await waitFor(popup.evaluate,
    `(() => { const el = document.getElementById('status'); // null until the popup page has loaded
              return !el || /Checking/.test(el.innerText) ? null : el.innerText; })()`);
  expect('popup shows the backend status', scenario === 'offline'
    ? /offline/.test(out.popupStatus || '') : /online/.test(out.popupStatus || ''), out.popupStatus);

  if (scenario !== 'offline') {
    await popup.evaluate(`document.getElementById('auto-check').click()`);
    await sleep(500);
    const s3 = await stats();
    await page.evaluate('openEmail(3)');
    out.email3Idle = await waitFor(page.evaluate, BANNER);
    await sleep(1000);
    expect('auto-check off: button shown, nothing sent', out.email3Idle && out.email3Idle.state === 'idle'
      && (await stats()).total === s3.total);
    await page.evaluate(`document.querySelector('#view [data-phishguard]').shadowRoot.querySelector('button.action').click()`);
    out.email3AfterClick = await waitFor(page.evaluate, BANNER.replace(`=== 'loading'`, `!== 'done'`));
    expect('auto-check off: clicking "Check this email" shows a result', out.email3AfterClick && out.email3AfterClick.state === 'done');
    await popup.evaluate(`document.getElementById('auto-check').click()`); // back on (the profile is deleted anyway)
  }
} catch (err) {
  expect('test ran without errors', false, String(err.message || err));
} finally {
  out.checks = checks;
  out.passed = checks.filter(c => c.pass).length;
  out.total = checks.length;
  out.allPassed = checks.every(c => c.pass);
  // Same commit stamp as src/evaluate.py: "+dirty" if the tree had uncommitted changes.
  const git = cmd => execSync(cmd, { cwd: REPO, encoding: 'utf8' }).trim();
  out.git_commit = git('git rev-parse --short HEAD') + (git('git status --porcelain') ? '+dirty' : '');
  out.generated_at = new Date().toISOString();
  out.command = `node scripts/extension_test.mjs ${scenario}`;
  writeFileSync(join(REPO, 'results', `extension_test_${scenario}.json`), JSON.stringify(out, null, 2) + '\n');
  console.log(JSON.stringify(out, null, 2));
  chrome.kill();
  await sleep(500);
  rmSync(work, { recursive: true, force: true });
  process.exit(out.allPassed ? 0 : 1);
}
