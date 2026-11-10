// Browser test of frontend/PhishGuard_AI.html opened from disk (file://).
//
// Start the backend first (or stop it to test the fallback), then:
//   CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
//   "$CHROME" --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/pg-chrome about:blank &
//   node scripts/browser_test.mjs 9333 "file://$PWD/frontend/PhishGuard_AI.html" online   # or: offline
//
// It logs in with the page's pre-filled demo account, analyses one email and
// one URL through the normal Analyse button, and prints what the page shows
// (engine, verdict, probability, fallback note, word chips, indicators).
// Uses a throw-away Chrome profile; needs Node 22+ (built-in WebSocket).
// Usage: node scripts/browser_test.mjs <debug-port> <file-url> <online|offline>
const [port, fileUrl, scenario] = process.argv.slice(2);

const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = targets.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
ws.addEventListener('message', ev => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
});
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => {
  const res = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
  if (res.result.exceptionDetails) throw new Error(JSON.stringify(res.result.exceptionDetails));
  return res.result.result.value;
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

await send('Page.enable');
await send('Page.navigate', { url: fileUrl });
await sleep(2500);

const result = await evaluate(`(async () => {
  const wait = ms => new Promise(r => setTimeout(r, ms));
  doLogin(); await wait(800);
  const skip = [...document.querySelectorAll('button')].find(b => /SKIP/.test(b.innerText) && b.offsetParent);
  if (skip) skip.click();
  nav('analyse'); await wait(1500);
  const out = { protocol: location.protocol, origin: String(location.origin),
                connectorLoaded: typeof callBertAPI === 'function', registryCount: STATE.models.length,
                badge: document.getElementById('engine-status').innerText };
  const runs = [['email', 'URGENT: your account will be suspended. Verify your password immediately at http://192.168.10.5/login'],
                ['url', 'https://github.com/EstAlvB/Phishing-Detection-with-BERT']];
  out.runs = [];
  for (const [type, text] of runs) {
    switchInputType(type);
    document.getElementById(type === 'url' ? 'input-url' : 'input-email-body').value = text;
    document.getElementById('analyse-btn').disabled = false;
    document.getElementById('analyse-btn').click();
    await wait(${scenario === 'offline' ? 4500 : 3500});
    const r = STATE.analyses[0];
    out.runs.push({ type, engine: r.engine, verdict: r.verdict, shownEngine: document.getElementById('engine-name').textContent,
                    shownProb: document.getElementById('engine-prob').textContent,
                    note: document.getElementById('engine-note').textContent,
                    tokens: document.getElementById('engine-tokens').innerText.replace(/\\n/g, ' | '),
                    indicators: [...document.querySelectorAll('#indicators-list .indicator-name')].map(e => e.innerText.replace(/\\n/g, ' / ')).slice(0, 4) });
  }
  return out;
})()`);
console.log(JSON.stringify(result, null, 2));
ws.close();
process.exit(0);
