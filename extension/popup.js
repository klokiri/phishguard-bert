/*
 * Toolbar popup: shows whether the local backend is reachable and switches
 * automatic checking on or off (saved in chrome.storage.local as autoCheck).
 * Text is written with textContent only.
 */
const statusEl = document.getElementById('status');
const autoBox = document.getElementById('auto-check');

function showStatus(res) {
  statusEl.className = 'status ' + (res && res.ok ? 'online' : 'offline');
  statusEl.replaceChildren();
  if (res && res.ok) {
    statusEl.textContent = `Backend online · ${res.model} · ${res.device.toUpperCase()}`;
    return;
  }
  statusEl.append('Backend offline. In the project folder, start it with ');
  const code = document.createElement('code');
  code.textContent = '.venv/bin/python -m backend.app';
  statusEl.append(code);
}

chrome.runtime.sendMessage({ type: 'health' })
  .then(showStatus)
  .catch(() => showStatus(null));

chrome.storage.local.get({ autoCheck: true }).then(({ autoCheck }) => {
  autoBox.checked = autoCheck !== false;
});
autoBox.addEventListener('change', () => {
  chrome.storage.local.set({ autoCheck: autoBox.checked });
});
