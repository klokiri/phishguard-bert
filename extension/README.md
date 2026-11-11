# PhishGuard AI for Gmail (Chrome extension)

Checks each email you open in Gmail with the PhishGuard BERT model running on your own computer, and shows the result in a banner above the message.

## Install (once)

1. Start the backend from the project folder:
   ```bash
   .venv/bin/python -m backend.app
   ```
2. In Chrome, open `chrome://extensions`, switch on **Developer mode** (top right), click **Load unpacked** and choose this `extension/` folder.
3. Open Gmail (or reload the tab if it was already open) and open an email.

After changing any file in this folder, click the reload icon on the extension's card in `chrome://extensions`, then reload the Gmail tab.

## What it does

- When an email is opened, it reads the sender (`From: name <address>`), the subject, the body as displayed, and the targets of the links in the body. It then sends them to `http://127.0.0.1:5000/api/v1/analyse`. In an HTML email the visible link text ("Verify now") can hide the real address, so the addresses are added to the text, one per line.
- The banner shows:
  - the verdict and the model's phishing probability;
  - the rule-based warning signs that fired;
  - the words that moved the model's score (leave-one-word-out, in log-odds);
  - under **Why?**, the backend's full explanation.
- Each email is sent once per page load. Opening it again reuses the saved answer.
- The toolbar popup shows whether the backend is reachable. It also switches automatic checking off, in which case the banner shows a **Check this email** button instead.
- If the backend is not running, the banner says so and shows the start command. Gmail itself is unaffected.

## Privacy

Email text goes only to the backend on this computer (127.0.0.1). The address is fixed in `background.js`. The backend stores a SHA-256 hash of each checked email, not its text.

The extension asks Chrome for:
- `storage`, to remember the automatic-checking setting;
- access to `http://127.0.0.1:5000/*`, to reach the backend;
- a content script on `https://mail.google.com/*`, to read the open email.

## Files

| File | Role |
|---|---|
| `manifest.json` | Chrome extension manifest (Manifest V3) |
| `content.js` | Runs in the Gmail tab: finds opened emails, reads them, draws the banner |
| `background.js` | Service worker: the only part that uses the network; talks to the local backend |
| `popup.html`, `popup.js` | Toolbar popup: backend status and the automatic-checking switch |
| `icons/` | Toolbar icons, drawn by `scripts/make_extension_icons.py` |

## Limitations

- **Gmail's page structure is not a public API.** `content.js` finds emails by Gmail's class names (`GMAIL` at the top of the file). If Gmail renames them, the banner stops appearing and Gmail keeps working; update `GMAIL` to fix it.
- **The sender line is always included.** So the backend's "Contains an email address" check fires for every email.
- **Long emails are truncated.** BERT reads only the first 512 tokens, so in a long email the appended link addresses may not be read by the model.
- **Rate limit.** The backend allows 100 checks per hour. With automatic checking on, opening many emails quickly can reach it; the banner then says when to retry.
- **A model estimate, not a guarantee.** The out-of-distribution test (report §5b) shows that the model misses some modern phishing.

## Testing

`scripts/extension_test.mjs` loads the extension into a headless Chrome and opens `tests/extension/gmail_fixture.html`, a local page that copies the parts of Gmail's page structure that `content.js` reads. It also checks the popup and the automatic-checking switch, with the backend running and with it stopped. See that script's header for the commands. Results: `results/extension_test_online.json`, `results/extension_test_offline.json`.

The test cannot sign in to Gmail, so check live Gmail by hand after installing.
