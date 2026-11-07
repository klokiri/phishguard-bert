"""
Prove the Python heuristic port gives the same answers as the original JS.

Run:  .venv/bin/python scripts/check_heuristic_parity.py     (needs Node.js)

How: it cuts the simulateAnalysis() function out of frontend/PhishGuard_AI.html
as-is, makes two mechanical edits so it can run in Node (remove the artificial
delay; also return the internal "normalised" score), runs it on every test
row, and compares score and verdict with src/models/heuristic.py.
Writes results/heuristic_parity.json.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import load_test  # noqa: E402
from src.models.heuristic import analyse  # noqa: E402

HTML = REPO_ROOT / "frontend" / "PhishGuard_AI.html"


def extract_js_function() -> str:
    html = HTML.read_text(encoding="utf-8")
    start = html.index("async function simulateAnalysis(")
    end = html.index("function renderResult(", start)
    # Cut back to the end of simulateAnalysis (the closing "}" before the
    # RENDER RESULT comment banner).
    src = html[start:end]
    src = src[: src.rindex("}") + 1]
    delay = "await new Promise(r => setTimeout(r, 1800 + Math.random() * 800));"
    assert delay in src, "delay line not found - has the HTML changed?"
    src = src.replace(delay, "")
    assert "return { verdict," in src
    src = src.replace("return { verdict,", "return { normalised, verdict,")
    return src


def input_type_for(type_guess: str) -> str:
    # The web app only has "url" and "email" modes; HTML and SMS go to "email".
    return "url" if type_guess == "url" else "email"


def main() -> int:
    test = load_test()
    items = [{"c": t, "t": input_type_for(g)} for t, g in zip(test["text"], test["type_guess"])]

    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "items.json"
        out_path = Path(tmp) / "out.json"
        js_path = Path(tmp) / "run.js"
        data_path.write_text(json.dumps(items))
        js_path.write_text(
            extract_js_function()
            + """
const fs = require('fs');
(async () => {
  const items = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const out = [];
  for (const it of items) {
    const r = await simulateAnalysis(it.c, it.t, '');
    out.push([r.normalised, r.verdict]);
  }
  fs.writeFileSync(process.argv[3], JSON.stringify(out));
})();
"""
        )
        subprocess.run(["node", str(js_path), str(data_path), str(out_path)], check=True)
        js_results = json.loads(out_path.read_text())

    mismatches = []
    for i, (item, (js_norm, js_verdict)) in enumerate(zip(items, js_results)):
        py = analyse(item["c"], item["t"])
        if abs(py["normalised"] - js_norm) > 1e-9 or py["verdict"] != js_verdict:
            mismatches.append({"row_id": int(test["row_id"].iloc[i]), "js": [js_norm, js_verdict],
                               "py": [py["normalised"], py["verdict"]]})

    report = {
        "rows_compared": len(items),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:20],
        "method": "simulateAnalysis() extracted from frontend/PhishGuard_AI.html, run in Node.js "
                  "(delay removed, normalised score returned) vs src/models/heuristic.py",
    }
    (REPO_ROOT / "results" / "heuristic_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"compared {len(items):,} rows, mismatches: {len(mismatches)}")
    for m in mismatches[:5]:
        print("  ", m)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
