"""
Measure backend test coverage (NFR-09) and save the real figure.

Run:  .venv/bin/python scripts/coverage_report.py
Writes results/coverage.json.

NFR-09 (proposal): ">= 80% statement coverage for all backend service
modules, measured with pytest-cov". This runs the whole test suite with
coverage measured on the backend/ package and records the total and the
per-file numbers exactly as pytest-cov reports them.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = 80.0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "cov.json"
        cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--cov=backend",
               f"--cov-report=json:{raw_path}", "--cov-report=term-missing"]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        print(proc.stdout[-3000:])
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            print("tests failed - coverage not recorded")
            return 1
        raw = json.loads(raw_path.read_text())

    files = {path: {"statements": f["summary"]["num_statements"], "missed": f["summary"]["missing_lines"],
                    "percent": round(f["summary"]["percent_covered"], 1)}
             for path, f in sorted(raw["files"].items())}
    total = round(raw["totals"]["percent_covered"], 1)
    lowest = min(f["percent"] for f in files.values())
    summary_line = [line for line in proc.stdout.splitlines() if " passed" in line]
    report = {
        "requirement": "NFR-09: >= 80% statement coverage for backend service modules (pytest-cov)",
        "measured_package": "backend/",
        "total_percent": total,
        "lowest_module_percent": lowest,
        "target_percent": TARGET,
        "meets_target_total": total >= TARGET,
        "meets_target_every_module": lowest >= TARGET,
        "files": files,
        "tests": summary_line[-1].strip() if summary_line else "unknown",
        "tool": f"coverage.py {raw['meta']['version']} via pytest-cov",
        "command": ".venv/bin/python scripts/coverage_report.py",
    }
    (REPO_ROOT / "results" / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"total {total}% (lowest module {lowest}%), target {TARGET}% -> results/coverage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
