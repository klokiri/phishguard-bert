"""
Build the out-of-distribution (OOD) email test set.

Run:  .venv/bin/python -I scripts/build_ood.py
  (-I: isolated mode, so nothing in the downloaded data can be imported as code)
Writes data/ood/ood.parquet (git-ignored: email text is never committed) and
results/ood_manifest.json (sources, licences, hashes, counts; committed).

Why: every score so far comes from one dataset. An OOD set from DIFFERENT
public sources shows how well the models transfer to data collected in
another way.

Sources (none is part of ealvaradob/phishing-dataset):
  phishing   Jose Nazario's phishing corpus, years 2023 and 2024
             https://monkey.org/~jose/phishing/  (CC BY 4.0, attribution required)
             Real phishing emails, hand-classified by the collector.
  legitimate SpamAssassin public corpus, easy_ham and hard_ham (2002-2003)
             https://spamassassin.apache.org/old/publiccorpus/
             (copyright stays with the original senders; distributed for
             testing filters). hard_ham = legitimate mail that looks spammy.

Each email becomes one text: "Subject: ...\n" + its body. The body is the
text/plain part if there is one, otherwise the HTML part with tags removed.
Then: empty texts dropped, exact duplicates dropped, and any text that also
appears (after lower-casing and collapsing whitespace) in our train/val/test
splits is dropped, so the OOD set cannot overlap the original data.

Known bias, reported with the results: the legitimate emails are ~20 years
older than the phishing emails, so a mixed accuracy could be inflated by
"era" cues. The results are therefore reported per source: recall on the
phishing emails and false-positive rate on the legitimate ones.
"""

import email
import email.policy
import hashlib
import json
import mailbox
import re
import tarfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "ood" / "raw"
OUT = REPO_ROOT / "data" / "ood" / "ood.parquet"
MANIFEST = REPO_ROOT / "results" / "ood_manifest.json"

SOURCES = {
    "nazario_2023": ("https://monkey.org/~jose/phishing/phishing-2023", "mbox", 1,
                     "Nazario phishing corpus 2023", "CC BY 4.0"),
    "nazario_2024": ("https://monkey.org/~jose/phishing/phishing-2024", "mbox", 1,
                     "Nazario phishing corpus 2024", "CC BY 4.0"),
    "sa_easy_ham": ("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2", "tar", 0,
                    "SpamAssassin public corpus, easy_ham (2003-02-28)", "copyright with original senders"),
    "sa_hard_ham": ("https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2", "tar", 0,
                    "SpamAssassin public corpus, hard_ham (2003-02-28)", "copyright with original senders"),
}
MAX_BODY_CHARS = 100_000  # same input limit as the backend


class _TextOnly(HTMLParser):
    """Collect visible text from HTML, skipping <script> and <style>."""

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextOnly()
    try:
        parser.feed(html)
    except Exception:  # malformed HTML: keep what was parsed
        pass
    return re.sub(r"[ \t\r\f\v]+", " ", " ".join(parser.parts)).strip()


def decode_part(part) -> str:
    payload = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:  # unknown charset name in the header
        return payload.decode("utf-8", errors="replace")


def message_to_text(msg) -> str:
    plain, html = [], []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue  # containers and attachments
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(decode_part(part))
        elif ctype == "text/html":
            html.append(decode_part(part))
    body = "\n".join(plain) if any(p.strip() for p in plain) else html_to_text("\n".join(html))
    subject = str(msg.get("Subject", "") or "").strip()
    text = (f"Subject: {subject}\n" if subject else "") + body.strip()
    return text[:MAX_BODY_CHARS]


def download(name: str, url: str) -> Path:
    folder = RAW / name  # one folder per source
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / url.rsplit("/", 1)[-1]
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "phishguard-capstone-research"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
            f.write(resp.read())
    return path


def read_messages(path: Path, kind: str):
    if kind == "mbox":
        for msg in mailbox.mbox(str(path), factory=lambda f: email.message_from_binary_file(f, policy=email.policy.compat32)):
            yield msg
    else:
        # Read members from the archive in memory; nothing is extracted to disk,
        # so file names inside the archive cannot write anywhere.
        with tarfile.open(path, "r:bz2") as tar:
            for member in tar.getmembers():
                if member.isfile() and not member.name.endswith("cmds"):
                    yield email.message_from_bytes(tar.extractfile(member).read(), policy=email.policy.compat32)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows, files = [], {}
    for name, (url, kind, label, description, licence) in SOURCES.items():
        path = download(name, url)
        files[name] = {"url": url, "description": description, "licence": licence,
                       "bytes": path.stat().st_size, "sha256": sha256(path), "label": label}
        n = 0
        for msg in read_messages(path, kind):
            rows.append({"source": name, "label": label, "text": message_to_text(msg)})
            n += 1
        files[name]["messages"] = n
        print(f"{name:14} {n:5} messages  ({path.stat().st_size:,} bytes)")

    df = pd.DataFrame(rows)
    counts = {"parsed": len(df)}
    df = df[df["text"].str.strip().str.len() > 0]
    counts["after_dropping_empty"] = len(df)
    df = df.drop_duplicates("text")
    counts["after_dropping_duplicates"] = len(df)

    # Remove anything that also appears in the original dataset splits.
    seen = set()
    for split in ("train", "val", "test"):
        seen |= set(pd.read_parquet(REPO_ROOT / "data" / "splits" / f"{split}.parquet", columns=["text"])["text"].map(normalise))
    overlap = df["text"].map(normalise).isin(seen)
    counts["overlapping_with_original_dataset_removed"] = int(overlap.sum())
    df = df[~overlap].reset_index(drop=True)
    counts["final"] = len(df)

    df.insert(0, "row_id", range(len(df)))
    df["type_guess"] = df["source"]  # src/evaluate.py reports metrics per type_guess -> per source
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    manifest = {
        "purpose": "Out-of-distribution email test set (sources not in ealvaradob/phishing-dataset)",
        "files": files,
        "counts": counts,
        "final_by_source": {k: int(v) for k, v in df["source"].value_counts().sort_index().items()},
        "final_by_label": {"phishing": int((df["label"] == 1).sum()), "legitimate": int((df["label"] == 0).sum())},
        "text_construction": "'Subject: ...' line + text/plain body (or HTML body with tags removed); max 100,000 chars",
        "overlap_check": "lower-cased, whitespace-collapsed exact match against train/val/test texts",
        "content_sha256": hashlib.sha256("\n".join(f"{r.row_id}\t{r.label}\t{r.text}" for r in df.itertuples()).encode()).hexdigest(),
        "note": "Email text is not committed (copyright of senders; recent phishing mails are not anonymised).",
        "command": ".venv/bin/python -I scripts/build_ood.py",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "final_by_source", "final_by_label")}, indent=2))


if __name__ == "__main__":
    main()
