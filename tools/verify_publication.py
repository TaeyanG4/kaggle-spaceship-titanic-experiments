"""Check the publication artifacts (standard library only; no data, models or network).

Checks: evidence files parse; every file in `submissions/` matches its manifest hash, has the
official format and appears in the Kaggle submission snapshot; ledger discovery IDs exist in
`discoveries.md`; diagram hashes match `docs/diagrams/manifest.json`; relative Markdown links and
images resolve; the Kaggle notebook and its metadata are consistent; and no tracked text file looks
like it contains a credential. It does not certify that a model is leak-free or significant.
Run: `python tools/verify_publication.py`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
problems: list[str] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def check_evidence() -> None:
    for name in ["data-fingerprints.json", "environment-snapshot.json", "final-stackers.json",
                 "submission-manifest.json"]:
        json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "submission-manifest.json").read_text(encoding="utf-8"))
    with open(EVIDENCE / "kaggle-submissions.csv", encoding="utf-8") as handle:
        kaggle = {row["fileName"]: row for row in csv.DictReader(handle)}
    files = sorted(p.name for p in (ROOT / "submissions").glob("*.csv"))
    check(sorted(manifest) == files, f"manifest {sorted(manifest)} != submissions/ {files}")
    for name in files:
        path = ROOT / "submissions" / name
        check(manifest.get(name, {}).get("sha256") == sha256(path), f"hash mismatch: {name}")
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        check(rows[0] == ["PassengerId", "Transported"], f"bad header: {name}")
        check(len(rows) == 4278, f"{name}: {len(rows) - 1} rows, expected 4277")
        check({r[1] for r in rows[1:]} <= {"True", "False"}, f"non-boolean labels: {name}")
        check(name in kaggle and kaggle[name]["publicScore"] != "", f"no Kaggle score: {name}")
    check(len(kaggle) == 12, f"expected 12 submissions in the snapshot, found {len(kaggle)}")


def check_ledger() -> None:
    discoveries = set(re.findall(r"^## (D-[A-Z]+-\d+) ", (ROOT / "discoveries.md")
                                 .read_text(encoding="utf-8"), flags=re.MULTILINE))
    with open(EVIDENCE / "experiment-ledger.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            check(row["discovery"] in discoveries, f"ledger cites unknown {row['discovery']}")
            float(row["mean_delta"])


def check_diagrams() -> None:
    manifest = json.loads((ROOT / "docs/diagrams/manifest.json").read_text(encoding="utf-8"))
    for name, meta in manifest["diagrams"].items():
        check(meta["source_sha256"] == sha256(ROOT / f"docs/diagrams/{name}.dot"),
              f"diagram source changed without re-render: {name}")
        check(meta["svg_sha256"] == sha256(ROOT / f"docs/assets/{name}.svg"), f"svg hash: {name}")
        check(meta["png_sha256"] == sha256(ROOT / f"docs/assets/{name}.png"), f"png hash: {name}")


def check_links() -> None:
    pages = [ROOT / "README.md", ROOT / "README.en.md", ROOT / "NOTICE.md",
             ROOT / "submissions/README.md", ROOT / "data/README.md",
             ROOT / "kaggle_notebook/README.md", *sorted((ROOT / "docs").glob("*.md"))]
    pattern = re.compile(r"\]\(([^)\s]+)\)|<img[^>]+src=\"([^\"]+)\"")
    for page in pages:
        for match in pattern.finditer(page.read_text(encoding="utf-8")):
            target = match.group(1) or match.group(2)
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            path = (page.parent / target.split("#")[0]).resolve()
            check(path.exists(), f"{page.relative_to(ROOT)}: broken link {target}")


def check_notebook() -> None:
    meta = json.loads((ROOT / "kaggle_notebook/kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = ROOT / "kaggle_notebook" / meta["code_file"]
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    check(meta["competition_sources"] == ["spaceship-titanic"], "notebook competition source")
    check(any("TABPFN_TOKEN" in "".join(c["source"]) for c in nb["cells"]),
          "notebook does not read TABPFN_TOKEN from secrets")


def check_secrets() -> None:
    tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             cwd=ROOT, capture_output=True, text=True, check=False).stdout.split()
    secret = re.compile(r"(tabpfn_[A-Za-z0-9]{24,}|KGAT_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{30,}"
                        r"|\"key\"\s*:\s*\"[0-9a-f]{32}\")")
    for name in tracked:
        path = ROOT / name
        if path.suffix.lower() in {".png", ".lock"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        check(not secret.search(text), f"possible credential in {name}")


def main() -> None:
    for step in (check_evidence, check_ledger, check_diagrams, check_links, check_notebook,
                 check_secrets):
        try:
            step()
        except Exception as error:  # noqa: BLE001 - report every failing step
            problems.append(f"{step.__name__} failed: {error!r}")
    if problems:
        print("PUBLICATION CHECK FAILED")
        for problem in problems:
            print(" -", problem)
        sys.exit(1)
    print("publication check passed")


if __name__ == "__main__":
    main()
