"""Snapshot publication evidence (run once, locally, before publishing).

Writes into `docs/evidence/`:
- `kaggle-submissions.csv`: this account's Spaceship Titanic submissions from the Kaggle CLI
  (read-only API call; needs Kaggle credentials).
- `data-fingerprints.json`: SHA256, row and column counts of the official CSVs (no data copied).
- `environment-snapshot.json`: Python, OS, GPU and key package versions.
- `submission-manifest.json`: SHA256 and positive rate of every CSV copied into `submissions/`.
Also copies the published prediction files from `outputs/submissions/` into `submissions/`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
PUBLISHED = {
    "baseline_catboost.csv": "CatBoost baseline (optimistic stopping) - first submission",
    "hm10_innercv_logloss.csv": "CatBoost with honest inner-CV Logloss iteration choice (promoted)",
    "ha12_tabpfn.csv": "TabPFN v3.5 frozen (promoted)",
    "ha27_stack3.csv": "Nested LR stack of three TabPFN v3.5 variants (promoted, official champion)",
    "sub_stack3_plus_a1ne2.csv": "Champion stack + 100-epoch fine-tune with 2 inference estimators",
    "sub_a1mix_stack.csv": "Frozen + averaged 100-epoch fine-tune variants + fine-tune-refit",
    "sub_avg4.csv": "Frozen + 5-run fine-tune average + fine-tune-refit + full-context fine-tune",
    "sub_combo7.csv": "Champion stack + four small-gain fine-tuning members (post hoc)",
    "sub_combo5.csv": "Champion stack + two 100-epoch fine-tune variants",
    "sub_stack_a1ctx.csv": "Frozen + full-context 100-epoch fine-tune + fine-tune-refit",
    "sub_combo7_bag3.csv": "sub_combo7 with test predictions bagged over three fold seeds",
    "sub_a1mix_single.csv": "Averaged 100-epoch fine-tune variants, single member",
}
PACKAGES = ["numpy", "pandas", "scikit-learn", "catboost", "xgboost", "lightgbm", "torch",
            "tabpfn", "tabicl", "pytabkit", "matplotlib"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["kaggle", "competitions", "submissions", "-c", "spaceship-titanic",
                          "--csv"], capture_output=True, text=True, check=True).stdout
    (EVIDENCE / "kaggle-submissions.csv").write_text(out.strip() + "\n", encoding="utf-8")

    fingerprints = {}
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        frame = pd.read_csv(ROOT / "data/raw" / name)
        fingerprints[name] = {"sha256": sha256(ROOT / "data/raw" / name),
                              "rows": len(frame), "columns": frame.shape[1]}
    (EVIDENCE / "data-fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n",
                                                     encoding="utf-8")

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    env = {"python": platform.python_version(), "os": platform.platform(), "gpu": gpu,
           "packages": versions,
           "note": "Main environment only; TabDPT/TabSTAR ran in a separate venv and AutoGluon "
                   "in another (see docs/05-reproduction.md)."}
    (EVIDENCE / "environment-snapshot.json").write_text(json.dumps(env, indent=2) + "\n",
                                                        encoding="utf-8")

    target = ROOT / "submissions"
    target.mkdir(exist_ok=True)
    manifest = {}
    for name, description in PUBLISHED.items():
        src = ROOT / "outputs/submissions" / name
        shutil.copyfile(src, target / name)
        frame = pd.read_csv(target / name)
        manifest[name] = {"description": description, "sha256": sha256(target / name),
                          "rows": len(frame),
                          "positive_rate": round(float(frame.Transported.mean()), 6)}
    (EVIDENCE / "submission-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    print("evidence written:", sorted(p.name for p in EVIDENCE.iterdir()))


if __name__ == "__main__":
    main()
