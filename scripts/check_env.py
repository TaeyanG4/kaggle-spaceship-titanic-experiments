import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "catboost",
    "lightgbm",
    "xgboost",
    "optuna",
    "jupyterlab",
]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Project: {project_root}")
    print()

    missing_packages: list[str] = []
    for package in PACKAGES:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            missing_packages.append(package)
            print(f"{package}: MISSING")

    print()
    raw_dir = project_root / "data" / "raw"
    expected = ["train.csv", "test.csv", "sample_submission.csv"]
    missing_data = [name for name in expected if not (raw_dir / name).exists()]
    if missing_data:
        print("Kaggle data: not ready")
        print("Missing:", ", ".join(missing_data))
        print("Join/accept the competition rules, then download the official files.")
    else:
        print("Kaggle data: ready")

    if missing_packages:
        print()
        print("Environment check FAILED. Run: uv sync")
        return 1

    print()
    print("Environment check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

