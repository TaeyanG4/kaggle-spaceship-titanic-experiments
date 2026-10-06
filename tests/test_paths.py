from pathlib import Path

from spaceship_titanic.paths import PROJECT_ROOT, RAW_DATA_DIR, SUBMISSION_DIR

ROOT = Path(__file__).resolve().parents[1]


def test_project_root() -> None:
    assert PROJECT_ROOT == ROOT


def test_expected_directories_exist() -> None:
    assert RAW_DATA_DIR.is_dir()
    assert SUBMISSION_DIR.is_dir()
