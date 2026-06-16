from __future__ import annotations

from pathlib import Path

PACKAGE_NAME = "retrieval_lab"
PRODUCT_NAME = "Retrieval Lab"
DEFAULT_ARTIFACT_ROOT = Path(".tmp") / "retrieval_lab"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_paths() -> dict[str, str]:
    root = project_root()
    return {
        "product_name": PRODUCT_NAME,
        "package_name": PACKAGE_NAME,
        "project_root": str(root),
        "package_root": str(root / "src" / PACKAGE_NAME),
        "legacy_baseline_package": "mocktesting",
        "default_artifact_root": str(root / DEFAULT_ARTIFACT_ROOT),
        "legacy_tmp_root": str(root / ".tmp"),
    }
