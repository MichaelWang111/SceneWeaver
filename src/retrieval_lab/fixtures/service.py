from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_lab.datasets.service import DEFAULT_DATASET_PATH, fixture_inventory, read_cases


def inspect_fixtures(path: Path = DEFAULT_DATASET_PATH, *, split: str = "all", limit: int = 0) -> dict[str, Any]:
    cases = read_cases(path, split=split, limit=limit)
    fixtures = fixture_inventory(cases)
    return {
        "method": "retrieval_lab_fixture_inventory",
        "source_path": str(path),
        "split": split,
        "limit": limit,
        "summary": {
            "case_count": len(cases),
            "fixture_count": len(fixtures),
            "max_cases_per_fixture": max([row["case_count"] for row in fixtures] or [0]),
        },
        "fixtures": fixtures,
    }


__all__ = ["inspect_fixtures"]
