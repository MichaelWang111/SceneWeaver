from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

from retrieval_lab.artifacts import file_sha256, read_json


VALID_SPLITS = {"dev", "test", "hidden", "all"}
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "mocktesting" / "eval_inputs" / "review_generated_inputs.json"


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    data = read_json(Path(path))
    if not isinstance(data, dict):
        raise ValueError("dataset JSON must be an object")
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("dataset JSON must contain a cases list")
    return data


def read_cases(path: Path = DEFAULT_DATASET_PATH, *, split: str = "all", limit: int = 0) -> list[dict[str, Any]]:
    cases = split_cases(load_dataset(path)["cases"], split)
    if limit > 0:
        return cases[:limit]
    return cases


def split_cases(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "all":
        return list(cases)
    return [case for case in cases if case_split(str(case.get("case_id", ""))) == split]


def case_split(case_id: str) -> str:
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket <= 3:
        return "dev"
    if bucket <= 7:
        return "test"
    return "hidden"


def case_fixture_id(case: dict[str, Any]) -> str:
    target = case.get("target", {}) if isinstance(case.get("target"), dict) else {}
    return str(target.get("fixture_id", ""))


def dataset_manifest(path: Path = DEFAULT_DATASET_PATH, *, split: str = "all", limit: int = 0) -> dict[str, Any]:
    path = Path(path)
    data = load_dataset(path)
    all_cases = list(data.get("cases", []))
    selected = read_cases(path, split=split, limit=limit)
    fixtures = fixture_inventory(selected)
    summary = dataset_summary(all_cases, selected, split=split, limit=limit)
    return {
        "method": "retrieval_lab_dataset_manifest",
        "dataset_id": data.get("dataset_id", path.stem),
        "source_layer": data.get("source_layer", ""),
        "source_path": str(path),
        "source_size_bytes": path.stat().st_size if path.exists() else 0,
        "source_sha256": file_sha256(path) if path.exists() else "",
        "split": split,
        "limit": limit,
        "summary": summary,
        "schema": dataset_schema(all_cases),
        "case_type_counts": dict(sorted(Counter(str(case.get("case_type", "")) for case in selected).items())),
        "difficulty_counts": dict(sorted(Counter(str(case.get("difficulty", "")) for case in selected).items())),
        "expected_relation_counts": dict(sorted(Counter(str(case.get("expected_relation", "")) for case in selected).items())),
        "stage_counts": dict(sorted(Counter(target_value(case, "script_stage") for case in selected).items())),
        "industry_counts": dict(sorted(Counter(target_value(case, "industry") for case in selected).items())),
        "style_counts": dict(sorted(Counter(target_value(case, "style") for case in selected).items())),
        "split_counts": split_counts(all_cases),
        "fixtures": fixtures,
        "sample_cases": [compact_case(case) for case in selected[:10]],
    }


def dataset_summary(all_cases: list[dict[str, Any]], selected: list[dict[str, Any]], *, split: str, limit: int) -> dict[str, Any]:
    fixture_ids = {case_fixture_id(case) for case in selected if case_fixture_id(case)}
    target_ids = {target_item_id(case) for case in selected if target_item_id(case)}
    return {
        "total_case_count": len(all_cases),
        "selected_case_count": len(selected),
        "split": split,
        "limit": limit,
        "fixture_count": len(fixture_ids),
        "target_scene_count": len(target_ids),
        "case_type_count": len({str(case.get("case_type", "")) for case in selected}),
        "has_generation_policy": bool(all_cases),
    }


def dataset_schema(cases: list[dict[str, Any]]) -> dict[str, Any]:
    top_fields = sorted({field for case in cases for field in case.keys()})
    target_fields = sorted(
        {
            field
            for case in cases
            if isinstance(case.get("target"), dict)
            for field in case["target"].keys()
        }
    )
    return {
        "top_level_fields": top_fields,
        "target_fields": target_fields,
        "required_top_level_fields_present": all(
            {"case_id", "case_type", "user_input", "expected_relation", "target"} <= set(case.keys())
            for case in cases
        ),
    }


def fixture_inventory(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fixture: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        fixture_id = case_fixture_id(case)
        if fixture_id:
            by_fixture.setdefault(fixture_id, []).append(case)
    rows = []
    for fixture_id, fixture_cases in sorted(by_fixture.items()):
        rows.append(
            {
                "fixture_id": fixture_id,
                "case_count": len(fixture_cases),
                "target_scene_count": len({target_value(case, "scene_id") for case in fixture_cases}),
                "case_type_counts": dict(sorted(Counter(str(case.get("case_type", "")) for case in fixture_cases).items())),
                "split_counts": dict(sorted(Counter(case_split(str(case.get("case_id", ""))) for case in fixture_cases).items())),
                "script_stage_counts": dict(sorted(Counter(target_value(case, "script_stage") for case in fixture_cases).items())),
                "industries": sorted({target_value(case, "industry") for case in fixture_cases if target_value(case, "industry")}),
                "styles": sorted({target_value(case, "style") for case in fixture_cases if target_value(case, "style")}),
            }
        )
    return rows


def split_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(case_split(str(case.get("case_id", ""))) for case in cases).items()))


def compact_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "split": case_split(str(case.get("case_id", ""))),
        "case_type": case.get("case_type"),
        "difficulty": case.get("difficulty"),
        "expected_relation": case.get("expected_relation"),
        "fixture_id": case_fixture_id(case),
        "target_scene_id": target_value(case, "scene_id"),
        "target_stage": target_value(case, "script_stage"),
        "target_purposes": target_value(case, "creative_purpose"),
    }


def target_value(case: dict[str, Any], key: str) -> Any:
    target = case.get("target", {}) if isinstance(case.get("target"), dict) else {}
    value = target.get(key, "")
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return str(value)


def target_item_id(case: dict[str, Any]) -> str:
    fixture_id = case_fixture_id(case)
    scene_id = target_value(case, "scene_id")
    return f"{fixture_id}::{scene_id}" if fixture_id and scene_id else ""


__all__ = [
    "DEFAULT_DATASET_PATH",
    "VALID_SPLITS",
    "case_fixture_id",
    "case_split",
    "compact_case",
    "dataset_manifest",
    "fixture_inventory",
    "load_dataset",
    "read_cases",
    "split_cases",
]
