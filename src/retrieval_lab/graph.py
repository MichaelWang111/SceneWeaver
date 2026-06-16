from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, write_json
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases


DEFAULT_SCENE_GRAPH_MANIFEST = Path(".tmp") / "retrieval_lab" / "scene_graph_manifest.json"
DEFAULT_SCENE_GRAPH_REPORT = Path(".tmp") / "retrieval_lab" / "scene_graph_report.json"


def build_scene_graph_manifest_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    cases = read_cases(
        Path(getattr(args, "dataset", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "test")),
        limit=int(getattr(args, "limit", 0)),
    )
    graph = scene_graph_from_cases(cases)
    output = Path(getattr(args, "output", DEFAULT_SCENE_GRAPH_MANIFEST))
    write_json(output, graph)
    summary = graph_summary(graph)
    summary.update({"output": str(output), "elapsed_seconds": round(time.perf_counter() - started, 3)})
    report = {
        "method": "retrieval_lab_scene_graph_manifest",
        "summary": summary,
        "manifest": str(output),
        "node_type_counts": dict(sorted(Counter(node["type"] for node in graph["nodes"]).items())),
        "edge_type_counts": dict(sorted(Counter(edge["type"] for edge in graph["edges"]).items())),
        "fingerprint": graph["fingerprint"],
    }
    report_output = Path(getattr(args, "report_output", DEFAULT_SCENE_GRAPH_REPORT))
    write_json(report_output, report)
    return {"method": report["method"], "output": str(output), "summary": summary}


def scene_graph_from_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    purpose_pairs = Counter()
    style_conflicts = Counter()
    for case in cases:
        card = target_for_case(case)
        card_id = item_id_for_card(card, fallback=str(case.get("case_id", "")))
        case_id = str(case.get("case_id", ""))
        add_node(nodes, card_id, "card", label=str(card.get("title", card_id)), case_id=case_id)
        scene_id = str(card.get("scene_id", "") or card.get("retrieval_id", "") or card_id)
        if scene_id:
            add_node(nodes, scene_id, "scene", label=scene_id)
            add_edge(edges, card_id, scene_id, "card_to_scene")
        add_node(nodes, case_id, "video_or_case", label=case_id)
        add_edge(edges, card_id, case_id, "same_case_scene")
        stage = str(card.get("script_stage", ""))
        if stage:
            stage_id = f"stage::{stage}"
            add_node(nodes, stage_id, "stage", label=stage)
            add_edge(edges, card_id, stage_id, "card_to_stage")
        purposes = [str(value) for value in card.get("creative_purpose", []) or []]
        for purpose in purposes:
            purpose_id = f"purpose::{purpose}"
            add_node(nodes, purpose_id, "purpose", label=purpose)
            add_edge(edges, card_id, purpose_id, "card_to_purpose")
        for left_index, left in enumerate(purposes):
            for right in purposes[left_index + 1 :]:
                purpose_pairs[tuple(sorted((left, right)))] += 1
        for risk in card.get("style_risks", []) or []:
            risk_id = f"style_risk::{risk}"
            add_node(nodes, risk_id, "style_risk", label=str(risk))
            add_edge(edges, card_id, risk_id, "card_to_style_risk")
            style_conflicts[(str(risk), str(card.get("style", "")))] += 1
        style = str(card.get("style", ""))
        if style:
            style_id = f"style_trait::{style}"
            add_node(nodes, style_id, "style_trait", label=style)
            add_edge(edges, card_id, style_id, "card_to_style_trait")
        signature = card.get("scene_signature", {}) if isinstance(card.get("scene_signature"), dict) else {}
        for key, values in signature.items():
            value_list = values if isinstance(values, list) else [values]
            for value in value_list:
                token = str(value)
                if not token:
                    continue
                node_id = f"signature_{key}::{token}"
                add_node(nodes, node_id, f"signature_{key}", label=token)
                add_edge(edges, card_id, node_id, "card_to_signature_token")
    for (left, right), weight in purpose_pairs.items():
        add_edge(edges, f"purpose::{left}", f"purpose::{right}", "purpose_co_occurs_with_purpose", weight=weight)
    for (risk, style), weight in style_conflicts.items():
        if risk and style:
            add_edge(edges, f"style_risk::{risk}", f"style_trait::{style}", "style_risk_conflicts_with_positive_style", weight=weight)
    graph = {
        "method": "retrieval_lab_scene_graph_manifest",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "nodes": sorted(nodes.values(), key=lambda row: row["id"]),
        "edges": sorted(edges.values(), key=lambda row: (row["source"], row["target"], row["type"])),
    }
    graph["fingerprint"] = data_sha256({"nodes": graph["nodes"], "edges": graph["edges"]})
    return graph


def target_for_case(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_prefer")
    if isinstance(expected, dict):
        return expected
    target = case.get("target")
    return target if isinstance(target, dict) else {}


def item_id_for_card(card: dict[str, Any], *, fallback: str) -> str:
    fixture = str(card.get("fixture_id", ""))
    scene = str(card.get("scene_id", ""))
    retrieval = str(card.get("retrieval_id", ""))
    parts = [part for part in (fixture, scene, retrieval) if part]
    return "::".join(parts) if parts else fallback


def add_node(nodes: dict[str, dict[str, Any]], node_id: str, node_type: str, **attrs: Any) -> None:
    if not node_id:
        return
    nodes.setdefault(node_id, {"id": node_id, "type": node_type, **attrs})


def add_edge(edges: dict[tuple[str, str, str], dict[str, Any]], source: str, target: str, edge_type: str, *, weight: int = 1) -> None:
    if not source or not target:
        return
    key = (source, target, edge_type)
    if key not in edges:
        edges[key] = {"source": source, "target": target, "type": edge_type, "weight": weight}
    else:
        edges[key]["weight"] = int(edges[key].get("weight", 1)) + weight


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_count": len({node.get("type") for node in nodes}),
        "edge_type_count": len({edge.get("type") for edge in edges}),
    }


__all__ = ["DEFAULT_SCENE_GRAPH_MANIFEST", "DEFAULT_SCENE_GRAPH_REPORT", "build_scene_graph_manifest_command", "scene_graph_from_cases"]
