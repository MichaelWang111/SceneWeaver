from __future__ import annotations

import json

from sceneweaver.analysis.tags import add_tags_to_scene_raw, build_query_tags
from sceneweaver.analysis.taxonomy import TAXONOMY_PATH, TagNormalizer, load_taxonomy
from sceneweaver.schemas import SceneAnalysis
from sceneweaver.split import scene_detector
from sceneweaver.split.scene_detector import SceneSpan


def test_taxonomy_json_is_runtime_source():
    taxonomy = load_taxonomy()

    assert TAXONOMY_PATH.exists()
    assert any(entry.tag == "direct_address" for entry in taxonomy["interaction_mode"])
    assert "direct_address" in TagNormalizer(taxonomy).tags_from_text("direct_address")["interaction_mode"]


def test_query_tags_logs_unmanaged_terms_to_candidate_pool(tmp_path):
    candidate_log = tmp_path / "analysis" / "tag_candidates.jsonl"

    tags = build_query_tags("obsidian latency ritual", candidate_log_path=candidate_log)

    assert tags.symbolic_logic == ["general_expression"]
    rows = [json.loads(line) for line in candidate_log.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["source_type"] for row in rows} == {"query"}
    assert "obsidian" in {row["normalized"] for row in rows}


def test_scene_tags_override_llm_injected_tags_and_log_candidates(tmp_path):
    raw = {
        "scene_id": "scene_001",
        "time_range": {
            "start": "00:00:00.000",
            "end": "00:00:03.000",
            "duration_seconds": 3.0,
        },
        "visual_observation": {
            "setting": "warehouse with obsidian latency ritual",
            "characters": "one presenter",
            "action_change": "standing still",
            "composition": "medium shot",
            "lighting": "soft",
            "color": "neutral",
            "camera_motion": "locked",
            "confidence_notes": "clear frame",
        },
        "director_interpretation": {
            "narrative_function": "plain setup",
            "emotional_function": "calm",
            "brand_personality_signal": "restrained",
            "underlying_emotion": "quiet clarity",
            "audience_projection": "viewer",
            "shooting_techniques": ["medium shot"],
            "why_it_works": "obsidian latency ritual creates a new expression",
        },
        "experience_candidates": [
            {
                "keywords": ["obsidian latency"],
                "emotion": "quiet",
                "narrative_logic": "hold attention",
                "techniques": ["locked shot"],
                "reuse_condition": "new expression test",
            }
        ],
        "emotion_temperature": 0.4,
        "tags": {
            "emotion_core": ["llm_injected"],
            "audience_projection": ["llm_injected"],
            "narrative_function": ["llm_injected"],
            "interaction_mode": ["llm_injected"],
            "visual_motifs": ["llm_injected"],
            "symbolic_logic": ["llm_injected"],
            "rhythm_pattern": ["llm_injected"],
            "evidence": [],
            "confidence": 0.99,
        },
    }
    candidate_log = tmp_path / "analysis" / "tag_candidates.jsonl"

    updated = add_tags_to_scene_raw(raw, candidate_log_path=candidate_log)
    scene = SceneAnalysis.model_validate(updated)

    for values in (
        scene.tags.emotion_core,
        scene.tags.audience_projection,
        scene.tags.narrative_function,
        scene.tags.interaction_mode,
        scene.tags.visual_motifs,
        scene.tags.symbolic_logic,
        scene.tags.rhythm_pattern,
    ):
        assert "llm_injected" not in values
    rows = [json.loads(line) for line in candidate_log.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["source_type"] for row in rows} == {"scene"}


def test_detect_scenes_does_not_create_scenes_dir_when_split_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        scene_detector,
        "_detect_scene_spans",
        lambda video_path, threshold: [SceneSpan("scene_001", 1, 0.0, 3.0)],
    )
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)

    scenes_dir = tmp_path / "scenes"

    spans = scene_detector.detect_scenes(
        tmp_path / "video.mp4",
        scenes_dir,
        split_video=False,
        runner=fake_runner,
    )

    assert spans[0].scene_id == "scene_001"
    assert not scenes_dir.exists()
    assert calls == []
