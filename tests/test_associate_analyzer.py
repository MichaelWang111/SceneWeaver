from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sceneweaver.analysis.associate_analyzer import (
    associate_input,
    build_associate_user_prompt,
    load_associate_prompt,
)
from sceneweaver.cli import app, build_key_associate_output_path
from sceneweaver.schemas import AssociationAnalysis
from sceneweaver.storage.json_store import read_json, write_json


def _item(category: str, term: str) -> dict:
    return {
        "term": term,
        "category": category,
        "meaning": f"{term}的导演含义",
        "emotion": "明亮、向前",
        "image_hint": f"{term}的可拍摄画面",
        "usage_hint": f"{term}适合用于情绪推进",
    }


def _valid_association_data(input_text: str = "青年逆光奔跑") -> dict:
    categories = [
        "visual_imagery",
        "character_state",
        "action_motifs",
        "emotional_keywords",
        "narrative_seeds",
        "spatial_symbols",
        "light_color_texture",
        "copy_tone",
    ]
    return {
        "input_text": input_text,
        "query_tags": {
            "emotion_core": ["ambition", "creativity"],
            "audience_projection": ["future_builder"],
            "narrative_function": ["invitation"],
            "interaction_mode": ["team_collaboration"],
            "visual_motifs": ["silhouette", "upward_motion"],
            "symbolic_logic": ["becoming"],
            "rhythm_pattern": ["explosive_build"],
            "evidence": [
                {
                    "source_id": "query",
                    "source_type": "query",
                    "field": "input_text",
                    "quote": input_text,
                    "note": "test.md query tags",
                }
            ],
            "confidence": 0.75,
        },
        "core_reading": "一群年轻人从不确定走向共同发光。",
        "emotional_arc": {
            "origin": "最初是不确定和试探。",
            "development": "关系和动作逐渐把个体推向群体。",
            "release": "在逆光奔跑中完成情绪释放。",
            "arc_summary": "从试探到并肩，从被看见到主动发光。",
        },
        "association_count": len(categories),
        "association_map": {category: [_item(category, category)] for category in categories},
        "director_possibilities": [
            {
                "name": "晨光群像",
                "concept": "用清晨和奔跑建立年轻人的共同方向。",
                "emotional_direction": "明亮、笃定",
                "visual_direction": "逆光、低机位、宽画幅",
                "narrative_direction": "个体汇入群体",
            },
            {
                "name": "城市冲刺",
                "concept": "把挑战放进真实城市空间。",
                "emotional_direction": "紧张、兴奋",
                "visual_direction": "街道、天桥、快速剪辑",
                "narrative_direction": "从压力到突破",
            },
            {
                "name": "创意爆发",
                "concept": "用动作和视觉转场呈现创意生成。",
                "emotional_direction": "灵感、释放",
                "visual_direction": "手部细节、纸张、光斑",
                "narrative_direction": "从想法到行动",
            },
        ],
        "avoid_cliches": ["不要硬喊梦想口号"],
    }


class FakeTextClient:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.calls = []

    def analyze_text_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        retries: int = 0,
        stream_callback=None,
        reasoning_callback=None,
        enable_thinking=None,
        thinking_budget=None,
    ) -> dict:
        self.calls.append(
            (
                system_prompt,
                user_prompt,
                max_tokens,
                timeout_seconds,
                retries,
                stream_callback,
                reasoning_callback,
                enable_thinking,
                thinking_budget,
            )
        )
        if stream_callback is not None:
            stream_callback('{"streamed": true}')
        if reasoning_callback is not None:
            reasoning_callback("thinking")
        return self.data


def test_association_schema_validates_required_shape():
    analysis = AssociationAnalysis.model_validate(_valid_association_data())

    assert analysis.association_count == 8
    assert analysis.association_map.visual_imagery[0].category == "visual_imagery"


def test_association_schema_rejects_empty_input():
    data = _valid_association_data(input_text=" ")

    with pytest.raises(ValueError, match="input_text"):
        AssociationAnalysis.model_validate(data)


def test_association_schema_rejects_empty_category():
    data = _valid_association_data()
    data["association_map"]["visual_imagery"] = []
    data["association_count"] = 7

    with pytest.raises(ValueError, match="visual_imagery"):
        AssociationAnalysis.model_validate(data)


def test_association_schema_rejects_missing_item_fields():
    data = _valid_association_data()
    del data["association_map"]["visual_imagery"][0]["meaning"]

    with pytest.raises(ValueError, match="meaning"):
        AssociationAnalysis.model_validate(data)

    data = _valid_association_data()
    del data["association_map"]["visual_imagery"][0]["usage_hint"]

    with pytest.raises(ValueError, match="usage_hint"):
        AssociationAnalysis.model_validate(data)


def test_association_schema_requires_count_to_match_items():
    data = _valid_association_data()
    data["association_count"] = 9

    with pytest.raises(ValueError, match="association_count"):
        AssociationAnalysis.model_validate(data)


def test_associate_input_reads_prompt_calls_llm_and_writes_output(tmp_path):
    prompt_path = tmp_path / "associate.md"
    prompt_path.write_text("system prompt", encoding="utf-8")
    output_path = tmp_path / "association.json"
    client = FakeTextClient(_valid_association_data("青年逆光奔跑"))

    analysis = associate_input(
        "青年逆光奔跑",
        client=client,
        prompt_path=prompt_path,
        output_path=output_path,
        max_items=60,
        timeout_seconds=12,
        retries=2,
    )

    assert analysis.input_text == "青年逆光奔跑"
    assert analysis.query_tags.visual_motifs
    assert client.calls[0][0] == "system prompt"
    assert "max_items: 60" in client.calls[0][1]
    assert client.calls[0][3] == 12
    assert client.calls[0][4] == 2
    assert read_json(output_path, AssociationAnalysis).input_text == "青年逆光奔跑"


def test_associate_input_passes_stream_callback(tmp_path):
    streamed_chunks = []
    client = FakeTextClient(_valid_association_data("青年逆光奔跑"))

    associate_input(
        "青年逆光奔跑",
        client=client,
        output_path=tmp_path / "association.json",
        stream_callback=streamed_chunks.append,
    )

    assert client.calls[0][5] is not None
    assert streamed_chunks == ['{"streamed": true}', "\n"]


def test_associate_input_passes_thinking_options(tmp_path):
    reasoning_chunks = []
    client = FakeTextClient(_valid_association_data("闈掑勾閫嗗厜濂旇窇"))

    associate_input(
        "闈掑勾閫嗗厜濂旇窇",
        client=client,
        output_path=tmp_path / "association.json",
        reasoning_callback=reasoning_chunks.append,
        enable_thinking=True,
        thinking_budget=512,
    )

    assert client.calls[0][6] is not None
    assert client.calls[0][7] is True
    assert client.calls[0][8] == 512
    assert reasoning_chunks == ["thinking"]


def test_default_associate_prompt_exists():
    prompt = load_associate_prompt()

    assert "导演前期创意联想器" in prompt


def test_build_associate_user_prompt_includes_input_and_max_items():
    prompt = build_associate_user_prompt("青春 / 逆光", max_items=64)

    assert "青春 / 逆光" in prompt
    assert "max_items: 64" in prompt


def test_associate_cli_accepts_chinese_input_output_and_max_items(monkeypatch, tmp_path):
    calls = {}

    def fake_associate_input(input_text: str, **kwargs):
        calls["input_text"] = input_text
        calls.update(kwargs)
        analysis = AssociationAnalysis.model_validate(_valid_association_data(input_text))
        output_path: Path | None = kwargs["output_path"]
        if output_path is not None:
            write_json(output_path, analysis)
        return analysis

    monkeypatch.setattr("sceneweaver.cli.associate_input", fake_associate_input)
    output_path = tmp_path / "association.json"

    result = CliRunner().invoke(
        app,
        [
            "associate",
            "我需要一群年轻人逆光奔跑",
            "--output",
            str(output_path),
            "--max-items",
            "60",
            "--timeout-seconds",
            "30",
            "--retries",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert calls["input_text"] == "我需要一群年轻人逆光奔跑"
    assert calls["max_items"] == 60
    assert calls["timeout_seconds"] == 30
    assert calls["retries"] == 2
    assert output_path.exists()
    rendered = json.loads(result.stdout)
    assert rendered["input_text"] == "我需要一群年轻人逆光奔跑"


def test_associate_cli_debug_prints_progress_to_stderr(monkeypatch, tmp_path):
    def fake_associate_input(input_text: str, **kwargs):
        kwargs["log"]("fake progress")
        analysis = AssociationAnalysis.model_validate(_valid_association_data(input_text))
        write_json(kwargs["output_path"], analysis)
        return analysis

    monkeypatch.setattr("sceneweaver.cli.associate_input", fake_associate_input)
    output_path = tmp_path / "association.json"

    result = CliRunner().invoke(
        app,
        [
            "associate",
            "我需要一群年轻人逆光奔跑",
            "--output",
            str(output_path),
            "--debug",
        ],
    )

    assert result.exit_code == 0
    assert "Associate debug:" in result.stderr
    assert "[associate] fake progress" in result.stderr


def test_associate_cli_streams_raw_chunks_to_stderr(monkeypatch, tmp_path):
    def fake_associate_input(input_text: str, **kwargs):
        kwargs["log"]("fake stream progress")
        kwargs["stream_callback"]('{"partial":')
        kwargs["stream_callback"]('"json"}')
        analysis = AssociationAnalysis.model_validate(_valid_association_data(input_text))
        write_json(kwargs["output_path"], analysis)
        return analysis

    monkeypatch.setattr("sceneweaver.cli.associate_input", fake_associate_input)
    output_path = tmp_path / "association.json"

    result = CliRunner().invoke(
        app,
        [
            "associate",
            "我需要一群年轻人逆光奔跑",
            "--output",
            str(output_path),
            "--stream",
        ],
    )

    assert result.exit_code == 0
    assert "[associate] fake stream progress" in result.stderr
    assert '{"partial":"json"}' in result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["input_text"] == "我需要一群年轻人逆光奔跑"


def test_associate_cli_streams_thinking_to_stderr(monkeypatch, tmp_path):
    calls = {}

    def fake_associate_input(input_text: str, **kwargs):
        calls.update(kwargs)
        kwargs["reasoning_callback"]("thinking trace")
        analysis = AssociationAnalysis.model_validate(_valid_association_data(input_text))
        write_json(kwargs["output_path"], analysis)
        return analysis

    monkeypatch.setattr("sceneweaver.cli.associate_input", fake_associate_input)
    output_path = tmp_path / "association.json"

    result = CliRunner().invoke(
        app,
        [
            "associate",
            "鎴戦渶瑕佷竴缇ゅ勾杞讳汉閫嗗厜濂旇窇",
            "--output",
            str(output_path),
            "--thinking",
            "--thinking-budget",
            "512",
        ],
    )

    assert result.exit_code == 0
    assert calls["enable_thinking"] is True
    assert calls["thinking_budget"] == 512
    assert "thinking trace" in result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["input_text"] == "鎴戦渶瑕佷竴缇ゅ勾杞讳汉閫嗗厜濂旇窇"


def test_associate_cli_writes_to_default_key_associates_dir(monkeypatch, tmp_path):
    calls = {}
    default_output = tmp_path / "outputs" / "key_associates" / "associate.json"

    def fake_default_output_path(input_text: str) -> Path:
        calls["default_input_text"] = input_text
        return default_output

    def fake_associate_input(input_text: str, **kwargs):
        calls["input_text"] = input_text
        calls.update(kwargs)
        analysis = AssociationAnalysis.model_validate(_valid_association_data(input_text))
        write_json(kwargs["output_path"], analysis)
        return analysis

    monkeypatch.setattr("sceneweaver.cli.build_key_associate_output_path", fake_default_output_path)
    monkeypatch.setattr("sceneweaver.cli.associate_input", fake_associate_input)

    result = CliRunner().invoke(app, ["associate", "中文输入"])

    assert result.exit_code == 0
    assert calls["default_input_text"] == "中文输入"
    assert calls["output_path"] == default_output
    assert default_output.exists()


def test_build_key_associate_output_path_uses_safe_default_for_chinese_input():
    output_path = build_key_associate_output_path("年轻人逆光奔跑")

    assert output_path.parent.as_posix().endswith("outputs/key_associates")
    assert output_path.name.endswith(".json")
    assert "_associate_" in output_path.name
