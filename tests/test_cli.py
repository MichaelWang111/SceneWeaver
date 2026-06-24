from __future__ import annotations

import json
from typing import Any

from typer.testing import CliRunner

from sceneweaver.cli import app
from sceneweaver.pipeline.mock_pipeline import build_mock_artifacts
from sceneweaver.storage.json_store import write_json


def test_cli_mock_run(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "cli_mock"

    result = runner.invoke(app, ["mock-run", "--output", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "analysis" / "experience_cards.jsonl").exists()
    assert not (output_dir / "fingerprints").exists()


def test_cli_extract_experience(tmp_path):
    runner = CliRunner()
    _, _, scenes, *_ = build_mock_artifacts()
    output_dir = tmp_path / "video"
    write_json(output_dir / "analysis" / "scenes.json", scenes)

    result = runner.invoke(app, ["extract-experience", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "analysis" / "experience_cards.jsonl").exists()


def test_cli_retrieve_cards_outputs_matches(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "video"
    runner.invoke(app, ["mock-run", "--output", str(output_dir)])

    result = runner.invoke(
        app,
        [
            "retrieve-cards",
            str(output_dir),
            "招聘宣传片，年轻人作为团队一员共同创造未来",
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "query_tags" in data
    assert data["results"]
    assert data["results"][0]["matched_dimensions"]


def test_cli_keyword_loop_streams_and_thinking(monkeypatch, tmp_path):
    calls: dict[str, Any] = {}

    class FakeKeywordLoopResult:
        def model_dump(self, *, mode: str = "python") -> dict:
            return {
                "input_text": "young people running",
                "mode": "just-tags",
                "association_path": str(tmp_path / "association.json"),
                "candidate_log_path": str(tmp_path / "tag_candidates.jsonl"),
                "experience_cards_path": str(tmp_path / "experience_cards.jsonl"),
                "experience_cards_paths": [str(tmp_path / "experience_cards.jsonl")],
                "unindexed_scene_dirs": [],
                "searched_card_count": 1,
                "matched_card_count": 0,
                "semantic_enabled": True,
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "semantic_weight": 4.0,
                "intent_weight": 0.0,
                "top_matches": [],
                "association_analysis": None,
                "tag_expansion_analysis": {"input_text": "young people running"},
                "intent_analysis": None,
                "retrieval": {"query_tags": {}, "results": []},
                "next_actions": [],
            }

    def fake_keyword_loop(input_text: str, card_source, **kwargs):
        calls["input_text"] = input_text
        calls["card_source"] = card_source
        calls.update(kwargs)
        kwargs["log"]("fake keyword loop progress")
        kwargs["stream_callback"]('{"partial":')
        kwargs["stream_callback"]('"json"}')
        kwargs["reasoning_callback"]("thinking trace")
        return FakeKeywordLoopResult()

    monkeypatch.setattr("sceneweaver.cli.run_keyword_loop", fake_keyword_loop)

    result = CliRunner().invoke(
        app,
        [
            "keyword-loop",
            str(tmp_path / "film_analysis"),
            "young people running",
            "--just-tags",
            "--semantic",
            "--embedding-model",
            "BAAI/bge-small-zh-v1.5",
            "--semantic-weight",
            "4",
            "--stream",
            "--thinking",
            "--thinking-budget",
            "256",
            "--debug",
        ],
    )

    assert result.exit_code == 0
    assert calls["input_text"] == "young people running"
    assert calls["stream_callback"] is not None
    assert calls["just_tags"] is True
    assert calls["intent"] is False
    assert calls["intent_weight"] == 3.0
    assert calls["semantic"] is True
    assert calls["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert calls["semantic_weight"] == 4
    assert calls["reasoning_callback"] is not None
    assert calls["enable_thinking"] is True
    assert calls["thinking_budget"] == 256
    assert "[keyword-loop] fake keyword loop progress" in result.stderr
    assert '{"partial":"json"}' in result.stderr
    assert "thinking trace" in result.stderr
    assert json.loads(result.stdout)["input_text"] == "young people running"


def test_cli_llm_check_uses_configured_client(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self) -> None:
            self.config = type(
                "Config",
                (),
                {
                    "api_key": "test.md-key",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen3.6-plus",
                    "temperature": 0.2,
                    "max_tokens": 1800,
                    "request_timeout_seconds": 180.0,
                    "stream_idle_timeout_seconds": 10.0,
                    "enable_thinking": None,
                    "thinking_budget": None,
                },
            )()

        def analyze_text_json(self, **kwargs):
            captured.update(kwargs)
            return {"reply": "pong"}

    monkeypatch.setattr("sceneweaver.cli.VisionLLMClient", lambda: FakeClient())

    result = CliRunner().invoke(app, ["llm-check", "ping", "--timeout-seconds", "12"])

    assert result.exit_code == 0
    assert captured["user_prompt"] == "ping"
    assert captured["timeout_seconds"] == 12.0
    assert json.loads(result.stdout)["reply"] == "pong"


def test_cli_llm_status_uses_central_status_payload(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_status_payload(**kwargs):
        captured.update(kwargs)
        return {
            "method": "sceneweaver_llm_provider_status",
            "summary": {"provider": "deepseek", "api_key_configured": True},
            "status": {"provider": "deepseek"},
        }

    monkeypatch.setattr("sceneweaver.cli.llm_status_payload", fake_status_payload)

    result = CliRunner().invoke(app, ["llm-status", "--provider", "deepseek", "--include-models"])

    assert result.exit_code == 0
    assert captured["provider"] == "deepseek"
    assert captured["include_models"] is True
    assert json.loads(result.stdout)["summary"]["provider"] == "deepseek"
