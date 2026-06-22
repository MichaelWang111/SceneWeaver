from __future__ import annotations

import pytest

from sceneweaver.user_web import generate_script_from_payload, list_sources, safe_filename, status_payload


def test_user_web_status_payload_has_workspace():
    payload = status_payload()

    assert payload["status"] == "ok"
    assert payload["workspace_root"]
    assert payload["default_output_root"].endswith("outputs\\film_analysis") or payload["default_output_root"].endswith("outputs/film_analysis")


def test_safe_filename_strips_unsafe_characters():
    assert safe_filename("../my video 01.mp4") == "my_video_01.mp4"
    assert safe_filename("***") == "video.mp4"


def test_list_sources_returns_list():
    assert isinstance(list_sources(), list)


def test_generate_script_payload_requires_query():
    with pytest.raises(ValueError, match="query is required"):
        generate_script_from_payload({})


def test_generate_script_payload_calls_core(monkeypatch):
    called = {}

    def fake_generate_script(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return {"status": "ok", "script": {"script_markdown": "draft"}}

    monkeypatch.setattr("sceneweaver.user_web.generate_script", fake_generate_script)

    result = generate_script_from_payload(
        {
            "query": "真实团队",
            "sources": ["outputs/film_analysis"],
            "script_brief": "招聘短片",
            "top_k": 3,
            "duration_seconds": 60,
            "max_tokens": 7000,
        }
    )

    assert result["status"] == "ok"
    assert called["args"][0] == "真实团队"
    assert called["kwargs"]["script_brief"] == "招聘短片"
    assert called["kwargs"]["top_k"] == 3
    assert called["kwargs"]["duration_seconds"] == 60
    assert called["kwargs"]["max_tokens"] == 7000
