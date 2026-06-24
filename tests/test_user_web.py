from __future__ import annotations

import pytest

from sceneweaver.user_web import (
    INGEST_JOBS,
    INGEST_JOBS_LOCK,
    IngestJob,
    cancel_ingest_job,
    generate_script_from_payload,
    ingest_job_payload,
    ingest_video_from_payload,
    list_sources,
    ping_llm_from_payload,
    safe_filename,
    script_agent_from_payload,
    search_scenes_from_payload,
    start_ingest_job,
    status_payload,
)


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


def test_search_payload_preserves_zero_top_k(monkeypatch):
    called = {}

    def fake_search_scenes(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return {"status": "ok", "matches": []}

    monkeypatch.setattr("sceneweaver.user_web.search_scenes", fake_search_scenes)

    result = search_scenes_from_payload({"query": "真实生活", "top_k": 0, "candidate_depth": 0})

    assert result["status"] == "ok"
    assert called["kwargs"]["top_k"] == 0
    assert called["kwargs"]["candidate_depth"] == 100


def test_ingest_payload_passes_log_callback(monkeypatch):
    called = {}
    logs = []

    def fake_ingest_video(source, **kwargs):
        called["source"] = source
        called["kwargs"] = kwargs
        kwargs["log"]("packaging")
        return {"status": "ready", "video_id": "case", "output_dir": "out"}

    monkeypatch.setattr("sceneweaver.user_web.ingest_video", fake_ingest_video)

    result = ingest_video_from_payload({"source": "movie.mp4", "concurrency": 40}, log=logs.append)

    assert result["status"] == "ready"
    assert called["source"] == "movie.mp4"
    assert called["kwargs"]["concurrency"] == 40
    assert logs == ["packaging"]


def test_start_ingest_job_records_success(monkeypatch):
    with INGEST_JOBS_LOCK:
        INGEST_JOBS.clear()

    def fake_ingest_video_from_payload(payload, *, log=None, cancel_check=None):
        assert cancel_check is not None
        log("analysis step")
        return {
            "status": "ready",
            "video_id": "demo",
            "output_dir": "outputs/film_analysis/demo",
            "scene_count": 2,
            "analysis_scene_count": 2,
            "card_count": 2,
            "scene_analysis_model": "qwen3.7-plus",
        }

    monkeypatch.setattr("sceneweaver.user_web.ingest_video_from_payload", fake_ingest_video_from_payload)

    result = start_ingest_job({"source": "movie.mp4", "concurrency": 40}, run_async=False)
    job = ingest_job_payload(result["job_id"])

    assert job is not None
    assert job["status"] == "ready"
    assert job["result"]["video_id"] == "demo"
    assert any("并发=40" in line for line in job["logs"])
    assert any("analysis step" in line for line in job["logs"])
    assert any("入库完成" in line for line in job["logs"])


def test_start_ingest_job_records_error(monkeypatch):
    with INGEST_JOBS_LOCK:
        INGEST_JOBS.clear()

    def fake_ingest_video_from_payload(payload, *, log=None, cancel_check=None):
        log("before failure")
        raise RuntimeError("provider timeout")

    monkeypatch.setattr("sceneweaver.user_web.ingest_video_from_payload", fake_ingest_video_from_payload)

    result = start_ingest_job({"source": "movie.mp4"}, run_async=False)
    job = ingest_job_payload(result["job_id"])

    assert job is not None
    assert job["status"] == "error"
    assert job["error"] == "provider timeout"
    assert any("before failure" in line for line in job["logs"])
    assert any("入库失败" in line for line in job["logs"])


def test_cancel_ingest_job_marks_job_canceling():
    with INGEST_JOBS_LOCK:
        INGEST_JOBS.clear()
        INGEST_JOBS["job-1"] = IngestJob(job_id="job-1", status="running")

    result = cancel_ingest_job("job-1")
    job = ingest_job_payload("job-1")

    assert result is not None
    assert job is not None
    assert job["status"] == "canceling"
    assert job["cancel_requested"] is True
    assert any("已请求终止处理" in line for line in job["logs"])


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
            "creator_intent_prompt": "先询问创作者真实意图",
            "prompt_revision_rounds": 2,
            "fine_tune_instruction": "更克制一点",
            "variant_index": 2,
            "variant_count": 3,
        }
    )

    assert result["status"] == "ok"
    assert called["args"][0] == "真实团队"
    assert called["kwargs"]["script_brief"] == "招聘短片"
    assert called["kwargs"]["top_k"] == 3
    assert called["kwargs"]["duration_seconds"] == 60
    assert called["kwargs"]["max_tokens"] == 7000
    assert called["kwargs"]["creator_intent_prompt"] == "先询问创作者真实意图"
    assert called["kwargs"]["prompt_revision_rounds"] == 2
    assert called["kwargs"]["fine_tune_instruction"] == "更克制一点"
    assert called["kwargs"]["variant_index"] == 2
    assert called["kwargs"]["variant_count"] == 3
    assert called["kwargs"]["reference_matches"] is None


def test_generate_script_payload_accepts_manual_reference_matches(monkeypatch):
    called = {}
    reference_matches = [{"item_id": "match-1", "rank": 1}]

    def fake_generate_script(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return {"status": "ok", "generation_contract": {"reference_mode": "manual_reference_matches"}}

    monkeypatch.setattr("sceneweaver.user_web.generate_script", fake_generate_script)

    result = generate_script_from_payload(
        {
            "query": "真实团队",
            "sources": ["outputs/film_analysis"],
            "top_k": 1,
            "reference_matches": reference_matches,
        }
    )

    assert result["status"] == "ok"
    assert called["args"][0] == "真实团队"
    assert called["kwargs"]["top_k"] == 1
    assert called["kwargs"]["reference_matches"] == reference_matches


def test_generate_script_payload_rejects_invalid_manual_reference_matches():
    with pytest.raises(ValueError, match="reference_matches must be a list"):
        generate_script_from_payload({"query": "真实团队", "reference_matches": {"item_id": "match-1"}})


def test_script_agent_payload_calls_core(monkeypatch):
    called = {}

    def fake_run_script_agent_task(mode, **kwargs):
        called["mode"] = mode
        called["kwargs"] = kwargs
        return {"status": "ok", "mode": mode, "reply": "建议保留真实感"}

    monkeypatch.setattr("sceneweaver.user_web.run_script_agent_task", fake_run_script_agent_task)

    result = script_agent_from_payload(
        {
            "mode": "intent",
            "user_input": "我想要更真实一点",
            "context": {"script_form": {"query": "真实团队"}},
            "history": [{"role": "user", "content": "偏纪实"}],
            "max_tokens": 1200,
        }
    )

    assert result["reply"] == "建议保留真实感"
    assert called["mode"] == "intent"
    assert called["kwargs"]["user_input"] == "我想要更真实一点"
    assert called["kwargs"]["context"] == {"script_form": {"query": "真实团队"}}
    assert called["kwargs"]["history"] == [{"role": "user", "content": "偏纪实"}]
    assert called["kwargs"]["max_tokens"] == 1200


def test_script_agent_payload_rejects_invalid_context():
    with pytest.raises(ValueError, match="context must be an object"):
        script_agent_from_payload({"mode": "intent", "context": []})


def test_ping_llm_payload_requires_profile():
    with pytest.raises(ValueError, match="profile is required"):
        ping_llm_from_payload({})


def test_ping_llm_payload_calls_central_helper(monkeypatch):
    called = {}

    def fake_ping(profile, **kwargs):
        called["profile"] = profile
        called["kwargs"] = kwargs
        return {"status": "ok", "profile": profile}

    monkeypatch.setattr("sceneweaver.user_web.ping_llm_profile", fake_ping)

    result = ping_llm_from_payload({"profile": "basic", "profile_data": {"model": "m"}, "timeout_seconds": 3})

    assert result == {"status": "ok", "profile": "basic"}
    assert called["profile"] == "basic"
    assert called["kwargs"]["profile_data"] == {"model": "m"}
    assert called["kwargs"]["timeout_seconds"] == 3.0
