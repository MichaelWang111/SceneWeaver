from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal

from retrieval_lab.retrieval import DEFAULT_RETRIEVAL_RUN_OUTPUT, retrieval_run

from sceneweaver.analysis.experience_extractor import extract_experience_cards
from sceneweaver.analysis.scene_analyzer import analyze_scene_packages
from sceneweaver.input.bilibili import extract_video_id
from sceneweaver.llm.client import LLMConfig, VisionLLMClient, llm_config_metadata
from sceneweaver.llm.providers import default_base_url, infer_provider_from_model, normalize_model_id, normalize_provider
from sceneweaver.pipeline.local_video import local_video_id, package_local_video
from sceneweaver.pipeline.package_video import run_package_video
from sceneweaver.schemas import SceneAnalysis, ScenePackage
from sceneweaver.storage.json_store import read_json

SourceType = Literal["auto", "file", "url"]

DEFAULT_OUTPUT_ROOT = Path("outputs") / "film_analysis"
DEFAULT_CHANNEL_POLICY = "all"
DEFAULT_PLANNER = "multi_query"
DEFAULT_RANKING_KEY = "hybrid_rrf_constraints_signature"
DEFAULT_DIRECTOR_PROMPT = Path(__file__).resolve().parents[2] / "prompts" / "Director.md"
SCRIPT_GENERATION_MAX_TOKENS = 6000
USER_INGEST_SCENE_ANALYSIS_MODEL = "qwen3.7-plus"


def ingest_video(
    source: str | Path,
    *,
    source_type: SourceType = "auto",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output_dir: Path | None = None,
    video_id: str | None = None,
    scene_threshold: float = 27.0,
    subtitle_path: Path | None = None,
    split_video: bool = False,
    force: bool = False,
    frame_workers: int | None = None,
    burn_subtitles: bool = False,
    analyze: bool = True,
    extract_cards: bool = True,
    limit: int | None = None,
    concurrency: int = 1,
    timeout_seconds: float = 180.0,
    retries: int = 0,
    scene_analysis_model: str = USER_INGEST_SCENE_ANALYSIS_MODEL,
    log=None,
) -> dict[str, Any]:
    """Ingest a video into SceneWeaver artifacts usable by Retrieval Lab."""
    resolved_source_type = resolve_source_type(source, source_type)
    resolved_video_id = video_id or infer_video_id(source, resolved_source_type)
    target_dir = (output_dir or Path(output_root) / resolved_video_id).resolve()

    if resolved_source_type == "file":
        packaged_dir = package_local_video(
            Path(source),
            target_dir,
            video_id=resolved_video_id,
            scene_threshold=scene_threshold,
            subtitle_path=subtitle_path,
            split_video=split_video,
            force=force,
            frame_workers=frame_workers,
            burn_subtitles=burn_subtitles,
            log=log,
        )
    else:
        packaged_dir = run_package_video(
            url=str(source),
            output_dir=target_dir,
            scene_threshold=scene_threshold,
            subtitle_path=subtitle_path,
            split_video=split_video,
            force=force,
            frame_workers=frame_workers,
            burn_subtitles=burn_subtitles,
            log=log,
        )

    scene_count = package_scene_count(packaged_dir)
    analysis_scene_count = 0
    card_count = existing_card_count(packaged_dir)
    status = "packaged"
    active_scene_analysis_model = ""

    if analyze:
        scene_llm_client = build_scene_analysis_client(scene_analysis_model)
        active_scene_analysis_model = scene_llm_client.config.model
        scenes = analyze_scene_packages(
            packaged_dir,
            client=scene_llm_client,
            limit=limit,
            force=force,
            max_workers=concurrency,
            timeout_seconds=timeout_seconds,
            retries=retries,
            log=log,
        )
        analysis_scene_count = scenes.scene_count
        status = "analyzed"
        if extract_cards:
            cards = extract_experience_cards(packaged_dir, force=force, log=log)
            card_count = len(cards)
            status = "ready"
    elif extract_cards:
        cards = extract_experience_cards(packaged_dir, force=force, log=log)
        card_count = len(cards)
        status = "ready"

    cards_path = packaged_dir / "analysis" / "experience_cards.jsonl"
    return {
        "status": status,
        "source_type": resolved_source_type,
        "source": str(source),
        "video_id": resolved_video_id,
        "output_dir": str(packaged_dir.resolve()),
        "packages_path": str((packaged_dir / "packages").resolve()),
        "analysis_path": str((packaged_dir / "analysis").resolve()),
        "cards_path": str(cards_path.resolve()),
        "cards_exists": cards_path.exists(),
        "scene_count": scene_count,
        "analysis_scene_count": analysis_scene_count,
        "card_count": card_count,
        "scene_analysis_model": active_scene_analysis_model,
    }


def build_scene_analysis_client(model: str = USER_INGEST_SCENE_ANALYSIS_MODEL) -> VisionLLMClient:
    config = LLMConfig.from_env()
    normalized_model = normalize_model_id(model)
    if normalized_model:
        config = replace(config, model=normalized_model)
    if infer_provider_from_model(normalized_model) == "dashscope":
        config = replace(
            config,
            provider="dashscope",
            api_key=_dashscope_scene_analysis_api_key(config),
            base_url=_dashscope_scene_analysis_base_url(config),
        )
    return VisionLLMClient(config)


def _dashscope_scene_analysis_api_key(config: LLMConfig) -> str:
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("VIDEO_ANALYZER_API_KEY")
        or (config.api_key if normalize_provider(config.provider) == "dashscope" else "")
    )


def _dashscope_scene_analysis_base_url(config: LLMConfig) -> str:
    return (
        os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("VIDEO_ANALYZER_BASE_URL")
        or (config.base_url if normalize_provider(config.provider) == "dashscope" else default_base_url("dashscope"))
    )


def search_scenes(
    query: str,
    sources: Iterable[str | Path] | None = None,
    *,
    top_k: int = 5,
    candidate_depth: int = 100,
    planner: str = DEFAULT_PLANNER,
    planner_cache: Path | None = None,
    ranking_key: str = DEFAULT_RANKING_KEY,
    channel_policy: str = DEFAULT_CHANNEL_POLICY,
    run_name: str = "user_search",
    include_payload: bool = True,
    include_channels: bool = False,
) -> dict[str, Any]:
    """Search SceneWeaver cards through Retrieval Lab and attach scene frame paths."""
    source_paths = [Path(source) for source in (sources or [DEFAULT_OUTPUT_ROOT])]
    artifact = retrieval_run(
        card_sources=source_paths,
        queries=[query],
        planner=planner,
        planner_cache=planner_cache,
        top_k=top_k,
        candidate_depth=candidate_depth,
        run_name=run_name,
        ranking_key=ranking_key,
        channel_policy=channel_policy,
    )
    rows = next(iter(artifact.get("run_rows", {}).values()), [])
    row = rows[0] if rows else {}
    matches = [
        build_scene_match(result, rank=index, include_payload=include_payload, include_channels=include_channels)
        for index, result in enumerate(row.get("top_results", []), start=1)
    ]
    return {
        "query": query,
        "sources": [str(path) for path in source_paths],
        "top_k": top_k,
        "planner": planner,
        "ranking_key": ranking_key,
        "channel_policy": channel_policy,
        "summary": artifact.get("summary", {}),
        "query_plan": row.get("query_plan", {}),
        "matches": matches,
    }


def generate_script(
    query: str,
    sources: Iterable[str | Path] | None = None,
    *,
    script_brief: str = "",
    top_k: int = 5,
    duration_seconds: int | None = None,
    tone: str = "",
    audience: str = "",
    must_include: str = "",
    avoid: str = "",
    candidate_depth: int = 100,
    planner: str = DEFAULT_PLANNER,
    planner_cache: Path | None = None,
    ranking_key: str = DEFAULT_RANKING_KEY,
    channel_policy: str = DEFAULT_CHANNEL_POLICY,
    run_name: str = "script_generation_search",
    prompt_path: Path | None = None,
    client: Any | None = None,
    timeout_seconds: float | None = None,
    retries: int = 0,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query is required")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    search_result = search_scenes(
        clean_query,
        sources,
        top_k=top_k,
        candidate_depth=candidate_depth,
        planner=planner,
        planner_cache=planner_cache,
        ranking_key=ranking_key,
        channel_policy=channel_policy,
        run_name=run_name,
        include_payload=True,
        include_channels=False,
    )
    reference_items, _image_paths = build_script_reference_items(search_result.get("matches", []))
    if not reference_items:
        raise ValueError("no usable reference frames were found for script generation")

    llm_client = client or VisionLLMClient()
    system_prompt = load_director_prompt(prompt_path)
    user_prompt = build_director_user_prompt(
        query=clean_query,
        script_brief=script_brief,
        duration_seconds=duration_seconds,
        tone=tone,
        audience=audience,
        must_include=must_include,
        avoid=avoid,
        reference_items=reference_items,
    )
    generation_tokens = max_tokens or SCRIPT_GENERATION_MAX_TOKENS
    raw_script = llm_client.analyze_text_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=generation_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    reference_ids = {item["reference_id"] for item in reference_items}
    script = normalize_generated_script(raw_script)
    errors = script_contract_errors(script, reference_ids)
    if errors:
        repaired = llm_client.analyze_text_json(
            system_prompt=system_prompt,
            user_prompt=build_script_repair_prompt(
                original_user_prompt=user_prompt,
                invalid_output=raw_script,
                errors=errors,
            ),
            max_tokens=generation_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        script = normalize_generated_script(repaired)
        errors = script_contract_errors(script, reference_ids)
    if errors:
        raise ValueError("script generation returned an incomplete shooting script: " + "; ".join(errors))
    validate_script_references(script, reference_ids)
    llm_metadata: dict[str, Any] = {}
    config = getattr(llm_client, "config", None)
    if config is not None:
        llm_metadata = llm_config_metadata(config)
    return {
        "status": "ok",
        "query": clean_query,
        "sources": search_result.get("sources", []),
        "top_k": top_k,
        "script_brief": script_brief,
        "duration_seconds": duration_seconds,
        "tone": tone,
        "audience": audience,
        "script": script,
        "generation_contract": {
            "mode": "shooting_script_from_scene_json_text",
            "max_tokens": generation_tokens,
        },
        "reference_items": reference_items,
        "search_result": search_result,
        "llm_metadata": llm_metadata,
    }


def build_scene_match(
    result: dict[str, Any],
    *,
    rank: int,
    include_payload: bool,
    include_channels: bool,
) -> dict[str, Any]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    scene_id = str(metadata.get("scene_id") or first_source_scene_id(result) or "")
    cards_path = Path(str(metadata.get("source_path") or "")) if metadata.get("source_path") else None
    film_dir = film_dir_from_cards_path(cards_path) if cards_path else None
    package_context = scene_package_context(film_dir, scene_id) if film_dir and scene_id else {}

    match = {
        "rank": rank,
        "item_id": result.get("item_id", ""),
        "score": result.get("score", result.get("final_score", 0.0)),
        "final_score": result.get("final_score", result.get("score", 0.0)),
        "video_id": metadata.get("video_id") or metadata.get("source_video_id", ""),
        "scene_id": scene_id,
        "card_id": metadata.get("card_id") or metadata.get("retrieval_id", ""),
        "time_range": package_context.get("time_range", {}),
        "frames": package_context.get("frames", {}),
        "source": {
            "film_dir": str(film_dir.resolve()) if film_dir else "",
            "cards_path": str(cards_path.resolve()) if cards_path and cards_path.exists() else str(cards_path or ""),
            "package_path": package_context.get("package_path", ""),
        },
        "summary_text": summary_text(result),
        "metadata": metadata,
        "diagnostics": {
            "lexical_score": result.get("lexical_score", 0.0),
            "semantic_score": result.get("semantic_score", result.get("embedding_score", 0.0)),
            "constraint_score": result.get("constraint_score", 0.0),
            "purpose_score": result.get("purpose_score", 0.0),
            "signature_score": result.get("signature_score", 0.0),
            "style_score": result.get("style_score", 0.0),
            "channel_scores": result.get("channel_scores", {}),
            "constraint_hits": result.get("constraint_hits", {}),
            "style_guardrail_action": result.get("style_guardrail_action", ""),
            "risk_evidence": result.get("risk_evidence", []),
            "explanation": result.get("explanation", ""),
        },
    }
    if include_payload:
        match["payload"] = result.get("payload", {})
    if include_channels:
        match["channels"] = result.get("channels", {})
    return match


def build_script_reference_items(matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Path]]:
    reference_items: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for index, match in enumerate(matches, start=1):
        frame = preferred_reference_frame(match)
        if not frame:
            continue
        image_paths.append(Path(frame["path"]))
        payload = match.get("payload", {}) if isinstance(match.get("payload"), dict) else {}
        reference_id = f"reference_{len(reference_items) + 1:03d}"
        reference_items.append(
            {
                "reference_id": reference_id,
                "rank": match.get("rank", index),
                "score": match.get("final_score", match.get("score", 0.0)),
                "video_id": match.get("video_id", ""),
                "scene_id": match.get("scene_id", ""),
                "card_id": match.get("card_id", ""),
                "time_range": match.get("time_range", {}),
                "summary_text": match.get("summary_text", ""),
                "frame_label": frame.get("label", "middle"),
                "frame_path": frame.get("path", ""),
                "frame_relative_path": frame.get("relative_path", ""),
                "package_path": (match.get("source", {}) or {}).get("package_path", ""),
                "director_strategy": payload.get("director_strategy", ""),
                "narrative_logic": payload.get("narrative_logic", ""),
                "reuse_condition": payload.get("reuse_condition", ""),
                "underlying_emotion": payload.get("underlying_emotion", ""),
                "shooting_techniques": payload.get("shooting_techniques", []),
                "visual_symbols": payload.get("visual_symbols", []),
                "style_traits": payload.get("style_traits", []),
                "copywriting_tone": payload.get("copywriting_tone", ""),
                "avoid": payload.get("avoid", []),
                "script_usecase": payload.get("script_usecase", {}),
            }
        )
    return reference_items, image_paths


def preferred_reference_frame(match: dict[str, Any]) -> dict[str, Any] | None:
    frames = match.get("frames", {}) if isinstance(match.get("frames"), dict) else {}
    for label in ("middle", "start", "end"):
        frame = frames.get(label)
        if isinstance(frame, dict) and frame.get("exists") and frame.get("path"):
            return {**frame, "label": label}
    return None


def load_director_prompt(prompt_path: Path | None = None) -> str:
    return (prompt_path or DEFAULT_DIRECTOR_PROMPT).read_text(encoding="utf-8")


def build_director_user_prompt(
    *,
    query: str,
    script_brief: str,
    duration_seconds: int | None,
    tone: str,
    audience: str,
    must_include: str,
    avoid: str,
    reference_items: list[dict[str, Any]],
) -> str:
    payload = {
        "task": "Generate an original script draft inspired by retrieved director experience references.",
        "query": query,
        "script_brief": script_brief,
        "duration_seconds": duration_seconds,
        "tone": tone,
        "audience": audience,
        "must_include": must_include,
        "avoid": avoid,
        "reference_semantics": {
            "mode": "creative_reference_soft_constraint",
            "input_mode": "scene_card_text_only_no_images",
            "meaning": "References provide real directing experience and reusable strategy. The retrieved frame only locates the scene record; no image is sent to the LLM in this step.",
        },
        "reference_scene_items": director_prompt_reference_items(reference_items),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def director_prompt_reference_items(reference_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return scene/card text references for the LLM without image paths."""
    blocked_keys = {"frame_path", "frame_relative_path", "package_path"}
    return [{key: value for key, value in item.items() if key not in blocked_keys} for item in reference_items]


def normalize_generated_script(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    script = raw.get("script") if isinstance(raw.get("script"), dict) else raw
    normalized = dict(script)
    if not normalized.get("script_markdown"):
        for key in ("markdown", "shooting_script", "script", "draft", "content"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["script_markdown"] = value
                break
    if not normalized.get("creative_strategy"):
        for key in ("strategy", "direction", "creative_direction"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized["creative_strategy"] = value
                break
    normalized.setdefault("title", "")
    normalized.setdefault("logline", "")
    normalized.setdefault("creative_strategy", "")
    normalized.setdefault("script_markdown", "")
    normalized.setdefault("beats", [])
    normalized.setdefault("reference_takeaways", [])
    normalized.setdefault("risks", [])
    return normalized


def script_contract_errors(script: dict[str, Any], reference_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not text_value(script.get("script_markdown")):
        errors.append("missing script_markdown shooting script body")
    elif len(text_value(script.get("script_markdown"))) < 120:
        errors.append("script_markdown is too short to be a usable shooting script")
    if not text_value(script.get("creative_strategy")):
        errors.append("missing creative_strategy")
    beats = script.get("beats")
    if not isinstance(beats, list) or not beats:
        errors.append("missing beats")
    else:
        for index, beat in enumerate(beats, start=1):
            if not isinstance(beat, dict):
                errors.append(f"beat_{index:03d} is not an object")
                continue
            if not text_value(beat.get("voiceover")) and not text_value(beat.get("visual_direction")):
                errors.append(f"beat_{index:03d} lacks voiceover or visual_direction")
    try:
        validate_script_references(script, reference_ids)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def build_script_repair_prompt(
    *,
    original_user_prompt: str,
    invalid_output: dict[str, Any],
    errors: list[str],
) -> str:
    payload = {
        "task": "Repair the previous response into a complete Chinese shooting script JSON.",
        "errors": errors,
        "requirements": [
            "Return JSON only.",
            "script_markdown must be the main deliverable: a complete shootable promotional film script, not a summary of references.",
            "Include scene/beat headings, voiceover, visual direction, shot notes, and transition rhythm.",
            "Use the provided scene/card text as creative reference and direction, but create a new recruitment promotional film script.",
            "Every inspired_by_reference_ids and reference_takeaways.reference_id must exist in the provided reference_scene_items.",
        ],
        "original_generation_input": json.loads(original_user_prompt),
        "previous_invalid_output": invalid_output,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def text_value(value: Any) -> str:
    return str(value or "").strip()


def validate_script_references(script: dict[str, Any], reference_ids: set[str]) -> None:
    missing: set[str] = set()
    beats = script.get("beats", []) if isinstance(script.get("beats"), list) else []
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        for reference_id in beat.get("inspired_by_reference_ids", []) or []:
            if reference_id not in reference_ids:
                missing.add(str(reference_id))
    takeaways = script.get("reference_takeaways", []) if isinstance(script.get("reference_takeaways"), list) else []
    for takeaway in takeaways:
        if not isinstance(takeaway, dict):
            continue
        reference_id = str(takeaway.get("reference_id", ""))
        if reference_id and reference_id not in reference_ids:
            missing.add(reference_id)
    if missing:
        raise ValueError(f"script referenced unknown reference ids: {', '.join(sorted(missing))}")


def resolve_source_type(source: str | Path, source_type: SourceType) -> Literal["file", "url"]:
    if source_type == "file":
        return "file"
    if source_type == "url":
        return "url"
    value = str(source)
    if Path(value).expanduser().exists():
        return "file"
    if "://" in value:
        return "url"
    return "file"


def infer_video_id(source: str | Path, source_type: Literal["file", "url"]) -> str:
    if source_type == "url":
        return extract_video_id(str(source))
    return local_video_id(Path(source))


def package_scene_count(output_dir: Path) -> int:
    manifest = output_dir / "packages" / "scene_packages.json"
    if manifest.exists():
        try:
            import json

            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            return int(data.get("scene_count", 0) or 0)
        except (OSError, ValueError, TypeError):
            pass
    return len(list((output_dir / "packages").glob("scene_*.json")))


def existing_card_count(output_dir: Path) -> int:
    cards_path = output_dir / "analysis" / "experience_cards.jsonl"
    if not cards_path.exists():
        return 0
    return sum(1 for line in cards_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def first_source_scene_id(result: dict[str, Any]) -> str:
    payload = result.get("payload", {}) if isinstance(result.get("payload"), dict) else {}
    scene_ids = payload.get("source_scene_ids", [])
    if isinstance(scene_ids, list) and scene_ids:
        return str(scene_ids[0])
    return ""


def film_dir_from_cards_path(cards_path: Path | None) -> Path | None:
    if cards_path is None:
        return None
    if cards_path.name == "experience_cards.jsonl" and cards_path.parent.name == "analysis":
        return cards_path.parent.parent
    if cards_path.is_dir():
        return cards_path
    return cards_path.parent


def scene_package_context(film_dir: Path, scene_id: str) -> dict[str, Any]:
    package_path = film_dir / "packages" / f"{scene_id}.json"
    if package_path.exists():
        package = read_json(package_path, ScenePackage)
        return {
            "package_path": str(package_path.resolve()),
            "time_range": package.time_range.model_dump(mode="json"),
            "frames": frames_payload(film_dir, package.frames.model_dump(mode="json")),
        }

    analysis_path = film_dir / "analysis" / f"{scene_id}.json"
    if analysis_path.exists():
        analysis = read_json(analysis_path, SceneAnalysis)
        return {
            "package_path": "",
            "time_range": analysis.time_range.model_dump(mode="json"),
            "frames": {},
        }

    return {"package_path": "", "time_range": {}, "frames": {}}


def frames_payload(film_dir: Path, frames: dict[str, str]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for label, relative_path in frames.items():
        path = (film_dir / relative_path).resolve()
        payload[label] = {
            "relative_path": relative_path,
            "path": str(path),
            "exists": path.exists(),
        }
    return payload


def summary_text(result: dict[str, Any]) -> str:
    channels = result.get("channels", {}) if isinstance(result.get("channels"), dict) else {}
    if channels.get("summary"):
        return str(channels["summary"])
    payload = result.get("payload", {}) if isinstance(result.get("payload"), dict) else {}
    for key in ("reuse_condition", "director_strategy", "narrative_logic"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


__all__ = [
    "DEFAULT_CHANNEL_POLICY",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PLANNER",
    "DEFAULT_RANKING_KEY",
    "DEFAULT_RETRIEVAL_RUN_OUTPUT",
    "USER_INGEST_SCENE_ANALYSIS_MODEL",
    "build_scene_match",
    "build_scene_analysis_client",
    "build_script_reference_items",
    "director_prompt_reference_items",
    "ingest_video",
    "generate_script",
    "search_scenes",
]
