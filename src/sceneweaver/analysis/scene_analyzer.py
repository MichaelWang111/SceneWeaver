from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Protocol

from sceneweaver.analysis.tags import add_tags_to_scene_raw, read_scene_analysis_with_tags
from sceneweaver.llm.client import VisionLLMClient
from sceneweaver.llm.runtime import LLMRunOptions, effective_concurrency
from sceneweaver.schemas import SceneAnalysis, ScenePackage, ScenesAnalysis
from sceneweaver.storage.json_store import read_json, write_json

LogFn = Callable[[str], None]
SCENE_PACKAGE_FILENAME_RE = re.compile(r"^scene_\d{3}\.json$")


class SceneLLMClient(Protocol):
    def analyze_images_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        timeout_seconds: float | None = None,
        retries: int = 0,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> dict:
        ...


@dataclass(frozen=True)
class SceneTask:
    index: int
    package: ScenePackage
    output_path: Path


def analyze_scene_packages(
    output_dir: Path,
    *,
    client: SceneLLMClient | None = None,
    prompt_path: Path | None = None,
    limit: int | None = None,
    force: bool = False,
    max_workers: int = 1,
    timeout_seconds: float | None = None,
    retries: int = 0,
    llm_options: LLMRunOptions | None = None,
    log: LogFn | None = None,
) -> ScenesAnalysis:
    output_dir = output_dir.resolve()
    packages_dir = output_dir / "packages"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    options = llm_options or LLMRunOptions(
        concurrency=max_workers,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    options.validate()

    system_prompt = load_scene_analysis_prompt(prompt_path)
    llm_client = client or VisionLLMClient()
    configured_workers = options.concurrency
    effective_workers = scene_effective_workers(configured_workers, llm_client)
    package_paths = sorted(
        path
        for path in packages_dir.glob("scene_*.json")
        if SCENE_PACKAGE_FILENAME_RE.match(path.name)
    )
    if limit is not None:
        package_paths = package_paths[:limit]

    total_count = len(package_paths)
    if log:
        if effective_workers != configured_workers:
            log(
                f"Preparing {total_count} scene(s) for analysis with concurrency={effective_workers} "
                f"(requested={configured_workers}, provider_limit_applied)."
            )
        else:
            log(f"Preparing {total_count} scene(s) for analysis with concurrency={effective_workers}.")

    analyses: list[SceneAnalysis | None] = [None] * total_count
    pending_tasks: list[SceneTask] = []
    source_url = ""
    video_id = ""
    reused_count = 0
    completed_count = 0

    for index, package_path in enumerate(package_paths):
        package = read_json(package_path, ScenePackage)
        source_url = package.metadata.source_url
        video_id = package.source_video_id
        output_path = analysis_dir / f"{package.scene_id}.json"
        if output_path.exists() and not force:
            analyses[index] = read_scene_analysis_with_tags(output_path, write_back=True)
            reused_count += 1
            completed_count += 1
            if log:
                log(f"[{completed_count}/{total_count}] Reused existing analysis for {package.scene_id}.")
            continue
        pending_tasks.append(SceneTask(index=index, package=package, output_path=output_path))

    if pending_tasks and log:
        log(f"Submitting {len(pending_tasks)} new scene request(s).")

    if pending_tasks:
        if effective_workers == 1:
            for task in pending_tasks:
                analyses[task.index] = _analyze_single_scene(
                    output_dir=output_dir,
                    task=task,
                    system_prompt=system_prompt,
                    client=llm_client,
                    options=options,
                )
                completed_count += 1
                if log:
                    log(f"[{completed_count}/{total_count}] Finished {task.package.scene_id}.")
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(
                        _analyze_single_scene,
                        output_dir=output_dir,
                        task=task,
                        system_prompt=system_prompt,
                        client=llm_client,
                        options=options,
                    ): task
                    for task in pending_tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    analyses[task.index] = future.result()
                    completed_count += 1
                    if log:
                        log(f"[{completed_count}/{total_count}] Finished {task.package.scene_id}.")

    completed_analyses = [analysis for analysis in analyses if analysis is not None]
    scenes = ScenesAnalysis(
        video_id=video_id,
        source_url=source_url,
        scene_count=len(completed_analyses),
        scenes=completed_analyses,
    )
    write_json(analysis_dir / "scenes.json", scenes)
    if log:
        log(
            "Analysis complete. "
            f"total={total_count}, reused={reused_count}, new={len(pending_tasks)}, written={len(completed_analyses)}."
        )
    return scenes


def _analyze_single_scene(
    *,
    output_dir: Path,
    task: SceneTask,
    system_prompt: str,
    client: SceneLLMClient,
    options: LLMRunOptions,
) -> SceneAnalysis:
    frame_paths = _resolve_frame_paths(output_dir, task.package)
    user_prompt = build_scene_user_prompt(task.package)
    raw = client.analyze_images_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=frame_paths,
        timeout_seconds=options.timeout_seconds,
        retries=options.retries,
        enable_thinking=options.enable_thinking,
        thinking_budget=options.thinking_budget,
    )
    raw = add_tags_to_scene_raw(raw, candidate_log_path=task.output_path.parent / "tag_candidates.jsonl")
    analysis = SceneAnalysis.model_validate(raw)
    write_json(task.output_path, analysis)
    return analysis


def scene_effective_workers(requested: int, client: SceneLLMClient) -> int:
    config = getattr(client, "config", None)
    if config is None:
        return requested
    provider = str(getattr(config, "provider", "auto") or "auto")
    model = str(getattr(config, "model", "") or "")
    return effective_concurrency(requested, provider=provider, model=model)


def load_scene_analysis_prompt(prompt_path: Path | None = None) -> str:
    path = prompt_path or Path(__file__).resolve().parents[3] / "prompts" / "scene_analysis.md"
    return path.read_text(encoding="utf-8")


def build_scene_user_prompt(package: ScenePackage) -> str:
    return f"""请分析这个 scene package。你会收到三张帧图，顺序分别是 start、middle、end。
scene_id: {package.scene_id}
source_video_id: {package.source_video_id}
time_range: {package.time_range.start} - {package.time_range.end}
duration_seconds: {package.time_range.duration_seconds}
subtitle_text: {package.subtitle_segment.text or "（无字幕）"}

请只返回符合 SceneAnalysis schema 的 JSON。"""


def _resolve_frame_paths(output_dir: Path, package: ScenePackage) -> list[Path]:
    paths = [
        output_dir / package.frames.start,
        output_dir / package.frames.middle,
        output_dir / package.frames.end,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frame files: {', '.join(missing)}")
    return paths
