from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sceneweaver.llm.client import VisionLLMClient
from sceneweaver.schemas import SceneAnalysis, ScenePackage, ScenesAnalysis
from sceneweaver.storage.json_store import read_json, write_json


class SceneLLMClient(Protocol):
    def analyze_images_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> dict:
        ...


def analyze_scene_packages(
    output_dir: Path,
    *,
    client: SceneLLMClient | None = None,
    prompt_path: Path | None = None,
    limit: int | None = None,
    force: bool = False,
) -> ScenesAnalysis:
    output_dir = output_dir.resolve()
    packages_dir = output_dir / "packages"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_scene_analysis_prompt(prompt_path)
    llm_client = client or VisionLLMClient()
    package_paths = sorted(packages_dir.glob("scene_*.json"))
    if limit is not None:
        package_paths = package_paths[:limit]

    analyses: list[SceneAnalysis] = []
    source_url = ""
    video_id = ""
    for package_path in package_paths:
        package = read_json(package_path, ScenePackage)
        source_url = package.metadata.source_url
        video_id = package.source_video_id
        output_path = analysis_dir / f"{package.scene_id}.json"
        if output_path.exists() and not force:
            analyses.append(read_json(output_path, SceneAnalysis))
            continue

        frame_paths = _resolve_frame_paths(output_dir, package)
        user_prompt = build_scene_user_prompt(package)
        raw = llm_client.analyze_images_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=frame_paths,
        )
        analysis = SceneAnalysis.model_validate(raw)
        write_json(output_path, analysis)
        analyses.append(analysis)

    scenes = ScenesAnalysis(
        video_id=video_id,
        source_url=source_url,
        scene_count=len(analyses),
        scenes=analyses,
    )
    write_json(analysis_dir / "scenes.json", scenes)
    return scenes


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
