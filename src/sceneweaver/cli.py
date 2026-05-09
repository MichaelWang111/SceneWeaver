from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Optional

import typer

from sceneweaver.analysis.associate_analyzer import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    associate_input,
)
from sceneweaver.analysis.fingerprint import generate_scene_fingerprints
from sceneweaver.input.bilibili import extract_bvid
from sceneweaver.llm.client import LLMConfig
from sceneweaver.pipeline.mock_pipeline import run_mock_pipeline
from sceneweaver.pipeline.package_video import run_package_video
from sceneweaver.analysis.scene_analyzer import analyze_scene_packages

app = typer.Typer(help="SceneWeaver director experience analysis CLI.")

OUTPUT_ROOT = Path("outputs")
KEY_ASSOCIATES_DIR = OUTPUT_ROOT / "key_associates"
FILM_ANALYSIS_DIR = OUTPUT_ROOT / "film_analysis"
MOCK_OUTPUT_DIR = OUTPUT_ROOT / "mock"


@app.callback()
def main() -> None:
    """SceneWeaver director experience analysis CLI."""


@app.command("mock-run")
def mock_run(
    output: Path = typer.Option(
        MOCK_OUTPUT_DIR / "mock_video",
        "--output",
        "-o",
        help="Directory where mock artifacts will be written.",
    ),
    source_url: str = typer.Option(
        "https://www.bilibili.com/video/BVxxxx",
        "--source-url",
        help="Source URL recorded in mock artifacts.",
    ),
) -> None:
    """Generate a validated local mock pipeline output."""
    output_dir = run_mock_pipeline(output.resolve(), source_url=source_url)
    typer.echo(f"Mock artifacts written to: {output_dir}")


@app.command("package-video")
def package_video(
    url: str = typer.Argument(..., help="Bilibili video URL."),
    output: Path = typer.Option(
        FILM_ANALYSIS_DIR / "video_package",
        "--output",
        "-o",
        help="Directory where video package artifacts will be written.",
    ),
    scene_threshold: float = typer.Option(
        27.0,
        "--scene-threshold",
        help="PySceneDetect content detection threshold.",
    ),
    subtitle_path: Optional[Path] = typer.Option(
        None,
        "--subtitle",
        help="Optional existing SRT subtitle file to segment into scenes.",
    ),
    split_video: bool = typer.Option(
        False,
        "--split-video",
        help="Also split scene clips with PySceneDetect. Slower and not required for packages.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate video and frame artifacts even if files exist.",
    ),
) -> None:
    """Download and package a real Bilibili video into validated scene packages."""
    output_dir = run_package_video(
        url=url,
        output_dir=output,
        scene_threshold=scene_threshold,
        subtitle_path=subtitle_path,
        split_video=split_video,
        force=force,
        log=typer.echo,
    )
    typer.echo(f"Scene packages written to: {output_dir / 'packages'}")


@app.command("analyze-scenes")
def analyze_scenes(
    output: Path = typer.Argument(..., help="Video output directory containing packages/ and frames/."),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Analyze only the first N scene packages.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing analysis files. By default, existing scene results are reused and skipped.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom scene analysis prompt path.",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "--workers",
        min=1,
        help="Number of scene analysis requests to run in parallel.",
    ),
) -> None:
    """Send scene packages and frames to a vision LLM and write validated scene analyses."""
    scenes = analyze_scene_packages(
        output,
        limit=limit,
        force=update,
        prompt_path=prompt_path,
        max_workers=concurrency,
        log=typer.echo,
    )
    typer.echo(f"Scene analysis written to: {output.resolve() / 'analysis'}")
    typer.echo(f"Scenes analyzed: {scenes.scene_count}")


@app.command("fingerprint-scenes")
def fingerprint_scenes(
    output: Path = typer.Argument(..., help="Video output directory containing analysis/scenes.json."),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing scene fingerprint files.",
    ),
) -> None:
    """Generate scene and film creative fingerprints from existing scene analyses."""
    film_fingerprint = generate_scene_fingerprints(output, force=update, log=typer.echo)
    typer.echo(f"Fingerprints written to: {output.resolve() / 'fingerprints'}")
    typer.echo(f"Scenes fingerprinted: {film_fingerprint.scene_count}")


@app.command("associate")
def associate(
    input_text: str = typer.Argument(..., help="Rough keywords or brief to expand into director associations."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="JSON file where association analysis will be written. Defaults to outputs/key_associates/.",
    ),
    max_items: int = typer.Option(
        DEFAULT_MAX_ITEMS,
        "--max-items",
        min=8,
        max=120,
        help="Target number of association items. The LLM may vary within a useful creative range.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom associate prompt path.",
    ),
    timeout_seconds: float = typer.Option(
        DEFAULT_TIMEOUT_SECONDS,
        "--timeout-seconds",
        min=1.0,
        help="LLM request timeout in seconds.",
    ),
    retries: int = typer.Option(
        DEFAULT_RETRIES,
        "--retries",
        min=0,
        help="Retry count for timeout, connection, rate-limit, and 5xx failures.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print request progress and provider metadata to stderr.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        "--flue",
        help="Stream raw LLM JSON chunks to stderr while preserving final validated JSON on stdout.",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking",
        help="Enable DashScope/Qwen thinking mode and stream reasoning_content to stderr.",
    ),
    thinking_budget: Optional[int] = typer.Option(
        None,
        "--thinking-budget",
        min=1,
        help="Maximum tokens for DashScope/Qwen thinking content. Implies --thinking.",
    ),
) -> None:
    """Expand rough keywords into structured director and screenwriter association material."""
    output_path = output or build_key_associate_output_path(input_text)
    request_thinking = thinking or thinking_budget is not None
    if debug or stream or request_thinking:
        config = LLMConfig.from_env()
        typer.echo(
            "Associate debug: "
            f"base_url={config.base_url!r}, model={config.model!r}, "
            f"output={str(output_path.resolve())!r}, "
            f"thinking={request_thinking!r}, thinking_budget={thinking_budget!r}",
            err=True,
        )
    if stream:
        typer.echo("[associate] Streaming raw LLM JSON chunks to stderr. Final validated JSON remains on stdout.", err=True)
    if request_thinking:
        typer.echo(
            "[associate] Streaming DashScope/Qwen reasoning_content to stderr. Final validated JSON remains on stdout.",
            err=True,
        )
    analysis = associate_input(
        input_text,
        output_path=output_path,
        max_items=max_items,
        prompt_path=prompt_path,
        timeout_seconds=timeout_seconds,
        retries=retries,
        log=(lambda message: typer.echo(f"[associate] {message}", err=True)) if debug or stream or request_thinking else None,
        stream_callback=_write_stderr_chunk if stream else None,
        reasoning_callback=_write_stderr_chunk if request_thinking else None,
        enable_thinking=True if request_thinking else None,
        thinking_budget=thinking_budget,
    )
    typer.echo(json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2))
    typer.echo(f"Association analysis written to: {output_path.resolve()}", err=True)


@app.command("run")
def run_pipeline(
    url: str = typer.Argument(..., help="Bilibili video URL."),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Analyze only the first N scene packages.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing downloaded assets, packages, frames, and analysis results.",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "--workers",
        min=1,
        help="Number of scene analysis requests to run in parallel.",
    ),
    scene_threshold: float = typer.Option(
        27.0,
        "--scene-threshold",
        help="PySceneDetect content detection threshold.",
    ),
    subtitle_path: Optional[Path] = typer.Option(
        None,
        "--subtitle",
        help="Optional existing SRT subtitle file to segment into scenes.",
    ),
    split_video: bool = typer.Option(
        False,
        "--split-video",
        help="Also split scene clips with PySceneDetect. Slower and not required for packages.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom scene analysis prompt path.",
    ),
) -> None:
    """Run video packaging and scene analysis end to end with an auto-generated outputs directory."""
    bvid = extract_bvid(url)
    output_name = bvid or "bilibili_unknown"
    output_dir = FILM_ANALYSIS_DIR / output_name

    typer.echo(f"Output directory: {output_dir.resolve()}")
    typer.echo("Phase 1/2: Packaging video into scene packages...")
    packaged_dir = run_package_video(
        url=url,
        output_dir=output_dir,
        scene_threshold=scene_threshold,
        subtitle_path=subtitle_path,
        split_video=split_video,
        force=update,
        log=typer.echo,
    )

    typer.echo("Phase 2/3: Analyzing scene packages...")
    scenes = analyze_scene_packages(
        packaged_dir,
        limit=limit,
        force=update,
        prompt_path=prompt_path,
        max_workers=concurrency,
        log=typer.echo,
    )
    typer.echo("Phase 3/3: Generating creative fingerprints...")
    film_fingerprint = generate_scene_fingerprints(
        packaged_dir,
        force=update,
        log=typer.echo,
    )
    typer.echo(f"Scene packages written to: {packaged_dir / 'packages'}")
    typer.echo(f"Scene analysis written to: {packaged_dir / 'analysis'}")
    typer.echo(f"Scenes analyzed: {scenes.scene_count}")
    typer.echo(f"Fingerprints written to: {packaged_dir / 'fingerprints'}")
    typer.echo(f"Scenes fingerprinted: {film_fingerprint.scene_count}")


def build_key_associate_output_path(input_text: str, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(input_text.strip().encode("utf-8")).hexdigest()[:8]
    slug = _safe_filename_slug(input_text)
    return KEY_ASSOCIATES_DIR / f"{timestamp}_{slug}_{digest}.json"


def _safe_filename_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    if not slug:
        return "associate"
    return slug[:32].strip("_") or "associate"


def _write_stderr_chunk(chunk: str) -> None:
    sys.stderr.write(chunk)
    sys.stderr.flush()


if __name__ == "__main__":
    app()
