from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from sceneweaver.input.bilibili import extract_bvid
from sceneweaver.pipeline.mock_pipeline import run_mock_pipeline
from sceneweaver.pipeline.package_video import run_package_video
from sceneweaver.analysis.scene_analyzer import analyze_scene_packages

app = typer.Typer(help="SceneWeaver director experience analysis CLI.")


@app.callback()
def main() -> None:
    """SceneWeaver director experience analysis CLI."""


@app.command("mock-run")
def mock_run(
    output: Path = typer.Option(
        Path("outputs/mock_video"),
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
        Path("outputs/video_package"),
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
    output_dir = Path("outputs") / output_name

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

    typer.echo("Phase 2/2: Analyzing scene packages...")
    scenes = analyze_scene_packages(
        packaged_dir,
        limit=limit,
        force=update,
        prompt_path=prompt_path,
        max_workers=concurrency,
        log=typer.echo,
    )
    typer.echo(f"Scene packages written to: {packaged_dir / 'packages'}")
    typer.echo(f"Scene analysis written to: {packaged_dir / 'analysis'}")
    typer.echo(f"Scenes analyzed: {scenes.scene_count}")


if __name__ == "__main__":
    app()
