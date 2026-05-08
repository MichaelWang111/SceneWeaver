from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

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
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-run scene analysis even if analysis/scene_XXX.json exists.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom scene analysis prompt path.",
    ),
) -> None:
    """Send scene packages and frames to a vision LLM and write validated scene analyses."""
    scenes = analyze_scene_packages(
        output,
        limit=limit,
        force=force,
        prompt_path=prompt_path,
    )
    typer.echo(f"Scene analysis written to: {output.resolve() / 'analysis'}")
    typer.echo(f"Scenes analyzed: {scenes.scene_count}")


if __name__ == "__main__":
    app()
