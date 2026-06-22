from __future__ import annotations

from pathlib import Path
from typing import Callable

from sceneweaver.analysis.scene_package_builder import build_scene_packages, write_scene_packages
from sceneweaver.input.downloader import download_video
from sceneweaver.input.subtitles import download_bilibili_subtitles
from sceneweaver.split.frame_sampler import sample_scene_frames
from sceneweaver.split.scene_detector import detect_scenes
from sceneweaver.split.subtitle_segmenter import parse_srt, segment_subtitles_for_scenes


def run_package_video(
    url: str,
    output_dir: Path,
    scene_threshold: float = 27.0,
    subtitle_path: Path | None = None,
    split_video: bool = False,
    force: bool = False,
    frame_workers: int | None = None,
    burn_subtitles: bool = False,
    browser_profile_subtitles: bool = False,
    browser_cdp_url: str | None = None,
    browser_profile_timeout_seconds: float = 30.0,
    browser_user_data_dir: Path | None = None,
    browser_profile_directory: str | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    def emit(message: str) -> None:
        if log:
            log(message)

    output_dir = output_dir.resolve()
    emit("[1/6] Downloading video and metadata...")
    asset = download_video(
        url,
        output_dir,
        force=force,
        browser_cdp_url=browser_cdp_url,
        log=log,
    )
    emit(f"[1/6] Video ready: {asset.video_path}")

    emit("[2/6] Detecting scenes with PySceneDetect...")
    scenes = detect_scenes(
        asset.video_path,
        output_dir / "scenes",
        threshold=scene_threshold,
        split_video=split_video,
    )
    emit(f"[2/6] Detected {len(scenes)} scenes.")

    emit("[3/6] Loading subtitles...")
    if subtitle_path:
        cues = parse_srt(subtitle_path)
    else:
        cues = download_bilibili_subtitles(
            url,
            output_dir,
            force=force,
            browser_profile=browser_profile_subtitles,
            browser_cdp_url=browser_cdp_url,
            browser_profile_timeout_seconds=browser_profile_timeout_seconds,
            browser_user_data_dir=browser_user_data_dir,
            browser_profile_directory=browser_profile_directory,
            log=log,
        )
    subtitle_segments = segment_subtitles_for_scenes(scenes, cues)
    emit(f"[3/6] Subtitle cues loaded: {len(cues)}.")

    emit("[4/6] Sampling start/middle/end frames...")
    frame_paths = sample_scene_frames(
        asset.video_path,
        scenes,
        output_dir / "frames",
        force=force,
        subtitle_cues=cues,
        burn_subtitles=burn_subtitles,
        frame_workers=frame_workers,
    )
    emit("[4/6] Frames sampled.")

    emit("[5/6] Building scene packages...")
    packages = build_scene_packages(
        video_id=asset.video_id,
        source_url=asset.source_url,
        scenes=scenes,
        frame_paths=frame_paths,
        subtitle_segments=subtitle_segments,
    )
    emit(f"[5/6] Built {len(packages)} scene packages.")

    emit("[6/6] Writing scene packages...")
    write_scene_packages(output_dir, packages)
    emit(f"[6/6] Packages written: {output_dir / 'packages'}")
    return output_dir
