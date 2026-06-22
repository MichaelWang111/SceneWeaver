from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Callable

from sceneweaver.analysis.scene_package_builder import build_scene_packages, write_scene_packages
from sceneweaver.split.frame_sampler import sample_scene_frames
from sceneweaver.split.scene_detector import detect_scenes
from sceneweaver.split.subtitle_segmenter import parse_srt, segment_subtitles_for_scenes

LogFn = Callable[[str], None]


def package_local_video(
    video_path: Path,
    output_dir: Path,
    *,
    video_id: str | None = None,
    source_url: str | None = None,
    scene_threshold: float = 27.0,
    subtitle_path: Path | None = None,
    split_video: bool = False,
    force: bool = False,
    frame_workers: int | None = None,
    burn_subtitles: bool = False,
    log: LogFn | None = None,
) -> Path:
    """Package an existing local video file into SceneWeaver scene packages."""

    def emit(message: str) -> None:
        if log:
            log(message)

    source = Path(video_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"video file not found: {source}")

    output_dir = Path(output_dir).resolve()
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    resolved_video_id = video_id or local_video_id(source)
    stored_video = source_dir / f"video{source.suffix.lower() or '.mp4'}"
    metadata_path = source_dir / "metadata.json"

    emit("[1/6] Preparing local video and metadata...")
    if force or not stored_video.exists() or source.resolve() != stored_video.resolve():
        if source.resolve() != stored_video.resolve():
            shutil.copy2(source, stored_video)
    metadata = {
        "video_id": resolved_video_id,
        "source_url": source_url or source.as_uri(),
        "original_path": str(source),
        "local_source_path": str(source),
        "stored_video_path": str(stored_video),
        "title": source.stem,
        "ingestion_method": "local_file",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit(f"[1/6] Video ready: {stored_video}")

    emit("[2/6] Detecting scenes with PySceneDetect...")
    scenes = detect_scenes(
        stored_video,
        output_dir / "scenes",
        threshold=scene_threshold,
        split_video=split_video,
    )
    emit(f"[2/6] Detected {len(scenes)} scenes.")

    emit("[3/6] Loading subtitles...")
    cues = parse_srt(subtitle_path) if subtitle_path else []
    subtitle_segments = segment_subtitles_for_scenes(scenes, cues)
    emit(f"[3/6] Subtitle cues loaded: {len(cues)}.")

    emit("[4/6] Sampling start/middle/end frames...")
    frame_paths = sample_scene_frames(
        stored_video,
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
        video_id=resolved_video_id,
        source_url=metadata["source_url"],
        scenes=scenes,
        frame_paths=frame_paths,
        subtitle_segments=subtitle_segments,
    )
    emit(f"[5/6] Built {len(packages)} scene packages.")

    emit("[6/6] Writing scene packages...")
    write_scene_packages(output_dir, packages)
    emit(f"[6/6] Packages written: {output_dir / 'packages'}")
    return output_dir


def local_video_id(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
        fingerprint_source = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        fingerprint_source = str(resolved)
    digest = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", resolved.stem).strip("_").lower()
    slug = slug[:48].strip("_") or "video"
    return f"local_{slug}_{digest}"


__all__ = ["local_video_id", "package_local_video"]
