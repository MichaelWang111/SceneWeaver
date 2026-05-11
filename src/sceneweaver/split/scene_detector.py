from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sceneweaver.split.timecode import seconds_to_timestamp

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class SceneSpan:
    scene_id: str
    scene_index: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return round(max(0.0, self.end_seconds - self.start_seconds), 3)

    @property
    def start(self) -> str:
        return seconds_to_timestamp(self.start_seconds)

    @property
    def end(self) -> str:
        return seconds_to_timestamp(self.end_seconds)


def detect_scenes(
    video_path: Path,
    scenes_dir: Path,
    threshold: float = 27.0,
    split_video: bool = True,
    runner: Runner = subprocess.run,
) -> list[SceneSpan]:
    """Detect scenes with PySceneDetect and optionally split clips with the CLI."""
    scene_spans = _detect_scene_spans(video_path, threshold)

    if split_video:
        scenes_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "scenedetect",
            "-i",
            str(video_path),
            "-o",
            str(scenes_dir),
            "detect-content",
            "-t",
            str(threshold),
            "split-video",
        ]
        runner(cmd, capture_output=True, text=True, check=True)

    if not scene_spans:
        scene_spans = [
            SceneSpan(
                scene_id="scene_001",
                scene_index=1,
                start_seconds=0.0,
                end_seconds=_probe_duration_seconds(video_path, runner=runner),
            )
        ]

    return scene_spans


def _probe_duration_seconds(video_path: Path, runner: Runner = subprocess.run) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = runner(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _detect_scene_spans(video_path: Path, threshold: float) -> list[SceneSpan]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise RuntimeError("PySceneDetect is required for scene detection") from exc

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    spans: list[SceneSpan] = []
    for index, (start, end) in enumerate(scene_manager.get_scene_list(), 1):
        spans.append(
            SceneSpan(
                scene_id=f"scene_{index:03d}",
                scene_index=index,
                start_seconds=start.get_seconds(),
                end_seconds=end.get_seconds(),
            )
        )
    return spans
