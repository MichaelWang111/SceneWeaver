from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from sceneweaver.split.scene_detector import SceneSpan
from sceneweaver.split.timecode import seconds_to_timestamp

Runner = Callable[..., subprocess.CompletedProcess]


def sample_scene_frames(
    video_path: Path,
    scenes: list[SceneSpan],
    frames_dir: Path,
    runner: Runner = subprocess.run,
    force: bool = False,
) -> dict[str, dict[str, str]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, str]] = {}

    for scene in scenes:
        points = {
            "start": scene.start_seconds + 0.05,
            "middle": (scene.start_seconds + scene.end_seconds) / 2,
            "end": max(scene.start_seconds, scene.end_seconds - 0.05),
        }
        scene_frames: dict[str, str] = {}
        for label, seconds in points.items():
            frame_path = frames_dir / f"{scene.scene_id}_{label}.jpg"
            if force or not frame_path.exists():
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    seconds_to_timestamp(seconds),
                    "-i",
                    str(video_path),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ]
                runner(cmd, capture_output=True, check=True)
            scene_frames[label] = frame_path.relative_to(frames_dir.parent).as_posix()
        results[scene.scene_id] = scene_frames

    return results
