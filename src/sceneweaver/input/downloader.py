from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sceneweaver.input.bilibili import extract_video_id

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    source_url: str
    video_path: Path
    metadata_path: Path
    title: str = ""
    uploader: str = ""


def get_video_info(url: str, runner: Runner = subprocess.run) -> dict:
    """Fetch lightweight metadata with yt-dlp, matching the reference pipeline pattern."""
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--print",
        "%(title)s",
        "--print",
        "%(uploader)s",
        "--print",
        "%(channel)s",
        "--print",
        "%(duration)s",
        "--print",
        "%(view_count)s",
        "--no-download",
        url,
    ]
    result = runner(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"success": False, "title": "", "uploader": ""}

    lines = result.stdout.strip().splitlines()
    return {
        "success": True,
        "title": lines[0] if len(lines) > 0 else "",
        "uploader": lines[1] if len(lines) > 1 else "",
        "channel": lines[2] if len(lines) > 2 else "",
        "duration": lines[3] if len(lines) > 3 else "",
        "view_count": lines[4] if len(lines) > 4 else "",
    }


def download_video(
    url: str,
    output_dir: Path,
    runner: Runner = subprocess.run,
    force: bool = False,
) -> VideoAsset:
    """Download one Bilibili video and persist basic metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(url)
    video_path = output_dir / "source" / "video.mp4"
    metadata_path = output_dir / "source" / "metadata.json"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    info = get_video_info(url, runner=runner)
    metadata = {"video_id": video_id, "source_url": url, **info}
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if force or not video_path.exists():
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o",
            str(video_path),
            url,
        ]
        runner(cmd, check=True)

    return VideoAsset(
        video_id=video_id,
        source_url=url,
        video_path=video_path,
        metadata_path=metadata_path,
        title=metadata.get("title", ""),
        uploader=metadata.get("uploader", ""),
    )
