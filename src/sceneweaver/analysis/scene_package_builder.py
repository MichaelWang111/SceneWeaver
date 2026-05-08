from __future__ import annotations

from pathlib import Path
import json

from sceneweaver.schemas import FrameSet, SceneMetadata, ScenePackage, SubtitleSegment, TimeRange
from sceneweaver.split.scene_detector import SceneSpan
from sceneweaver.storage.json_store import write_json


def build_scene_packages(
    *,
    video_id: str,
    source_url: str,
    scenes: list[SceneSpan],
    frame_paths: dict[str, dict[str, str]],
    subtitle_segments: dict[str, SubtitleSegment],
    language: str = "zh-CN",
) -> list[ScenePackage]:
    packages: list[ScenePackage] = []
    for scene in scenes:
        frames = frame_paths[scene.scene_id]
        packages.append(
            ScenePackage(
                scene_id=scene.scene_id,
                source_video_id=video_id,
                time_range=TimeRange(
                    start=scene.start,
                    end=scene.end,
                    duration_seconds=scene.duration_seconds,
                ),
                frames=FrameSet(
                    start=frames["start"],
                    middle=frames["middle"],
                    end=frames["end"],
                ),
                subtitle_segment=subtitle_segments.get(scene.scene_id, SubtitleSegment()),
                metadata=SceneMetadata(
                    scene_index=scene.scene_index,
                    source_url=source_url,
                    language=language,
                ),
            )
        )
    return packages


def write_scene_packages(output_dir: Path, packages: list[ScenePackage]) -> None:
    for package in packages:
        write_json(output_dir / "packages" / f"{package.scene_id}.json", package)
    write_scene_package_manifest(output_dir, packages)


def write_scene_package_manifest(output_dir: Path, packages: list[ScenePackage]) -> None:
    manifest_path = output_dir / "packages" / "scene_packages.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "video_id": packages[0].source_video_id if packages else "",
        "source_url": packages[0].metadata.source_url if packages else "",
        "scene_count": len(packages),
        "packages": [
            {
                "scene_id": package.scene_id,
                "scene_index": package.metadata.scene_index,
                "time_range": package.time_range.model_dump(mode="json"),
                "package_path": f"packages/{package.scene_id}.json",
                "frames": package.frames.model_dump(mode="json"),
                "has_subtitle": bool(package.subtitle_segment.text),
            }
            for package in packages
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
