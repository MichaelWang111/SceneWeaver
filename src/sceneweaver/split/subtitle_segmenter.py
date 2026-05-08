from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sceneweaver.schemas import SubtitleItem, SubtitleSegment
from sceneweaver.split.scene_detector import SceneSpan
from sceneweaver.split.timecode import seconds_to_timestamp, timestamp_to_seconds

TIMING_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass(frozen=True)
class SubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str


def parse_srt(path: Path) -> list[SubtitleCue]:
    if not path.exists():
        return []

    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_index == -1:
            continue
        match = TIMING_RE.search(lines[timing_index])
        if not match:
            continue
        text = " ".join(lines[timing_index + 1 :]).strip()
        if not text:
            continue
        cues.append(
            SubtitleCue(
                start_seconds=timestamp_to_seconds(match.group("start")),
                end_seconds=timestamp_to_seconds(match.group("end")),
                text=text,
            )
        )
    return cues


def segment_subtitles_for_scenes(
    scenes: list[SceneSpan],
    cues: list[SubtitleCue],
) -> dict[str, SubtitleSegment]:
    segments: dict[str, SubtitleSegment] = {}
    for scene in scenes:
        items = [
            SubtitleItem(
                start=seconds_to_timestamp(cue.start_seconds),
                end=seconds_to_timestamp(cue.end_seconds),
                text=cue.text,
            )
            for cue in cues
            if _overlaps(scene.start_seconds, scene.end_seconds, cue.start_seconds, cue.end_seconds)
        ]
        segments[scene.scene_id] = SubtitleSegment(
            text=" ".join(item.text for item in items),
            items=items,
        )
    return segments


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a

