from __future__ import annotations

import json
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
    return parse_subtitle_file(path)


def parse_subtitle_file(path: Path) -> list[SubtitleCue]:
    if not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_bilibili_subtitle_json(path)
    return parse_timed_text(path)


def parse_timed_text(path: Path) -> list[SubtitleCue]:
    blocks = re.split(r"\n\s*\n", _strip_vtt_header(path.read_text(encoding="utf-8-sig")).strip())
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timing_index == -1:
            continue
        match = TIMING_RE.search(lines[timing_index])
        if not match:
            continue
        text = " ".join(_clean_timed_text_line(line) for line in lines[timing_index + 1 :]).strip()
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


def cue_text_at(cues: list[SubtitleCue], seconds: float) -> str:
    matches = [cue.text for cue in cues if cue.start_seconds <= seconds <= cue.end_seconds and cue.text]
    return " ".join(matches)


def write_srt(path: Path, cues: list[SubtitleCue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n"
            f"{seconds_to_timestamp(cue.start_seconds).replace('.', ',')} --> "
            f"{seconds_to_timestamp(cue.end_seconds).replace('.', ',')}\n"
            f"{cue.text}"
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def parse_bilibili_subtitle_json(path: Path) -> list[SubtitleCue]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return []
    body = data.get("body") if isinstance(data, dict) else None
    if not isinstance(body, list):
        return []

    cues: list[SubtitleCue] = []
    for item in body:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content") or item.get("text") or "").strip()
        if not text:
            continue
        start = _optional_float(item.get("from") if "from" in item else item.get("start"))
        end = _optional_float(item.get("to") if "to" in item else item.get("end"))
        if start is None or end is None or end <= start:
            continue
        cues.append(SubtitleCue(start_seconds=start, end_seconds=end, text=text))
    return cues


def _strip_vtt_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip("\ufeff").strip().upper().startswith("WEBVTT"):
        return "\n".join(lines[1:])
    return text


def _clean_timed_text_line(line: str) -> str:
    return re.sub(r"<[^>]+>", "", line).strip()


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
