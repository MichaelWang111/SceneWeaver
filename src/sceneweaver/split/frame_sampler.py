from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import subprocess
from pathlib import Path
from typing import Callable

from sceneweaver.split.scene_detector import SceneSpan
from sceneweaver.split.subtitle_segmenter import SubtitleCue, cue_text_at
from sceneweaver.split.timecode import seconds_to_timestamp

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class FrameSampleTask:
    scene_id: str
    label: str
    seconds: float
    frame_path: Path
    subtitle_text: str = ""


def sample_scene_frames(
    video_path: Path,
    scenes: list[SceneSpan],
    frames_dir: Path,
    runner: Runner = subprocess.run,
    force: bool = False,
    subtitle_cues: list[SubtitleCue] | None = None,
    burn_subtitles: bool = False,
    frame_workers: int | None = None,
) -> dict[str, dict[str, str]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frame_workers is not None and frame_workers < 1:
        raise ValueError("frame_workers must be >= 1")

    results: dict[str, dict[str, str]] = {}
    tasks: list[FrameSampleTask] = []

    for scene in scenes:
        points = {
            "start": scene.start_seconds + 0.05,
            "middle": (scene.start_seconds + scene.end_seconds) / 2,
            "end": max(scene.start_seconds, scene.end_seconds - 0.05),
        }
        scene_frames: dict[str, str] = {}
        for label, seconds in points.items():
            frame_path = frames_dir / f"{scene.scene_id}_{label}.jpg"
            subtitle_text = cue_text_at(subtitle_cues or [], seconds) if burn_subtitles else ""
            task = FrameSampleTask(
                scene_id=scene.scene_id,
                label=label,
                seconds=seconds,
                frame_path=frame_path,
                subtitle_text=subtitle_text,
            )
            if _needs_frame_task(task, force=force):
                tasks.append(task)
            scene_frames[label] = frame_path.relative_to(frames_dir.parent).as_posix()
        results[scene.scene_id] = scene_frames

    _run_frame_tasks(video_path, tasks, runner=runner, force=force, frame_workers=frame_workers)
    return results


def _run_frame_tasks(
    video_path: Path,
    tasks: list[FrameSampleTask],
    *,
    runner: Runner,
    force: bool,
    frame_workers: int | None,
) -> None:
    if not tasks:
        return
    workers = _resolve_frame_workers(frame_workers, len(tasks))
    if workers == 1:
        for task in tasks:
            _process_frame_task(video_path, task, runner=runner, force=force)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_frame_task, video_path, task, runner=runner, force=force)
            for task in tasks
        ]
        for future in as_completed(futures):
            future.result()


def _needs_frame_task(task: FrameSampleTask, *, force: bool) -> bool:
    needs_extract = force or not _frame_file_ready(task.frame_path)
    needs_burn = bool(task.subtitle_text) and (needs_extract or not _has_subtitle_marker(task.frame_path))
    return needs_extract or needs_burn


def _process_frame_task(
    video_path: Path,
    task: FrameSampleTask,
    *,
    runner: Runner,
    force: bool,
) -> None:
    if force or not _frame_file_ready(task.frame_path):
        _extract_frame_with_fallbacks(video_path, task, runner=runner)
        _remove_subtitle_marker(task.frame_path)
    if task.subtitle_text and (force or not _has_subtitle_marker(task.frame_path)):
        burn_subtitle_into_frame(task.frame_path, task.subtitle_text)


def _resolve_frame_workers(requested: int | None, task_count: int) -> int:
    if task_count < 1:
        return 1
    if requested is not None:
        return min(requested, task_count)
    return min(task_count, max(1, min(8, os.cpu_count() or 4)))


def _extract_frame_with_fallbacks(video_path: Path, task: FrameSampleTask, *, runner: Runner) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for seconds in _frame_candidate_seconds(task.seconds):
        _remove_incomplete_frame(task.frame_path)
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
            "-update",
            "1",
            str(task.frame_path),
        ]
        try:
            runner(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            continue
        if _frame_file_ready(task.frame_path):
            return
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to extract frame: {task.frame_path}")


def _frame_candidate_seconds(seconds: float) -> list[float]:
    values: list[float] = []
    for candidate in (seconds, seconds - 0.25, seconds - 0.5, seconds - 1.0, 0.0):
        value = round(max(0.0, candidate), 3)
        if value not in values:
            values.append(value)
    return values


def _frame_file_ready(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _remove_incomplete_frame(path: Path) -> None:
    if _frame_file_ready(path):
        return
    if path.exists():
        path.unlink()

def burn_subtitle_into_frame(frame_path: Path, text: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to burn subtitles into sampled frames") from exc

    with Image.open(frame_path) as image:
        image = image.convert("RGBA")
        width, height = image.size
        font = _subtitle_font(width)
        wrapped = _wrap_subtitle_text(text, font, max_width=int(width * 0.88))
        if not wrapped:
            return

        draw_probe = ImageDraw.Draw(image)
        line_boxes = [draw_probe.textbbox((0, 0), line, font=font, stroke_width=2) for line in wrapped]
        line_height = max(box[3] - box[1] for box in line_boxes)
        line_gap = max(4, int(line_height * 0.2))
        block_height = len(wrapped) * line_height + (len(wrapped) - 1) * line_gap
        padding_y = max(12, int(height * 0.025))
        padding_x = max(16, int(width * 0.035))
        y0 = height - block_height - padding_y * 2

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            (padding_x, max(0, y0), width - padding_x, height - padding_y // 2),
            radius=max(6, int(width * 0.01)),
            fill=(0, 0, 0, 150),
        )
        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)
        y = max(0, y0) + padding_y
        for line, box in zip(wrapped, line_boxes):
            text_width = box[2] - box[0]
            x = (width - text_width) / 2
            draw.text(
                (x, y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 230),
            )
            y += line_height + line_gap

        rgb = image.convert("RGB")
        rgb.save(frame_path, quality=92)
    _write_subtitle_marker(frame_path)


def _subtitle_font(width: int):
    from PIL import ImageFont

    size = max(18, min(42, width // 30))
    for font_path in (
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _wrap_subtitle_text(text: str, font, max_width: int) -> list[str]:
    from PIL import Image, ImageDraw

    normalized = " ".join(text.split())
    if not normalized:
        return []
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    current = ""
    for char in normalized:
        candidate = current + char
        width = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2]
        if current and width > max_width:
            lines.append(current)
            current = char.lstrip()
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[-2:]


def _marker_path(frame_path: Path) -> Path:
    return frame_path.with_suffix(frame_path.suffix + ".subtitle")


def _has_subtitle_marker(frame_path: Path) -> bool:
    return _marker_path(frame_path).exists()


def _write_subtitle_marker(frame_path: Path) -> None:
    _marker_path(frame_path).write_text("burned\n", encoding="utf-8")


def _remove_subtitle_marker(frame_path: Path) -> None:
    marker_path = _marker_path(frame_path)
    if marker_path.exists():
        marker_path.unlink()
