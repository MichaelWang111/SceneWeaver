from __future__ import annotations


def seconds_to_timestamp(seconds: float, separator: str = ".") -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def timestamp_to_seconds(value: str) -> float:
    normalized = value.strip().replace(",", ".")
    hours, minutes, seconds = normalized.split(":")
    secs, millis = seconds.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(millis) / 1000

