from __future__ import annotations

import argparse
import json
import mimetypes
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from sceneweaver.user_api import (
    DEFAULT_OUTPUT_ROOT,
    USER_INGEST_SCENE_ANALYSIS_MODEL,
    generate_script,
    ingest_video,
    search_scenes,
)

WORKSPACE_ROOT = Path.cwd().resolve()
UI_ROOT = Path(__file__).resolve().parent / "ui"
UPLOAD_ROOT = WORKSPACE_ROOT / ".tmp" / "user_uploads"


class SceneWeaverUIHandler(BaseHTTPRequestHandler):
    server_version = "SceneWeaverUI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_file(UI_ROOT / "index.html", content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self._send_json(status_payload())
            return
        if parsed.path == "/api/sources":
            self._send_json({"sources": list_sources()})
            return
        if parsed.path == "/api/file":
            self._send_workspace_file(parsed.query)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/ingest":
                payload = self._read_json_body()
                result = ingest_video_from_payload(payload)
                self._send_json(result)
                return
            if parsed.path == "/api/upload-ingest":
                fields, files = self._read_multipart_body()
                result = ingest_uploaded_video(fields, files)
                self._send_json(result)
                return
            if parsed.path == "/api/search":
                payload = self._read_json_body()
                result = search_scenes_from_payload(payload)
                self._send_json(result)
                return
            if parsed.path == "/api/generate-script":
                payload = self._read_json_body()
                result = generate_script_from_payload(payload)
                self._send_json(result)
                return
        except Exception as exc:  # pragma: no cover - exercised through manual UI flows.
            self._send_json({"status": "error", "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} {format % args}")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length else b""

    def _read_json_body(self) -> dict[str, Any]:
        body = self._read_body()
        if not body:
            return {}
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _read_multipart_body(self) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        content_type = self.headers.get("Content-Type", "")
        match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
        if not match:
            raise ValueError("multipart boundary missing")
        boundary = match.group("boundary").strip().strip('"').encode("utf-8")
        body = self._read_body()
        return parse_multipart(body, boundary)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "file not found"}, status=HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        resolved_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", resolved_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_workspace_file(self, query: str) -> None:
        params = parse_qs(query)
        raw_path = params.get("path", [""])[0]
        if not raw_path:
            self._send_json({"error": "path is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        path = Path(unquote(raw_path)).resolve()
        if not is_within(path, WORKSPACE_ROOT):
            self._send_json({"error": "path is outside workspace"}, status=HTTPStatus.FORBIDDEN)
            return
        self._send_file(path)


def ingest_video_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source = str(payload.get("source") or "").strip()
    if not source:
        raise ValueError("source is required")
    return ingest_video(
        source,
        source_type=str(payload.get("source_type") or "auto"),
        output_root=Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT),
        output_dir=optional_path(payload.get("output_dir")),
        video_id=optional_text(payload.get("video_id")),
        scene_threshold=float(payload.get("scene_threshold") or 27.0),
        subtitle_path=optional_path(payload.get("subtitle_path")),
        split_video=bool(payload.get("split_video", False)),
        force=bool(payload.get("force", False)),
        frame_workers=optional_int(payload.get("frame_workers")),
        burn_subtitles=bool(payload.get("burn_subtitles", False)),
        analyze=bool(payload.get("analyze", True)),
        extract_cards=bool(payload.get("extract_cards", True)) and bool(payload.get("analyze", True)),
        limit=optional_int(payload.get("limit")),
        concurrency=int(payload.get("concurrency") or 1),
        timeout_seconds=float(payload.get("timeout_seconds") or 180.0),
        retries=int(payload.get("retries") or 0),
        scene_analysis_model=str(payload.get("scene_analysis_model") or USER_INGEST_SCENE_ANALYSIS_MODEL),
    )


def ingest_uploaded_video(fields: dict[str, str], files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    upload = files.get("video")
    if not upload:
        raise ValueError("video upload is required")
    options = json.loads(fields.get("options") or "{}")
    if not isinstance(options, dict):
        raise ValueError("options must be a JSON object")
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(str(upload.get("filename") or "video.mp4"))
    upload_path = UPLOAD_ROOT / f"{time.strftime('%Y%m%d_%H%M%S')}_{filename}"
    upload_path.write_bytes(upload.get("content") or b"")
    options["source"] = str(upload_path)
    options["source_type"] = "file"
    return ingest_video_from_payload(options)


def search_scenes_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    sources = payload.get("sources") or [DEFAULT_OUTPUT_ROOT]
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    return search_scenes(
        query,
        sources,
        top_k=int(payload.get("top_k") or 5),
        candidate_depth=int(payload.get("candidate_depth") or 100),
        planner=str(payload.get("planner") or "multi_query"),
        planner_cache=optional_path(payload.get("planner_cache")),
        ranking_key=str(payload.get("ranking_key") or "hybrid_rrf_constraints_signature"),
        channel_policy=str(payload.get("channel_policy") or "all"),
        run_name=str(payload.get("run_name") or "ui_search"),
        include_payload=bool(payload.get("include_payload", True)),
        include_channels=bool(payload.get("include_channels", False)),
    )


def generate_script_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    sources = payload.get("sources") or [DEFAULT_OUTPUT_ROOT]
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    return generate_script(
        query,
        sources,
        script_brief=str(payload.get("script_brief") or payload.get("brief") or ""),
        top_k=int(payload.get("top_k") or 5),
        duration_seconds=optional_int(payload.get("duration_seconds")),
        tone=str(payload.get("tone") or ""),
        audience=str(payload.get("audience") or ""),
        must_include=str(payload.get("must_include") or ""),
        avoid=str(payload.get("avoid") or ""),
        candidate_depth=int(payload.get("candidate_depth") or 100),
        planner=str(payload.get("planner") or "multi_query"),
        planner_cache=optional_path(payload.get("planner_cache")),
        ranking_key=str(payload.get("ranking_key") or "hybrid_rrf_constraints_signature"),
        channel_policy=str(payload.get("channel_policy") or "all"),
        run_name=str(payload.get("run_name") or "ui_script_generation"),
        prompt_path=optional_path(payload.get("prompt_path")),
        timeout_seconds=optional_float(payload.get("timeout_seconds")),
        retries=int(payload.get("retries") or 0),
        max_tokens=optional_int(payload.get("max_tokens")),
    )


def status_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "workspace_root": str(WORKSPACE_ROOT),
        "default_output_root": str((WORKSPACE_ROOT / DEFAULT_OUTPUT_ROOT).resolve()),
        "ui_root": str(UI_ROOT),
    }


def list_sources() -> list[dict[str, Any]]:
    root = WORKSPACE_ROOT / DEFAULT_OUTPUT_ROOT
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        cards = path / "analysis" / "experience_cards.jsonl"
        packages = path / "packages" / "scene_packages.json"
        rows.append(
            {
                "video_id": path.name,
                "path": str(path.resolve()),
                "cards_exists": cards.exists(),
                "cards_path": str(cards.resolve()),
                "card_count": count_jsonl(cards),
                "packages_exists": packages.exists(),
                "updated_at": int(path.stat().st_mtime),
            }
        )
    return rows


def parse_multipart(body: bytes, boundary: bytes) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    fields: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}
    marker = b"--" + boundary
    for raw_part in body.split(marker):
        part = raw_part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        if content.endswith(b"\r\n"):
            content = content[:-2]
        headers = parse_part_headers(header_blob)
        disposition = headers.get("content-disposition", "")
        name = disposition_value(disposition, "name")
        filename = disposition_value(disposition, "filename")
        if not name:
            continue
        if filename:
            files[name] = {
                "filename": filename,
                "content_type": headers.get("content-type", "application/octet-stream"),
                "content": content,
            }
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def parse_part_headers(header_blob: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in header_blob.decode("utf-8", errors="replace").split("\r\n"):
        key, separator, value = line.partition(":")
        if separator:
            headers[key.strip().lower()] = value.strip()
    return headers


def disposition_value(disposition: str, key: str) -> str:
    match = re.search(rf'{re.escape(key)}="(?P<value>[^"]*)"', disposition)
    return match.group("value") if match else ""


def safe_filename(value: str) -> str:
    name = Path(value).name or "video.mp4"
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._")
    return cleaned or "video.mp4"


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def optional_path(value: Any) -> Path | None:
    text = optional_text(value)
    return Path(text) if text else None


def optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.strip())


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SceneWeaver local UI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), SceneWeaverUIHandler)
    print(f"SceneWeaver UI running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping SceneWeaver UI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
