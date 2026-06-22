from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sceneweaver.user_api import (
    DEFAULT_CHANNEL_POLICY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLANNER,
    DEFAULT_RANKING_KEY,
    USER_INGEST_SCENE_ANALYSIS_MODEL,
    generate_script,
    ingest_video,
    search_scenes,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            result = run_ingest(args)
        elif args.command == "search":
            result = run_search(args)
        elif args.command == "script":
            result = run_script(args)
        else:
            parser.print_help()
            return 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    if bool(getattr(args, "markdown_only", False)):
        print(result.get("script", {}).get("script_markdown", ""))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sceneweaver-user",
        description="Stable user CLI over SceneWeaver video ingestion and Retrieval Lab scene search.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Parse one video into a SceneWeaver retrieval data source.")
    ingest.add_argument("source", help="Local video path by default, or a URL when --source-type=url is set.")
    ingest.add_argument("--source-type", choices=["auto", "file", "url"], default="auto")
    ingest.add_argument("--video-id", default="", help="Stable id to use under the output root.")
    ingest.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ingest.add_argument("--output-dir", type=Path, default=None)
    ingest.add_argument("--subtitle", type=Path, default=None, help="Optional SRT/VTT/Bilibili subtitle JSON file.")
    ingest.add_argument("--scene-threshold", type=float, default=27.0)
    ingest.add_argument("--limit", type=int, default=None, help="Analyze only the first N scene packages.")
    ingest.add_argument("--concurrency", "--workers", type=int, default=1)
    ingest.add_argument("--timeout-seconds", type=float, default=180.0)
    ingest.add_argument("--retries", type=int, default=0)
    ingest.add_argument("--scene-analysis-model", default=USER_INGEST_SCENE_ANALYSIS_MODEL)
    ingest.add_argument("--frame-workers", type=int, default=None)
    ingest.add_argument("--split-video", action="store_true")
    ingest.add_argument("--burn-subtitles", action="store_true")
    ingest.add_argument("--force", action="store_true")
    ingest.add_argument("--no-analyze", action="store_true", help="Only package scenes and sampled frames; skip Vision LLM.")
    ingest.add_argument("--no-extract-cards", action="store_true", help="Skip experience card extraction after analysis.")
    ingest.add_argument("--quiet", action="store_true")

    search = subparsers.add_parser("search", help="Search ingested SceneWeaver scene cards with Retrieval Lab.")
    search.add_argument("query", help="Keyword brief or natural-language scene request.")
    search.add_argument("--source", "--cards", dest="sources", type=Path, action="append", default=[])
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--candidate-depth", type=int, default=100)
    search.add_argument("--planner", default=DEFAULT_PLANNER)
    search.add_argument("--planner-cache", type=Path, default=None)
    search.add_argument("--ranking-key", default=DEFAULT_RANKING_KEY)
    search.add_argument("--channel-policy", default=DEFAULT_CHANNEL_POLICY)
    search.add_argument("--run-name", default="user_search")
    search.add_argument("--no-payload", action="store_true")
    search.add_argument("--include-channels", action="store_true")

    script = subparsers.add_parser("script", help="Generate a script draft from Retrieval Lab scene references.")
    script.add_argument("query", help="Keyword brief used to retrieve director experience references.")
    script.add_argument("--source", "--cards", dest="sources", type=Path, action="append", default=[])
    script.add_argument("--brief", "--script-brief", dest="script_brief", default="", help="Creative goal for the generated script.")
    script.add_argument("--top-k", type=int, default=5)
    script.add_argument("--duration-seconds", type=int, default=None)
    script.add_argument("--tone", default="")
    script.add_argument("--audience", default="")
    script.add_argument("--must-include", default="")
    script.add_argument("--avoid", default="")
    script.add_argument("--candidate-depth", type=int, default=100)
    script.add_argument("--planner", default=DEFAULT_PLANNER)
    script.add_argument("--planner-cache", type=Path, default=None)
    script.add_argument("--ranking-key", default=DEFAULT_RANKING_KEY)
    script.add_argument("--channel-policy", default=DEFAULT_CHANNEL_POLICY)
    script.add_argument("--run-name", default="script_generation_search")
    script.add_argument("--prompt", dest="prompt_path", type=Path, default=None)
    script.add_argument("--timeout-seconds", type=float, default=None)
    script.add_argument("--retries", type=int, default=0)
    script.add_argument("--max-tokens", type=int, default=None)
    script.add_argument("--markdown-only", action="store_true")

    return parser


def run_ingest(args: argparse.Namespace) -> dict[str, Any]:
    log = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    return ingest_video(
        args.source,
        source_type=args.source_type,
        output_root=args.output_root,
        output_dir=args.output_dir,
        video_id=args.video_id or None,
        scene_threshold=args.scene_threshold,
        subtitle_path=args.subtitle,
        split_video=args.split_video,
        force=args.force,
        frame_workers=args.frame_workers,
        burn_subtitles=args.burn_subtitles,
        analyze=not args.no_analyze,
        extract_cards=(not args.no_extract_cards and not args.no_analyze),
        limit=args.limit,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        scene_analysis_model=args.scene_analysis_model,
        log=log,
    )


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    sources = args.sources or [DEFAULT_OUTPUT_ROOT]
    return search_scenes(
        args.query,
        sources,
        top_k=args.top_k,
        candidate_depth=args.candidate_depth,
        planner=args.planner,
        planner_cache=args.planner_cache,
        ranking_key=args.ranking_key,
        channel_policy=args.channel_policy,
        run_name=args.run_name,
        include_payload=not args.no_payload,
        include_channels=args.include_channels,
    )


def run_script(args: argparse.Namespace) -> dict[str, Any]:
    sources = args.sources or [DEFAULT_OUTPUT_ROOT]
    return generate_script(
        args.query,
        sources,
        script_brief=args.script_brief,
        top_k=args.top_k,
        duration_seconds=args.duration_seconds,
        tone=args.tone,
        audience=args.audience,
        must_include=args.must_include,
        avoid=args.avoid,
        candidate_depth=args.candidate_depth,
        planner=args.planner,
        planner_cache=args.planner_cache,
        ranking_key=args.ranking_key,
        channel_policy=args.channel_policy,
        run_name=args.run_name,
        prompt_path=args.prompt_path,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
