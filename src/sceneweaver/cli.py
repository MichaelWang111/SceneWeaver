from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Optional

import typer

from sceneweaver.analysis.associate_analyzer import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    associate_input,
)
from sceneweaver.analysis.experience_extractor import extract_experience_cards
from sceneweaver.analysis.keyword_loop import run_keyword_loop
from sceneweaver.analysis.semantic import DEFAULT_EMBEDDING_MODEL, DEFAULT_SEMANTIC_WEIGHT
from sceneweaver.analysis.tags import (
    build_query_tags,
    generate_analysis_tags,
    retrieve_experience_card_matches_from_jsonl,
)
from sceneweaver.input.bilibili import extract_bvid
from sceneweaver.llm.client import LLMConfig
from sceneweaver.pipeline.mock_pipeline import run_mock_pipeline
from sceneweaver.pipeline.package_video import run_package_video
from sceneweaver.analysis.scene_analyzer import analyze_scene_packages
from sceneweaver.llm.client import VisionLLMClient

app = typer.Typer(help="SceneWeaver director experience analysis CLI.")

OUTPUT_ROOT = Path("outputs")
KEY_ASSOCIATES_DIR = OUTPUT_ROOT / "key_associates"
FILM_ANALYSIS_DIR = OUTPUT_ROOT / "film_analysis"
MOCK_OUTPUT_DIR = OUTPUT_ROOT / "mock"


@app.callback()
def main() -> None:
    """SceneWeaver director experience analysis CLI."""


@app.command("mock-run")
def mock_run(
    output: Path = typer.Option(
        MOCK_OUTPUT_DIR / "mock_video",
        "--output",
        "-o",
        help="Directory where mock artifacts will be written.",
    ),
    source_url: str = typer.Option(
        "https://www.bilibili.com/video/BVxxxx",
        "--source-url",
        help="Source URL recorded in mock artifacts.",
    ),
) -> None:
    """Generate a validated local mock pipeline output."""
    output_dir = run_mock_pipeline(output.resolve(), source_url=source_url)
    typer.echo(f"Mock artifacts written to: {output_dir}")


@app.command("package-video")
def package_video(
    url: str = typer.Argument(..., help="Bilibili video URL."),
    output: Path = typer.Option(
        FILM_ANALYSIS_DIR / "video_package",
        "--output",
        "-o",
        help="Directory where video package artifacts will be written.",
    ),
    scene_threshold: float = typer.Option(
        27.0,
        "--scene-threshold",
        help="PySceneDetect content detection threshold.",
    ),
    subtitle_path: Optional[Path] = typer.Option(
        None,
        "--subtitle",
        help="Optional existing SRT subtitle file to segment into scenes.",
    ),
    split_video: bool = typer.Option(
        False,
        "--split-video",
        help="Also split scene clips with PySceneDetect. Slower and not required for packages.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate video and frame artifacts even if files exist.",
    ),
) -> None:
    """Download and package a real Bilibili video into validated scene packages."""
    output_dir = run_package_video(
        url=url,
        output_dir=output,
        scene_threshold=scene_threshold,
        subtitle_path=subtitle_path,
        split_video=split_video,
        force=force,
        log=typer.echo,
    )
    typer.echo(f"Scene packages written to: {output_dir / 'packages'}")


@app.command("analyze-scenes")
def analyze_scenes(
    output: Path = typer.Argument(..., help="Video output directory containing packages/ and frames/."),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Analyze only the first N scene packages.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing analysis files. By default, existing scene results are reused and skipped.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom scene analysis prompt path.",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "--workers",
        min=1,
        help="Number of scene analysis requests to run in parallel.",
    ),
) -> None:
    """Send scene packages and frames to a vision LLM and write validated scene analyses."""
    scenes = analyze_scene_packages(
        output,
        limit=limit,
        force=update,
        prompt_path=prompt_path,
        max_workers=concurrency,
        log=typer.echo,
    )
    typer.echo(f"Scene analysis written to: {output.resolve() / 'analysis'}")
    typer.echo(f"Scenes analyzed: {scenes.scene_count}")


@app.command("fingerprint-scenes")
def fingerprint_scenes(
    output: Path = typer.Argument(..., help="Video output directory containing analysis/scenes.json."),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing tags in analysis files.",
    ),
) -> None:
    """Legacy command: add or refresh tags inside existing scene analyses."""
    scenes = generate_analysis_tags(output, force=update, log=typer.echo)
    typer.echo(f"Tags written to: {output.resolve() / 'analysis'}")
    typer.echo(f"Scenes tagged: {scenes.scene_count}")


@app.command("extract-experience")
def extract_experience(
    output: Path = typer.Argument(..., help="Video output directory containing analysis/scenes.json."),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing experience_cards.jsonl.",
    ),
) -> None:
    """Extract first-pass experience cards from scene analyses."""
    cards = extract_experience_cards(output, force=update, log=typer.echo)
    typer.echo(f"Experience cards written to: {output.resolve() / 'analysis' / 'experience_cards.jsonl'}")
    typer.echo(f"Experience cards extracted: {len(cards)}")


@app.command("retrieve-cards")
def retrieve_cards(
    output: Path = typer.Argument(..., help="Video output directory containing analysis/experience_cards.jsonl."),
    input_text: str = typer.Argument(..., help="Brief or keywords to match against experience cards."),
    top_k: int = typer.Option(
        5,
        "--top-k",
        min=1,
        help="Maximum number of matching experience cards to return.",
    ),
) -> None:
    """Retrieve grounded experience cards with an explainable tag match."""
    query_tags = build_query_tags(
        input_text,
        candidate_log_path=output.resolve() / "analysis" / "tag_candidates.jsonl",
    )
    result = retrieve_experience_card_matches_from_jsonl(
        query_tags,
        output.resolve() / "analysis" / "experience_cards.jsonl",
        top_k=top_k,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("llm-check")
def llm_check(
    prompt: str = typer.Argument(
        "hi",
        help="Short prompt used to verify that the configured LLM endpoint is reachable.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override the configured model for this check only.",
    ),
    timeout_seconds: float = typer.Option(
        30.0,
        "--timeout-seconds",
        min=1.0,
        help="Request timeout for the connectivity check.",
    ),
) -> None:
    """Send a minimal JSON request to the configured LLM endpoint and print the response."""
    client = VisionLLMClient()
    if model is not None:
        client.config = client.config.__class__(
            api_key=client.config.api_key,
            base_url=client.config.base_url,
            model=model,
            temperature=client.config.temperature,
            max_tokens=client.config.max_tokens,
            request_timeout_seconds=client.config.request_timeout_seconds,
            stream_idle_timeout_seconds=client.config.stream_idle_timeout_seconds,
            enable_thinking=client.config.enable_thinking,
            thinking_budget=client.config.thinking_budget,
        )
    typer.echo(
        "LLM check: "
        f"base_url={client.config.base_url!r}, model={client.config.model!r}, timeout_seconds={timeout_seconds:g}",
        err=True,
    )
    result = client.analyze_text_json(
        system_prompt="Return a one-field JSON object with key 'reply'.",
        user_prompt=prompt,
        timeout_seconds=timeout_seconds,
        retries=0,
    )
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("keyword-loop")
def keyword_loop(
    card_source: Path = typer.Argument(
        ...,
        help=(
            "Experience-card source: an experience_cards.jsonl file, a film output directory, "
            "or a collection directory such as outputs/film_analysis."
        ),
    ),
    input_text: str = typer.Argument(..., help="Brief or keywords to expand, tag, and match against experience cards."),
    top_k: int = typer.Option(
        5,
        "--top-k",
        min=1,
        help="Maximum number of matching experience cards to return.",
    ),
    just_tags: bool = typer.Option(
        False,
        "--just-tags",
        help="Use lightweight LLM tag expansion instead of the full director association prompt.",
    ),
    intent: bool = typer.Option(
        False,
        "--intent",
        "--core-intent",
        help="Use LLM creative-intent analysis for retrieval ranking instead of association or tag expansion.",
    ),
    intent_weight: float = typer.Option(
        3.0,
        "--intent-weight",
        min=0.0,
        help="Score multiplier for creative-intent must-match and avoid terms when --intent is enabled.",
    ),
    semantic: bool = typer.Option(
        False,
        "--semantic",
        help="Use local embedding similarity to rerank experience cards after tag scoring.",
    ),
    embedding_model: str = typer.Option(
        DEFAULT_EMBEDDING_MODEL,
        "--embedding-model",
        help="SentenceTransformer model used when --semantic is enabled.",
    ),
    semantic_weight: float = typer.Option(
        DEFAULT_SEMANTIC_WEIGHT,
        "--semantic-weight",
        min=0.0,
        help="Score multiplier for embedding cosine similarity when --semantic is enabled.",
    ),
    association_output: Optional[Path] = typer.Option(
        None,
        "--association-output",
        help="JSON file where the intermediate LLM association, tag, or intent analysis will be written.",
    ),
    result_output: Optional[Path] = typer.Option(
        None,
        "--result-output",
        help="Optional JSON file where the full keyword loop result will be written.",
    ),
    max_items: int = typer.Option(
        DEFAULT_MAX_ITEMS,
        "--max-items",
        min=8,
        max=120,
        help="Target number of association items. The LLM may vary within a useful creative range.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom associate prompt path.",
    ),
    timeout_seconds: float = typer.Option(
        DEFAULT_TIMEOUT_SECONDS,
        "--timeout-seconds",
        min=1.0,
        help="LLM request timeout in seconds.",
    ),
    retries: int = typer.Option(
        DEFAULT_RETRIES,
        "--retries",
        min=0,
        help="Retry count for timeout, connection, rate-limit, and 5xx failures.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print keyword-loop progress to stderr.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        "--flue",
        help="Stream raw LLM JSON chunks to stderr while preserving final validated JSON on stdout.",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking",
        help="Enable DashScope/Qwen thinking mode and stream reasoning_content to stderr.",
    ),
    thinking_budget: Optional[int] = typer.Option(
        None,
        "--thinking-budget",
        min=1,
        help="Maximum tokens for DashScope/Qwen thinking content. Implies --thinking.",
    ),
) -> None:
    """Expand one keyword brief, refresh tags, and retrieve matching experience cards."""
    request_thinking = thinking or thinking_budget is not None
    result = run_keyword_loop(
        input_text,
        card_source,
        association_output_path=association_output,
        result_output_path=result_output,
        top_k=top_k,
        just_tags=just_tags,
        intent=intent,
        intent_weight=intent_weight,
        semantic=semantic,
        embedding_model=embedding_model,
        semantic_weight=semantic_weight,
        max_items=max_items,
        prompt_path=prompt_path,
        timeout_seconds=timeout_seconds,
        retries=retries,
        log=(lambda message: typer.echo(f"[keyword-loop] {message}", err=True)) if debug or stream or request_thinking else None,
        stream_callback=_write_stderr_chunk if stream else None,
        reasoning_callback=_write_stderr_chunk if request_thinking else None,
        enable_thinking=True if request_thinking else None,
        thinking_budget=thinking_budget,
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if result_output is not None:
        typer.echo(f"Keyword loop result written to: {result_output.resolve()}", err=True)


@app.command("associate")
def associate(
    input_text: str = typer.Argument(..., help="Rough keywords or brief to expand into director associations."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="JSON file where association analysis will be written. Defaults to outputs/key_associates/.",
    ),
    max_items: int = typer.Option(
        DEFAULT_MAX_ITEMS,
        "--max-items",
        min=8,
        max=120,
        help="Target number of association items. The LLM may vary within a useful creative range.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom associate prompt path.",
    ),
    timeout_seconds: float = typer.Option(
        DEFAULT_TIMEOUT_SECONDS,
        "--timeout-seconds",
        min=1.0,
        help="LLM request timeout in seconds.",
    ),
    retries: int = typer.Option(
        DEFAULT_RETRIES,
        "--retries",
        min=0,
        help="Retry count for timeout, connection, rate-limit, and 5xx failures.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print request progress and provider metadata to stderr.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        "--flue",
        help="Stream raw LLM JSON chunks to stderr while preserving final validated JSON on stdout.",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking",
        help="Enable DashScope/Qwen thinking mode and stream reasoning_content to stderr.",
    ),
    thinking_budget: Optional[int] = typer.Option(
        None,
        "--thinking-budget",
        min=1,
        help="Maximum tokens for DashScope/Qwen thinking content. Implies --thinking.",
    ),
) -> None:
    """Expand rough keywords into structured director and screenwriter association material."""
    output_path = output or build_key_associate_output_path(input_text)
    request_thinking = thinking or thinking_budget is not None
    if debug or stream or request_thinking:
        config = LLMConfig.from_env()
        typer.echo(
            "Associate debug: "
            f"base_url={config.base_url!r}, model={config.model!r}, "
            f"output={str(output_path.resolve())!r}, "
            f"thinking={request_thinking!r}, thinking_budget={thinking_budget!r}",
            err=True,
        )
    if stream:
        typer.echo("[associate] Streaming raw LLM JSON chunks to stderr. Final validated JSON remains on stdout.", err=True)
    if request_thinking:
        typer.echo(
            "[associate] Streaming DashScope/Qwen reasoning_content to stderr. Final validated JSON remains on stdout.",
            err=True,
        )
    analysis = associate_input(
        input_text,
        output_path=output_path,
        max_items=max_items,
        prompt_path=prompt_path,
        timeout_seconds=timeout_seconds,
        retries=retries,
        log=(lambda message: typer.echo(f"[associate] {message}", err=True)) if debug or stream or request_thinking else None,
        stream_callback=_write_stderr_chunk if stream else None,
        reasoning_callback=_write_stderr_chunk if request_thinking else None,
        enable_thinking=True if request_thinking else None,
        thinking_budget=thinking_budget,
    )
    typer.echo(json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2))
    typer.echo(f"Association analysis written to: {output_path.resolve()}", err=True)


@app.command("run")
def run_pipeline(
    url: str = typer.Argument(..., help="Bilibili video URL."),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Analyze only the first N scene packages.",
    ),
    update: bool = typer.Option(
        False,
        "--update",
        "--force",
        help="Overwrite existing downloaded assets, packages, frames, and analysis results.",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "--workers",
        min=1,
        help="Number of scene analysis requests to run in parallel.",
    ),
    scene_threshold: float = typer.Option(
        27.0,
        "--scene-threshold",
        help="PySceneDetect content detection threshold.",
    ),
    subtitle_path: Optional[Path] = typer.Option(
        None,
        "--subtitle",
        help="Optional existing SRT subtitle file to segment into scenes.",
    ),
    split_video: bool = typer.Option(
        False,
        "--split-video",
        help="Also split scene clips with PySceneDetect. Slower and not required for packages.",
    ),
    prompt_path: Optional[Path] = typer.Option(
        None,
        "--prompt",
        help="Custom scene analysis prompt path.",
    ),
) -> None:
    """Run video packaging and scene analysis end to end with an auto-generated outputs directory."""
    bvid = extract_bvid(url)
    output_name = bvid or "bilibili_unknown"
    output_dir = FILM_ANALYSIS_DIR / output_name

    typer.echo(f"Output directory: {output_dir.resolve()}")
    typer.echo("Phase 1/3: Packaging video into scene packages...")
    packaged_dir = run_package_video(
        url=url,
        output_dir=output_dir,
        scene_threshold=scene_threshold,
        subtitle_path=subtitle_path,
        split_video=split_video,
        force=update,
        log=typer.echo,
    )

    typer.echo("Phase 2/3: Analyzing scene packages with tags...")
    scenes = analyze_scene_packages(
        packaged_dir,
        limit=limit,
        force=update,
        prompt_path=prompt_path,
        max_workers=concurrency,
        log=typer.echo,
    )
    typer.echo("Phase 3/3: Extracting experience cards...")
    cards = extract_experience_cards(
        packaged_dir,
        force=update,
        log=typer.echo,
    )
    typer.echo(f"Scene packages written to: {packaged_dir / 'packages'}")
    typer.echo(f"Scene analysis written to: {packaged_dir / 'analysis'}")
    typer.echo(f"Scenes analyzed: {scenes.scene_count}")
    typer.echo(f"Experience cards written to: {packaged_dir / 'analysis' / 'experience_cards.jsonl'}")
    typer.echo(f"Experience cards extracted: {len(cards)}")


def build_key_associate_output_path(input_text: str, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(input_text.strip().encode("utf-8")).hexdigest()[:8]
    slug = _safe_filename_slug(input_text)
    return KEY_ASSOCIATES_DIR / f"{timestamp}_{slug}_{digest}.json"


def _safe_filename_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    if not slug:
        return "associate"
    return slug[:32].strip("_") or "associate"


def _write_stderr_chunk(chunk: str) -> None:
    sys.stderr.write(chunk)
    sys.stderr.flush()


if __name__ == "__main__":
    app()
