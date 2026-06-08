from __future__ import annotations

from pathlib import Path
from typing import Callable

from sceneweaver.analysis.tags import build_card_tags, read_scenes_analysis_with_tags
from sceneweaver.retrieval.usecase import build_script_usecase
from sceneweaver.schemas import ExperienceCard, SceneAnalysis, ScenesAnalysis
from sceneweaver.storage.json_store import write_jsonl

LogFn = Callable[[str], None]


def extract_experience_cards(
    output_dir: Path,
    *,
    force: bool = False,
    log: LogFn | None = None,
) -> list[ExperienceCard]:
    output_dir = output_dir.resolve()
    output_path = output_dir / "analysis" / "experience_cards.jsonl"
    if output_path.exists() and not force:
        from sceneweaver.storage.json_store import read_jsonl

        cards = read_jsonl(output_path, ExperienceCard)
        if log:
            log(f"Reused existing experience cards: {len(cards)}")
        return cards

    scenes = read_scenes_analysis_with_tags(output_dir / "analysis" / "scenes.json", write_back=True)
    cards = build_experience_cards(scenes)
    write_jsonl(output_path, cards)
    if log:
        log(f"Experience cards written: {output_path}")
        log(f"Experience cards extracted: {len(cards)}")
    return cards


def build_experience_cards(scenes: ScenesAnalysis) -> list[ExperienceCard]:
    cards: list[ExperienceCard] = []
    next_id = 1
    for scene in scenes.scenes:
        for candidate in scene.experience_candidates:
            card = _build_card(
                card_id=f"exp_{next_id:06d}",
                source_video_id=scenes.video_id,
                scene=scene,
                keywords=candidate.keywords,
                underlying_emotion=candidate.emotion,
                narrative_logic=candidate.narrative_logic,
                director_strategy=candidate.narrative_logic,
                shooting_techniques=candidate.techniques,
                visual_symbols=candidate.keywords,
                copywriting_tone=_copywriting_tone(scene),
                reuse_condition=candidate.reuse_condition,
                confidence=min(0.9, max(0.45, scene.emotion_temperature)),
            )
            cards.append(card)
            next_id += 1
    return cards


def _build_card(
    *,
    card_id: str,
    source_video_id: str,
    scene: SceneAnalysis,
    keywords: list[str],
    underlying_emotion: str,
    narrative_logic: str,
    director_strategy: str,
    shooting_techniques: list[str],
    visual_symbols: list[str],
    copywriting_tone: str,
    reuse_condition: str,
    confidence: float,
) -> ExperienceCard:
    placeholder = ExperienceCard(
        card_id=card_id,
        source_video_id=source_video_id,
        source_scene_ids=[scene.scene_id],
        tags=scene.tags,
        keywords=keywords,
        underlying_emotion=underlying_emotion,
        narrative_logic=narrative_logic,
        director_strategy=director_strategy,
        shooting_techniques=shooting_techniques,
        visual_symbols=visual_symbols,
        copywriting_tone=copywriting_tone,
        avoid=["避免脱离原始 scene 证据的空泛口号"],
        emotion_temperature_range=_temperature_range(scene.emotion_temperature),
        reuse_condition=reuse_condition,
        script_usecase=build_script_usecase(
            scene.tags,
            text=" ".join(
                [
                    " ".join(keywords),
                    underlying_emotion,
                    narrative_logic,
                    director_strategy,
                    " ".join(shooting_techniques),
                    " ".join(visual_symbols),
                    copywriting_tone,
                    reuse_condition,
                ]
            ),
            base_confidence=confidence,
        ),
        confidence=confidence,
    )
    return placeholder.model_copy(update={"tags": build_card_tags(placeholder)})


def _copywriting_tone(scene: SceneAnalysis) -> str:
    interpretation = scene.director_interpretation
    return f"{interpretation.brand_personality_signal}；{interpretation.emotional_function}"


def _temperature_range(value: float) -> tuple[float, float]:
    low = round(max(0.0, value - 0.15), 2)
    high = round(min(1.0, value + 0.15), 2)
    return (low, high)
