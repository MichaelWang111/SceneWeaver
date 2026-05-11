from __future__ import annotations

from sceneweaver.analysis.tags import (
    ExperienceCardMatch,
    RetrievalResult,
    TagMatch,
    build_card_tags,
    build_film_tags,
    build_query_tags,
    build_scene_tags,
    generate_analysis_tags,
    match_experience_cards,
    retrieve_experience_card_matches_from_jsonl,
    retrieve_experience_cards,
    retrieve_experience_cards_from_jsonl,
)
from sceneweaver.schemas.fingerprint import FilmFingerprint, SceneFingerprint

FingerprintMatch = TagMatch


def build_query_fingerprint(*args, **kwargs):
    return build_query_tags(*args, **kwargs)


def build_card_fingerprint(*args, **kwargs):
    return build_card_tags(*args, **kwargs)


def build_scene_fingerprint(scene, *, source_video_id: str = ""):
    return SceneFingerprint(
        scene_id=scene.scene_id,
        source_video_id=source_video_id,
        fingerprint=build_scene_tags(scene),
    )


def build_film_fingerprint(*, video_id: str, scene_fingerprints: list[SceneFingerprint]):
    scenes = []
    for scene_fingerprint in scene_fingerprints:
        scenes.append(scene_fingerprint)
    tags = build_film_tags(
        [
            type(
                "_SceneWithTags",
                (),
                {"tags": scene_fingerprint.fingerprint},
            )()
            for scene_fingerprint in scenes
        ]
    )
    return FilmFingerprint(
        video_id=video_id,
        scene_count=len(scenes),
        fingerprint=tags,
        scenes=scenes,
    )


def generate_scene_fingerprints(*args, **kwargs):
    return generate_analysis_tags(*args, **kwargs)
