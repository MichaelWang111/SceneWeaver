from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_REVIEW_ROOT = Path(__file__).resolve().parent / "retrieval_review"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "eval_inputs" / "review_generated_inputs.json"

STAGE_WORDS = {
    "opening": "开场",
    "setup": "铺垫",
    "character_intro": "人物出场",
    "team_work": "团队协作",
    "technology_entrance": "技术入场",
    "technology_showcase": "技术展示",
    "scale_reveal": "规模揭示",
    "value_expression": "价值表达",
    "outcome": "结果落地",
    "growth": "成长升温",
    "ending": "结尾收束",
    "transition": "过渡",
}

INDUSTRY_WORDS = {
    "medical_technology_and_digital_health": "医疗科技与数字健康",
    "energy_and_utilities": "能源与公共事业",
    "automotive_and_mobility": "汽车与出行",
    "financial_services_and_insurance": "金融服务与保险",
    "consumer_retail_and_supply_chain": "消费零售与供应链",
    "technology_recruitment": "科技招聘",
}

PURPOSE_WORDS = {
    "attract_talent": "吸引人才",
    "bridge_sections": "承接上下段落",
    "build_reality": "建立真实感",
    "build_trust": "建立信任",
    "close_loop": "完成首尾闭环",
    "connect_expertise": "连接专业资源",
    "connect_feedback_to_mission": "把真实反馈和使命感连接起来",
    "establish_context": "建立环境和背景",
    "establish_need": "建立需求",
    "establish_problem": "建立问题",
    "express_value": "表达品牌价值",
    "general_expression": "通用表达",
    "human_centered_ai": "表现人本技术",
    "humanize_professional": "让专业场景更有人味",
    "introduce_people": "介绍人物",
    "introduce_tool": "引入工具",
    "keep_human_warmth": "保留人的温度",
    "land_value": "让价值落地",
    "leave_trust": "留下可信赖感",
    "prove_capability": "证明能力",
    "prove_organization_capability": "证明组织能力",
    "reduce_uncertainty": "降低不确定性",
    "show_collaboration": "表现协作",
    "show_distance": "表现距离感",
    "show_face_to_face_communication": "表现面对面沟通",
    "show_global_network": "表现全球网络",
    "show_growth": "表现成长",
    "show_long_termism": "表现长期主义",
    "show_network": "表现网络能力",
    "show_outcome": "表现结果",
    "show_pressure": "表现压力",
    "show_reality": "表现真实现场",
    "show_scale": "表现规模",
    "show_team": "表现团队",
    "show_technology": "表现技术能力",
    "stabilize_emotion": "稳定情绪",
    "avoid_overclaim": "避免夸大承诺",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock retrieval eval user inputs from review fixtures.")
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit-fixtures", type=int, default=0, help="0 means all fixtures.")
    args = parser.parse_args()

    dataset = generate_eval_inputs(args.review_root, limit_fixtures=args.limit_fixtures or None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {args.output}")
    print(f"cases: {len(dataset['cases'])}")


def generate_eval_inputs(review_root: Path = DEFAULT_REVIEW_ROOT, *, limit_fixtures: int | None = None) -> dict[str, Any]:
    items = load_review_items(review_root, limit_fixtures=limit_fixtures)
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        alternate = pick_alternate(items, index)
        cases.append(build_simple_positive(item))
        cases.append(build_hard_positive(item))
        cases.append(build_hard_negative(item, alternate))
    return {
        "dataset_id": "review_generated_retrieval_eval_inputs",
        "source_layer": "retrieval_review",
        "generation_policy": {
            "simple_positive": "接近复盘层一句话用途的简单正例。",
            "hard_positive": "根据脚本阶段、创作目的、标签和行业语境改写的困难正例。",
            "hard_negative": "保留部分相似视觉元素，但明确要求另一个脚本阶段或用途的困难负例。",
        },
        "case_count": len(cases),
        "cases": cases,
    }


def load_review_items(review_root: Path, *, limit_fixtures: int | None = None) -> list[dict[str, Any]]:
    manifest_path = review_root / "collection_manifest.json"
    if manifest_path.exists():
        fixture_ids = json.loads(manifest_path.read_text(encoding="utf-8"))["fixtures"]
    else:
        fixture_ids = sorted(path.name for path in review_root.iterdir() if path.is_dir())
    if limit_fixtures is not None:
        fixture_ids = fixture_ids[:limit_fixtures]

    rows: list[dict[str, Any]] = []
    for fixture_id in fixture_ids:
        fixture_dir = review_root / fixture_id
        manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
        retrieval = json.loads((fixture_dir / "retrieval.json").read_text(encoding="utf-8"))
        for item in retrieval["items"]:
            row = {
                "fixture_id": fixture_id,
                "video_id": retrieval["video_id"],
                "title": manifest["title"],
                "industry": manifest["industry"],
                "style": manifest["film_style"],
                "company_profile": manifest.get("company_profile", ""),
                "scene_id": item["scene_id"],
                "retrieval_id": item["retrieval_id"],
                "script_stage": item.get("script_stage", "general"),
                "creative_purpose": item.get("creative_purpose", []),
                "script_use_sentence": item.get("script_use_sentence", ""),
                "llm_tags": item.get("llm_tags", {}),
                "embedding_texts": item.get("embedding_texts", {}),
            }
            rows.append(row)
    return rows


def build_simple_positive(item: dict[str, Any]) -> dict[str, Any]:
    stage = stage_word(item)
    return base_case(
        case_type="simple_positive",
        item=item,
        user_input=f"我需要一个{stage}镜头：{item['script_use_sentence']}",
        difficulty="easy",
        expected_relation="should_match",
        notes="Intentionally close to the review script-use sentence.",
    )


def build_hard_positive(item: dict[str, Any]) -> dict[str, Any]:
    stage = stage_word(item)
    tags = flatten_tags(item["llm_tags"])
    tag_hint = "、".join(tags[:5])
    purpose_hint = "、".join(purpose_words(item))
    industry = industry_word(item)
    return base_case(
        case_type="hard_positive",
        item=item,
        user_input=(
            f"想找一个世界500强质感的{stage}段落，不要直白讲道理。"
            f"画面可以有{tag_hint}这类元素，重点是{purpose_hint}，"
            f"要能服务于{industry}的品牌表达。"
        ),
        difficulty="hard",
        expected_relation="should_match",
        notes="根据标签、脚本阶段和中文创作目的改写，不直接复制目标 summary。",
    )


def build_hard_negative(item: dict[str, Any], alternate: dict[str, Any]) -> dict[str, Any]:
    tags = flatten_tags(item["llm_tags"])
    alternate_stage = stage_word(alternate)
    alternate_purpose = "、".join(purpose_words(alternate))
    negative_hint = hard_negative_hint(item, alternate)
    return {
        **base_case(
            case_type="hard_negative",
            item=item,
            user_input=(
                f"画面可以借用{ '、'.join(tags[:4]) }这类相似元素，"
                f"但不要做成{stage_word(item)}。"
                f"我真正要的是{alternate_stage}，重点是{alternate_purpose}，"
                f"{negative_hint}"
            ),
            difficulty="hard",
            expected_relation="should_not_match",
            notes="保留目标的部分表面元素，但自然地要求另一个脚本阶段，不引用另一个 summary 原文。",
        ),
        "expected_prefer": target_ref(alternate),
    }


def base_case(
    *,
    case_type: str,
    item: dict[str, Any],
    user_input: str,
    difficulty: str,
    expected_relation: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "case_id": f"{case_type}__{item['fixture_id']}__{item['scene_id']}",
        "case_type": case_type,
        "difficulty": difficulty,
        "user_input": user_input,
        "expected_relation": expected_relation,
        "target": target_ref(item),
        "target_summary": item["script_use_sentence"],
        "target_tags_text": tags_text(item),
        "target_embedding_texts": item.get("embedding_texts", {}),
        "notes": notes,
    }


def target_ref(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": item["fixture_id"],
        "video_id": item["video_id"],
        "scene_id": item["scene_id"],
        "retrieval_id": item["retrieval_id"],
        "script_stage": item["script_stage"],
        "creative_purpose": item.get("creative_purpose", []),
        "title": item["title"],
        "industry": item["industry"],
        "style": item["style"],
    }


def pick_alternate(items: list[dict[str, Any]], index: int) -> dict[str, Any]:
    current = items[index]
    same_fixture = [
        item
        for item in items
        if item["fixture_id"] == current["fixture_id"] and item["scene_id"] != current["scene_id"]
    ]
    for candidate in same_fixture:
        if candidate["script_stage"] != current["script_stage"]:
            return candidate
    return items[(index + 1) % len(items)]


def stage_word(item: dict[str, Any]) -> str:
    return STAGE_WORDS.get(item.get("script_stage", ""), item.get("script_stage", "某个"))


def industry_word(item: dict[str, Any]) -> str:
    return INDUSTRY_WORDS.get(item.get("industry", ""), item.get("industry", "企业"))


def purpose_words(item: dict[str, Any], *, limit: int = 3) -> list[str]:
    words = []
    for purpose in item.get("creative_purpose", [])[:limit]:
        words.append(PURPOSE_WORDS.get(purpose, purpose.replace("_", " ")))
    return words or ["表达清晰的创作用途"]


def hard_negative_hint(item: dict[str, Any], alternate: dict[str, Any]) -> str:
    current_stage = item.get("script_stage", "")
    alternate_stage = alternate.get("script_stage", "")
    if current_stage == "opening":
        return "不要强调开场气势或问题建立，而要让观众看到后续情节的具体铺垫。"
    if current_stage == "setup":
        return "不要停留在需求铺垫，而要更像一段能直接进入主题的画面。"
    if current_stage == "team_work":
        return "不要拍成团队协作或会议共创，而要突出另一个明确的叙事功能。"
    if current_stage == "technology_showcase":
        return "不要像技术展示或功能说明，而要让画面承担别的叙事任务。"
    if current_stage == "value_expression":
        return "不要靠价值表达或信任感收束，而要服务于另一个更具体的段落目标。"
    if alternate_stage == "ending":
        return "最好能有片尾收束感，而不是中段说明感。"
    if alternate_stage == "growth":
        return "最好能看出人物或组织的变化，而不是单纯说明现状。"
    return "希望它的脚本位置和情绪功能明显不同。"


def flatten_tags(tags: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("entities", "relations", "actions_and_expressions", "background_and_setting", "social_relations"):
        for value in tags.get(key, []):
            if isinstance(value, str) and value not in values:
                values.append(value)
    return values


def tags_text(item: dict[str, Any]) -> str:
    return " ".join(flatten_tags(item["llm_tags"]))


if __name__ == "__main__":
    main()
