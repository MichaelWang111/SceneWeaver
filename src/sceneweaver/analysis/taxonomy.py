from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from sceneweaver.schemas.tags import TagEvidence, TagProfile

TAG_TAXONOMY_VERSION = "director_tags_v1"
TAG_DIMENSIONS = (
    "emotion_core",
    "audience_projection",
    "narrative_function",
    "interaction_mode",
    "visual_motifs",
    "symbolic_logic",
    "rhythm_pattern",
)
FINGERPRINT_TAXONOMY_VERSION = TAG_TAXONOMY_VERSION
FINGERPRINT_DIMENSIONS = TAG_DIMENSIONS
GENERAL_EXPRESSION_TAG = "general_expression"


@dataclass(frozen=True)
class TaxonomyTag:
    tag: str
    aliases: tuple[str, ...]


TAXONOMY: dict[str, tuple[TaxonomyTag, ...]] = {
    "emotion_core": (
        TaxonomyTag("awakening", ("启示", "唤醒", "显影", "idea", "起源", "开始")),
        TaxonomyTag("ambition", ("野心", "向往", "未来", "突破", "机会", "潜力")),
        TaxonomyTag("belonging", ("归属", "团队", "共同", "一员", "协作")),
        TaxonomyTag("trust", ("信任", "可信", "可靠", "稳重", "专业", "可靠感", "大公司")),
        TaxonomyTag("ease", ("轻松", "无负担", "松弛", "便利")),
        TaxonomyTag("awe", ("敬畏", "自豪", "仰望", "宏大")),
        TaxonomyTag("loneliness", ("孤独", "疏离", "局外人")),
        TaxonomyTag("delight", ("有趣", "快乐", "俏皮", "wink", "游戏")),
        TaxonomyTag("creativity", ("创意", "创新", "想法", "idea")),
        TaxonomyTag("human_care", ("关注个体", "关注于人", "以人为本", "人文", "科技向善")),
    ),
    "audience_projection": (
        TaxonomyTag("future_builder", ("未来", "共创", "潜在员工", "加入者", "发挥潜力")),
        TaxonomyTag("direct_listener", ("倾听者", "对话者", "屏幕后的观众", "屏幕后", "直视", "听")),
        TaxonomyTag("witness", ("见证者", "观察者", "旁观者")),
        TaxonomyTag("participant", ("一员", "参与", "投射为", "被邀请")),
        TaxonomyTag("user", ("用户", "食客", "玩家", "观众")),
    ),
    "narrative_function": (
        TaxonomyTag("establish_trust", ("建立", "可信", "信任", "证明", "可靠")),
        TaxonomyTag("origin_story", ("起源", "初心", "开始", "历史")),
        TaxonomyTag("scale_reveal", ("规模", "宏大", "十亿", "全球", "国际化", "世界五百强")),
        TaxonomyTag("culture_showcase", ("文化", "团队", "员工", "氛围")),
        TaxonomyTag("invitation", ("邀请", "召唤", "机会", "加入")),
    ),
    "interaction_mode": (
        TaxonomyTag(
            "direct_address",
            ("直视", "眼神接触", "打破第四面墙", "对话", "面对面", "直接说话", "直视镜头"),
        ),
        TaxonomyTag("observational", ("观察者", "旁观", "见证")),
        TaxonomyTag("product_demonstration", ("扫码", "功能", "支付", "演示", "使用")),
        TaxonomyTag("symbolic_reveal", ("隐喻", "揭示", "转场", "显影")),
        TaxonomyTag("team_collaboration", ("团队", "协作", "共同", "多人")),
    ),
    "visual_motifs": (
        TaxonomyTag("screen", ("屏幕", "显示器", "界面", "手机", "screen")),
        TaxonomyTag("silhouette", ("轮廓", "逆光", "剪影")),
        TaxonomyTag("upward_motion", ("仰望", "向上", "起跑", "加速")),
        TaxonomyTag("urban_architecture", ("建筑", "城市", "cbd", "大楼", "天际线")),
        TaxonomyTag("crowd_contrast", ("人群", "拥挤", "背景", "狂欢")),
        TaxonomyTag("object_metaphor", ("按钮", "存钱罐", "道具", "蛋糕")),
        TaxonomyTag("hands", ("手", "指尖", "触碰", "手部")),
    ),
    "symbolic_logic": (
        TaxonomyTag("becoming", ("成为", "成长", "生长", "潜力")),
        TaxonomyTag("connection", ("连接", "人与人", "同频", "共振")),
        TaxonomyTag(
            "human_centered_technology",
            ("科技服务于人", "科技向善", "关注于人", "以人为本", "人本科技", "更好生活"),
        ),
        TaxonomyTag("scale_to_person", ("宏大", "个体", "十亿", "大众", "全球")),
        TaxonomyTag("contrast", ("对比", "反差", "不协调")),
        TaxonomyTag("origin_to_future", ("起源", "未来", "初心")),
    ),
    "rhythm_pattern": (
        TaxonomyTag("explosive_build", ("加速", "突破", "释放", "冲刺")),
        TaxonomyTag("slow_reveal", ("拉远", "揭示", "逐步", "推进")),
        TaxonomyTag("calm_direct", ("克制", "平静", "直视", "对话", "稳重", "可靠")),
        TaxonomyTag("contrast_cut", ("反差", "转换", "对比")),
        TaxonomyTag("montage", ("剪辑", "群像", "切换")),
    ),
}


class TagNormalizer:
    def __init__(self, taxonomy: dict[str, tuple[TaxonomyTag, ...]] | None = None) -> None:
        self.taxonomy = taxonomy or TAXONOMY

    def normalize_text(
        self,
        text: str,
        *,
        evidence: list[TagEvidence],
    ) -> TagProfile:
        tags = self.tags_from_text(text)
        return TagProfile(
            **tags,
            evidence=evidence,
            confidence=confidence_from_tags(tags),
        )

    def tags_from_text(self, text: str) -> dict[str, list[str]]:
        lowered = text.lower()
        tags: dict[str, list[str]] = {}
        for dimension in TAG_DIMENSIONS:
            dimension_tags: list[str] = []
            for entry in self.taxonomy[dimension]:
                if any(alias.lower() in lowered for alias in entry.aliases):
                    dimension_tags.append(entry.tag)
            tags[dimension] = _dedupe(dimension_tags)
        if not any(tags.values()):
            tags["symbolic_logic"] = [GENERAL_EXPRESSION_TAG]
        return tags


def confidence_from_tags(tags: dict[str, list[str]]) -> float:
    tag_count = sum(len(values) for values in tags.values() if GENERAL_EXPRESSION_TAG not in values)
    return min(0.95, max(0.35, 0.35 + tag_count * 0.05))


def canonical_tags(profile: TagProfile, dimension: str) -> set[str]:
    return {tag for tag in getattr(profile, dimension) if tag != GENERAL_EXPRESSION_TAG}


FingerprintNormalizer = TagNormalizer


def normalize_candidate_tag(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower()).strip("_")


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_candidate_tag(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result
