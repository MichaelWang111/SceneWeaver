from __future__ import annotations

from sceneweaver.retrieval.query_plan import build_query_plan


def test_query_plan_extracts_stage_constraints_and_positive_query():
    plan = build_query_plan("不要技术展示，我真正要的是铺垫，要有人味")

    assert plan.forbidden_stage == ["technology_showcase"]
    assert plan.desired_stage == ["setup"]
    assert "技术展示" not in plan.positive_query
    assert "铺垫" in plan.positive_query
    assert "keep_human_warmth" in plan.positive_purposes


def test_query_plan_keeps_non_stage_negative_constraints():
    plan = build_query_plan("年轻团队协作，不要大厂味，也不要广告感")

    assert plan.forbidden_stage == []
    assert "大厂味" in plan.negative_constraints
    assert "广告感" in plan.negative_constraints
    assert "大厂味" not in plan.positive_query


def test_query_plan_extracts_style_constraints():
    plan = build_query_plan("要有人味、像纪录片、真实现场，不要大厂味，不要广告感，也不要炫技")

    assert plan.positive_style == ["human_warmth", "documentary", "real_location"]
    assert plan.negative_style == ["big_company_office", "ad_like", "tech_showoff"]
    assert plan.style_constraints == {
        "positive": ["human_warmth", "documentary", "real_location"],
        "negative": ["big_company_office", "ad_like", "tech_showoff"],
    }
    assert "大厂味" not in plan.positive_query
