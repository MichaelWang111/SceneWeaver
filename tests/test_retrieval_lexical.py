from __future__ import annotations

from sceneweaver.retrieval.lexical import bm25_scores, ranked_indices_from_scores, reciprocal_rank_fusion, tokenize


def test_tokenize_keeps_ascii_and_chinese_ngrams():
    tokens = tokenize("不要广告感，要 real location 真实现场")

    assert "real" in tokens
    assert "location" in tokens
    assert "广告" in tokens
    assert "真实" in tokens


def test_bm25_prefers_document_with_matching_style_terms():
    query = tokenize("纪录片 真实现场 有人味")
    docs = [
        tokenize("大厂办公 技术炫耀 口号"),
        tokenize("纪录片观察 真实现场 人的温度"),
    ]

    scores = bm25_scores(query, docs)

    assert scores[1] > scores[0]


def test_rrf_fuses_rankings_without_score_scale_assumptions():
    scores = reciprocal_rank_fusion([[0, 1], [1, 0]], item_count=2, k=60)

    assert scores[0] == scores[1]
    assert ranked_indices_from_scores([0.1, 4.0, 2.0]) == [1, 2, 0]
