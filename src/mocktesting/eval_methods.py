from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from sceneweaver.llm.client import VisionLLMClient

DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "eval_inputs" / "review_generated_inputs.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "eval_outputs"

METHOD_OUTPUT_FILES = {
    "summary-embedding": "summary_scores.json",
    "tags-embedding": "tag_scores.json",
    "llm-judge-lite": "llm_lite_scores.json",
    "llm-judge-batch": "llm_batch_scores.json",
    "llm-judge": "llm_scores.json",
    "all": "all_scores.json",
}

METHOD_DESCRIPTIONS = {
    "summary-embedding": "Compare user_input with target_summary by DashScope embedding cosine similarity.",
    "tags-embedding": "Compare user_input with flattened target_tags_text by DashScope embedding cosine similarity.",
    "llm-judge-lite": "Fast LLM judge. Return only fits and score, without commentary.",
    "llm-judge-batch": "Batch LLM judge. Return fits and score for multiple cases per request.",
    "llm-judge": "Review LLM judge. Return fits, score, and a short reason.",
    "all": "Run summary embedding, tags embedding, and review LLM judge.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock retrieval evaluation scoring methods.")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    parser.add_argument(
        "--method",
        choices=["summary-embedding", "tags-embedding", "llm-judge-lite", "llm-judge-batch", "llm-judge", "all"],
        default="summary-embedding",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to method-specific files under src/mocktesting/eval_outputs.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for each LLM judge request.",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=0,
        help="Retries for each LLM judge request.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Cases per request for llm-judge-batch.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent LLM requests for llm-judge-lite or llm-judge-batch.",
    )
    args = parser.parse_args()

    dataset = json.loads(args.inputs.read_text(encoding="utf-8"))
    cases = dataset["cases"][: args.limit if args.limit > 0 else None]
    started_at = time.perf_counter()
    report = score_cases(
        cases,
        method=args.method,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_retries=args.llm_retries,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    report["elapsed_seconds"] = elapsed_seconds
    report["seconds_per_case"] = round(elapsed_seconds / max(1, len(report["results"])), 3)
    output_path = resolve_output_path(args.method, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written: {output_path}")
    print(f"method: {args.method}")
    print(f"scored_cases: {len(report['results'])}")
    print(f"elapsed_seconds: {report['elapsed_seconds']}")
    print(f"seconds_per_case: {report['seconds_per_case']}")
    print(f"aggregates: {json.dumps(report['aggregates'], ensure_ascii=False)}")


def score_cases(
    cases: list[dict[str, Any]],
    *,
    method: str,
    llm_timeout_seconds: float = 60.0,
    llm_retries: int = 0,
    batch_size: int = 10,
    concurrency: int = 1,
) -> dict[str, Any]:
    if method == "llm-judge-batch":
        results = score_cases_llm_batch(
            cases,
            timeout_seconds=llm_timeout_seconds,
            retries=llm_retries,
            batch_size=batch_size,
            concurrency=concurrency,
        )
        return {
            "method": method,
            "description": METHOD_DESCRIPTIONS[method],
            "scored_case_count": len(results),
            "batch_size": batch_size,
            "concurrency": concurrency,
            "aggregates": build_aggregates(results),
            "results": results,
        }
    if method == "llm-judge-lite" and concurrency > 1:
        results = score_cases_llm_lite_parallel(
            cases,
            timeout_seconds=llm_timeout_seconds,
            retries=llm_retries,
            concurrency=concurrency,
        )
        return {
            "method": method,
            "description": METHOD_DESCRIPTIONS[method],
            "scored_case_count": len(results),
            "concurrency": concurrency,
            "aggregates": build_aggregates(results),
            "results": results,
        }

    results = []
    for case in cases:
        row = {
            "case_id": case["case_id"],
            "case_type": case["case_type"],
            "expected_relation": case["expected_relation"],
            "user_input": case["user_input"],
            "target": case["target"],
            "target_summary": case["target_summary"],
            "target_tags_text": case["target_tags_text"],
            "expected_prefer": case.get("expected_prefer"),
            "comparisons": {},
        }
        if method in {"summary-embedding", "all"}:
            row["comparisons"]["summary_embedding"] = compare_by_embedding(
                input_text=case["user_input"],
                reference_text=case["target_summary"],
                reference_field="target_summary",
            )
        if method in {"tags-embedding", "all"}:
            row["comparisons"]["tags_embedding"] = compare_by_embedding(
                input_text=case["user_input"],
                reference_text=case["target_tags_text"],
                reference_field="target_tags_text",
            )
        if method in {"llm-judge-lite"}:
            row["comparisons"]["llm_judge_lite"] = safe_llm_judge(
                case,
                mode="lite",
                timeout_seconds=llm_timeout_seconds,
                retries=llm_retries,
            )
        if method in {"llm-judge", "all"}:
            row["comparisons"]["llm_judge"] = safe_llm_judge(
                case,
                mode="review",
                timeout_seconds=llm_timeout_seconds,
                retries=llm_retries,
            )
        results.append(row)
    return {
        "method": method,
        "description": METHOD_DESCRIPTIONS[method],
        "scored_case_count": len(results),
        "aggregates": build_aggregates(results),
        "results": results,
    }


def score_cases_llm_batch(
    cases: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    retries: int,
    batch_size: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    indexed_batches = [
        (start, cases[start : start + batch_size])
        for start in range(0, len(cases), batch_size)
    ]
    if concurrency == 1:
        batch_outputs = [
            (start, batch_cases, safe_llm_judge_batch(batch_cases, timeout_seconds=timeout_seconds, retries=retries))
            for start, batch_cases in indexed_batches
        ]
    else:
        batch_outputs = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(
                    safe_llm_judge_batch,
                    batch_cases,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                ): (start, batch_cases)
                for start, batch_cases in indexed_batches
            }
            for future in as_completed(future_map):
                start, batch_cases = future_map[future]
                batch_outputs.append((start, batch_cases, future.result()))
        batch_outputs.sort(key=lambda row: row[0])

    results: list[dict[str, Any]] = []
    for _start, batch_cases, batch_result in batch_outputs:
        by_case_id = {
            str(item.get("case_id", "")): item
            for item in batch_result.get("results", [])
            if isinstance(item, dict)
        }
        batch_error = batch_result.get("error")
        for case in batch_cases:
            comparison = comparison_from_batch_result(case, by_case_id.get(case["case_id"]), batch_error)
            results.append(
                {
                    "case_id": case["case_id"],
                    "case_type": case["case_type"],
                    "expected_relation": case["expected_relation"],
                    "user_input": case["user_input"],
                    "target": case["target"],
                    "target_summary": case["target_summary"],
                    "target_tags_text": case["target_tags_text"],
                    "expected_prefer": case.get("expected_prefer"),
                    "comparisons": {
                        "llm_judge_batch": comparison,
                    },
                }
            )
    return results


def score_cases_llm_lite_parallel(
    cases: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    retries: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    rows: list[dict[str, Any] | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(
                safe_llm_judge,
                case,
                mode="lite",
                timeout_seconds=timeout_seconds,
                retries=retries,
            ): (index, case)
            for index, case in enumerate(cases)
        }
        for future in as_completed(future_map):
            index, case = future_map[future]
            rows[index] = {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "user_input": case["user_input"],
                "target": case["target"],
                "target_summary": case["target_summary"],
                "target_tags_text": case["target_tags_text"],
                "expected_prefer": case.get("expected_prefer"),
                "comparisons": {
                    "llm_judge_lite": future.result(),
                },
            }
    return [row for row in rows if row is not None]


def safe_llm_judge_batch(
    cases: list[dict[str, Any]],
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    try:
        return llm_judge_batch(cases, timeout_seconds=timeout_seconds, retries=retries)
    except Exception as exc:
        return {
            "error": str(exc),
            "results": [],
        }


def comparison_from_batch_result(
    case: dict[str, Any],
    item: dict[str, Any] | None,
    batch_error: str | None,
) -> dict[str, Any]:
    if batch_error is not None:
        return {
            "method": "llm_judge",
            "mode": "batch",
            "fits": None,
            "score": None,
            "error": batch_error,
        }
    if item is None:
        return {
            "method": "llm_judge",
            "mode": "batch",
            "fits": None,
            "score": None,
            "error": f"LLM batch response missing case_id={case['case_id']}",
        }
    return {
        "method": "llm_judge",
        "mode": "batch",
        "fits": bool(item.get("fits", False)),
        "score": float(item.get("score", 0.0)),
    }


def resolve_output_path(method: str, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return DEFAULT_OUTPUT_DIR / METHOD_OUTPUT_FILES[method]


def compare_by_embedding(*, input_text: str, reference_text: str, reference_field: str) -> dict[str, Any]:
    return {
        "method": "dashscope_text_embedding_v4_cosine",
        "reference_field": reference_field,
        "input_text": input_text,
        "reference_text": reference_text,
        "cosine": embedding_similarity(input_text, reference_text),
        "threshold_hint": 0.4,
    }


def embedding_similarity(text_a: str, text_b: str) -> float:
    vectors = qwen_embed([text_a, text_b])
    return round(cosine(vectors[0], vectors[1]), 6)


def qwen_embed(texts: list[str]) -> list[list[float]]:
    try:
        import dashscope
        from http import HTTPStatus
    except ImportError as exc:
        raise RuntimeError("dashscope package is required for embedding evaluation") from exc

    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("SCENEWEAVER_API_KEY")
        or os.environ.get("VIDEO_ANALYZER_API_KEY")
    )
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY, SCENEWEAVER_API_KEY, or VIDEO_ANALYZER_API_KEY is required")

    dashscope.base_http_api_url = os.environ.get("DASHSCOPE_HTTP_API_URL", "https://dashscope.aliyuncs.com/api/v1")
    model = os.environ.get("DASHSCOPE_EMBEDDING_MODEL", dashscope.TextEmbedding.Models.text_embedding_v4)
    dimension = int(os.environ.get("DASHSCOPE_EMBEDDING_DIMENSION", "1024"))
    response = dashscope.TextEmbedding.call(
        api_key=api_key,
        model=model,
        input=texts,
        dimension=dimension,
        output_type="dense",
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(f"dashscope embedding request failed: {response}")
    return [item["embedding"] for item in response.output["embeddings"]]


def cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def safe_llm_judge(
    case: dict[str, Any],
    *,
    mode: str,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    try:
        if mode == "lite":
            return llm_judge_lite_case(case, timeout_seconds=timeout_seconds, retries=retries)
        return llm_judge_case(case, timeout_seconds=timeout_seconds, retries=retries)
    except Exception as exc:
        return {
            "method": "llm_judge",
            "mode": mode,
            "fits": None,
            "score": None,
            "error": str(exc),
        }


def llm_judge_case(
    case: dict[str, Any],
    *,
    timeout_seconds: float = 60.0,
    retries: int = 0,
) -> dict[str, Any]:
    system_prompt = "你是检索测试评审。只输出 JSON。"
    user_prompt = json.dumps(
        {
            "task": "判断 user_input 是否符合 target_summary 所描述的镜头用途。不要苛刻，按检索原型评估。",
            "output_schema": {
                "fits": "boolean",
                "score": "0..1 number",
                "reason": "short Chinese sentence",
            },
            "user_input": case["user_input"],
            "target_summary": case["target_summary"],
            "target_tags_text": case["target_tags_text"],
            "expected_relation": case["expected_relation"],
        },
        ensure_ascii=False,
    )
    result = VisionLLMClient().analyze_text_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=500,
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    return {
        "method": "llm_judge",
        "mode": "review",
        "fits": bool(result.get("fits", False)),
        "score": float(result.get("score", 0.0)),
        "reason": str(result.get("reason", "")),
    }


def llm_judge_lite_case(
    case: dict[str, Any],
    *,
    timeout_seconds: float = 60.0,
    retries: int = 0,
) -> dict[str, Any]:
    system_prompt = "你是检索匹配二分类器。只输出 JSON。"
    user_prompt = json.dumps(
        {
            "rule": "判断 A 是否匹配 B。注意不要/不是/避免等否定词。",
            "output": {"fits": True, "score": 0.0},
            "A": case["user_input"],
            "B": case["target_summary"],
        },
        ensure_ascii=False,
    )
    result = VisionLLMClient().analyze_text_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=80,
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    return {
        "method": "llm_judge",
        "mode": "lite",
        "fits": bool(result.get("fits", False)),
        "score": float(result.get("score", 0.0)),
    }


def llm_judge_batch(
    cases: list[dict[str, Any]],
    *,
    timeout_seconds: float = 60.0,
    retries: int = 0,
) -> dict[str, Any]:
    system_prompt = "你是检索匹配批量二分类器。只输出 JSON。"
    payload_cases = [
        {
            "case_id": case["case_id"],
            "A": case["user_input"],
            "B": case["target_summary"],
        }
        for case in cases
    ]
    user_prompt = json.dumps(
        {
            "rule": "逐条判断 A 是否匹配 B。注意不要/不是/避免等否定词。",
            "output": {
                "results": [
                    {
                        "case_id": "string",
                        "fits": True,
                        "score": 0.0,
                    }
                ]
            },
            "cases": payload_cases,
        },
        ensure_ascii=False,
    )
    result = VisionLLMClient().analyze_text_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max(200, len(cases) * 60),
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    raw_results = result.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("LLM batch response must contain a results list")
    return {"results": raw_results}


def build_aggregates(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {}
    by_case_type: dict[str, dict[str, list[float]]] = {}
    for row in results:
        case_type = row["case_type"]
        by_case_type.setdefault(case_type, {})
        for name, comparison in row["comparisons"].items():
            value = numeric_score_from_comparison(comparison)
            if value is None:
                continue
            metric_values.setdefault(name, []).append(value)
            by_case_type[case_type].setdefault(name, []).append(value)
    return {
        "overall": {name: summarize_numbers(values) for name, values in metric_values.items()},
        "by_case_type": {
            case_type: {name: summarize_numbers(values) for name, values in metrics.items()}
            for case_type, metrics in by_case_type.items()
        },
    }


def numeric_score_from_comparison(comparison: dict[str, Any]) -> float | None:
    if "cosine" in comparison:
        return float(comparison["cosine"])
    if comparison.get("score") is not None:
        return float(comparison["score"])
    return None


def summarize_numbers(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "avg": round(sum(values) / len(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


if __name__ == "__main__":
    main()
