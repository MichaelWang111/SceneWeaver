from __future__ import annotations

import pytest

from sceneweaver.llm.client import PartialStreamError, _collect_stream_text


class _Delta:
    def __init__(self, content: str | None, reasoning_content: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, content: str | None, reasoning_content: str | None = None) -> None:
        self.delta = _Delta(content, reasoning_content)


class _Chunk:
    def __init__(self, choices) -> None:
        self.choices = choices


class _BadChoicesChunk:
    @property
    def choices(self):
        raise IndexError("list index out of range")


def test_collect_stream_text_ignores_empty_and_malformed_choice_chunks():
    streamed: list[str] = []
    response = [
        _Chunk([]),
        _BadChoicesChunk(),
        _Chunk([_Choice('{"ok":')]),
        _Chunk([_Choice("true}")]),
        _Chunk([_Choice(None)]),
    ]

    text = _collect_stream_text(response, streamed.append)

    assert text == '{"ok":true}'
    assert streamed == ['{"ok":', "true}"]


def test_collect_stream_text_uses_complete_json_when_stream_ends_with_error():
    streamed: list[str] = []

    def response():
        yield _Chunk([_Choice('{"ok":')])
        yield _Chunk([_Choice("true}")])
        raise TimeoutError("provider stream timed out")

    text = _collect_stream_text(response(), streamed.append)

    assert text == '{"ok":true}'
    assert streamed == ['{"ok":', "true}"]


def test_collect_stream_text_keeps_reasoning_separate_from_json_text():
    streamed: list[str] = []
    reasoning: list[str] = []
    response = [
        _Chunk([_Choice(None, "think ")]),
        _Chunk([_Choice(None, "first")]),
        _Chunk([_Choice('{"ok":')]),
        _Chunk([_Choice("true}")]),
    ]

    text = _collect_stream_text(response, streamed.append, reasoning_callback=reasoning.append)

    assert text == '{"ok":true}'
    assert streamed == ['{"ok":', "true}"]
    assert reasoning == ["think ", "first"]


def test_collect_stream_text_raises_partial_stream_error_for_incomplete_json():
    def response():
        yield _Chunk([_Choice('{"ok":')])
        raise TimeoutError("provider stream timed out")

    with pytest.raises(PartialStreamError) as exc_info:
        _collect_stream_text(response(), lambda _chunk: None)

    assert exc_info.value.partial_text == '{"ok":'
    assert isinstance(exc_info.value.original_error, TimeoutError)
