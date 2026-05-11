from __future__ import annotations

import json

from typer.testing import CliRunner

from sceneweaver.cli import app
from sceneweaver.pipeline.mock_pipeline import build_mock_artifacts
from sceneweaver.storage.json_store import write_json


def test_cli_mock_run(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "cli_mock"

    result = runner.invoke(app, ["mock-run", "--output", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "analysis" / "experience_cards.jsonl").exists()
    assert not (output_dir / "fingerprints").exists()


def test_cli_extract_experience(tmp_path):
    runner = CliRunner()
    _, _, scenes, *_ = build_mock_artifacts()
    output_dir = tmp_path / "video"
    write_json(output_dir / "analysis" / "scenes.json", scenes)

    result = runner.invoke(app, ["extract-experience", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "analysis" / "experience_cards.jsonl").exists()


def test_cli_retrieve_cards_outputs_matches(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "video"
    runner.invoke(app, ["mock-run", "--output", str(output_dir)])

    result = runner.invoke(
        app,
        [
            "retrieve-cards",
            str(output_dir),
            "招聘宣传片，年轻人作为团队一员共同创造未来",
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "query_tags" in data
    assert data["results"]
    assert data["results"][0]["matched_dimensions"]
