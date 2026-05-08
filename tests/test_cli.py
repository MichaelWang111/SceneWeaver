from __future__ import annotations

from typer.testing import CliRunner

from sceneweaver.cli import app


def test_cli_mock_run(tmp_path):
    runner = CliRunner()
    output_dir = tmp_path / "cli_mock"

    result = runner.invoke(app, ["mock-run", "--output", str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "analysis" / "experience_cards.jsonl").exists()

