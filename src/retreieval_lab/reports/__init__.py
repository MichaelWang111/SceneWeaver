"""Report generation for JSON, Markdown, SVG, and cycle summaries."""

from retreieval_lab.reports.eval_report import (
    DEFAULT_EVAL_REPORT_INPUT,
    DEFAULT_EVAL_REPORT_OUTPUT,
    eval_report_markdown,
    generate_eval_report_command,
)
from retreieval_lab.reports.markdown import capability_report_markdown, markdown_report

__all__ = [
    "DEFAULT_EVAL_REPORT_INPUT",
    "DEFAULT_EVAL_REPORT_OUTPUT",
    "capability_report_markdown",
    "eval_report_markdown",
    "generate_eval_report_command",
    "markdown_report",
]
