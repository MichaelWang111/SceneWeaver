"""Embedding, matrix, lexical, and prepared index implementations."""
"""Index/cache contracts for Retrieval Lab.

Heavy prepared-index execution still lives outside this layer; the native lab
starts by making index identity and fingerprinting explicit.
"""

from retreieval_lab.indexes.commands import index_inspect_command, index_manifest_command
from retreieval_lab.indexes.schema import (
    INDEX_SCHEMA_NAME,
    index_manifest_contract,
    normalize_index_manifest,
    validate_index_manifest,
)
from retreieval_lab.indexes.service import (
    DEFAULT_INDEX_MANIFEST_PATH,
    INDEX_CHANNELS,
    build_index_manifest,
    index_items_from_cases,
    index_manifest_summary,
    lexical_tokens,
    target_item_id,
)

__all__ = [
    "DEFAULT_INDEX_MANIFEST_PATH",
    "INDEX_CHANNELS",
    "INDEX_SCHEMA_NAME",
    "build_index_manifest",
    "index_inspect_command",
    "index_items_from_cases",
    "index_manifest_command",
    "index_manifest_contract",
    "index_manifest_summary",
    "lexical_tokens",
    "normalize_index_manifest",
    "target_item_id",
    "validate_index_manifest",
]
