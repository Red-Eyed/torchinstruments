"""Stable dashboard path construction shared by telemetry projections."""

from __future__ import annotations

from urllib.parse import quote


def tensor_path_prefix(module_name: str, call_index: int, tensor_path: str) -> str:
    """Build the dashboard path shared by scalar and histogram projections."""
    module_segment = "@root" if not module_name else path_segment(module_name)
    return f"modules/{module_segment}/call_{call_index}/{path_segment(tensor_path)}"


def path_segment(value: str) -> str:
    """Escape slash-like separators while preserving readable dotted names."""
    return quote(value, safe="._-")
