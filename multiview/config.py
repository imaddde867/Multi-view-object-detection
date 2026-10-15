"""Single home for how configs combine.

Every pipeline (label / train / run) layers: module defaults < YAML file <
CLI overrides. That layering used to be implemented four times with two
different meanings for ``None``; this module owns the one semantic: a
``None`` override means "not given" and never replaces a value from below.
"""

from __future__ import annotations

from typing import Any


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Layer ``overrides`` on top of ``base`` without mutating either.

    - ``None`` override values are skipped (the value was not provided).
    - Two dicts merge recursively; any other override type replaces wholesale.
    """
    out: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out
