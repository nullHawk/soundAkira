from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from soundakira.config import FilterRule


def first_failure(metrics: Mapping[str, float], rules: Sequence[FilterRule]) -> str | None:
    """Name of the first failing rule (used as a drop reason), or None if all pass."""
    for r in rules:
        v = metrics.get(r.field)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            if r.on_missing == "drop":
                return f"missing_{r.field}"
            continue
        if r.min is not None and v < r.min:
            return f"{r.field}_below_{r.min:g}"
        if r.max is not None and v > r.max:
            return f"{r.field}_above_{r.max:g}"
    return None
