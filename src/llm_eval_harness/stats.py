"""Small statistics helpers (no numpy needed)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
