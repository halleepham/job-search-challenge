"""
Safe coercion for columns that may hold a list, a numpy array, or nothing.

Parquet round-trips list columns as numpy arrays, and `value or []` **raises**
`ValueError: The truth value of an array with more than one element is
ambiguous` on an array rather than returning the default. Fixtures built from
Python lists never hit it; only real data does.

This bug appeared twice - once in the skill-floor filter and once in the BM25
text builder - so the coercion lives in one place rather than being re-derived
per call site.
"""

from __future__ import annotations


def as_list(value) -> list:
    """[] for None/NaN, otherwise the value's items. Never raises on an array."""
    if value is None:
        return []
    try:
        if value != value:          # NaN
            return []
    except (TypeError, ValueError):
        pass                        # arrays: elementwise compare, not a scalar
    try:
        return list(value)
    except TypeError:
        return []


def as_set(value) -> set:
    return set(as_list(value))
