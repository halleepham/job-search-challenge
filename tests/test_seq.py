"""
Regression tests for the numpy-truthiness bug class.

It shipped twice - `_skill_floor_mask` (AC-6.8) and `build_bm25_text` (AC-5.2) -
because fixtures used Python lists while Parquet returns numpy arrays. These
tests pin every shape the real columns take.
"""

import numpy as np
import pytest

from src.seq import as_list, as_set


@pytest.mark.parametrize("value,expected", [
    (["a", "b"], ["a", "b"]),
    (np.array(["a", "b"]), ["a", "b"]),
    (np.array([]), []),
    ([], []),
    (None, []),
    (float("nan"), []),
    (("a",), ["a"]),
])
def test_as_list_handles_every_real_shape(value, expected):
    assert as_list(value) == expected


def test_as_set_intersects_with_a_plain_set():
    """The operation that raised in production."""
    assert as_set(np.array(["python", "rust"])) & {"python", "sql"} == {"python"}


def test_multi_element_array_does_not_raise():
    """`value or []` raises here; as_list must not."""
    arr = np.array(["python", "sql", "airflow"])
    with pytest.raises(ValueError):
        _ = arr or []
    assert as_list(arr) == ["python", "sql", "airflow"]
