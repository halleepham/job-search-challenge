"""AC-derived tests for REQ-9 skill components (AC-9.3, AC-9.4)."""

import pytest

from src.scoring import preferred_skill_overlap, required_skill_overlap


# ---------------------------------------------------------------- AC-9.3

@pytest.mark.parametrize("n_required,expected_cap", [(1, 1 / 3), (2, 2 / 3), (3, 1.0), (10, 1.0)])
def test_ac_9_3_damping_caps_thin_postings(n_required, expected_cap):
    """A full match scores at most the confidence cap for that posting size."""
    required = {f"skill{i}" for i in range(n_required)}
    assert required_skill_overlap(required, required) == pytest.approx(expected_cap)


def test_ac_9_3_thin_posting_no_longer_outranks_rich_one():
    """
    The inversion this rule exists to fix: a 1-skill posting fully matched used
    to beat a 10-skill posting matched 8/10.
    """
    thin = required_skill_overlap({"python"}, {"python"})
    rich = required_skill_overlap({f"s{i}" for i in range(8)}, {f"s{i}" for i in range(10)})
    assert thin < rich, "thin posting still outranks stronger evidence"


def test_ac_9_3_damping_does_not_affect_normal_postings():
    """Median posting has 3 required skills — at or above the threshold."""
    assert required_skill_overlap({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)


def test_ac_9_3_partial_match_scales():
    assert required_skill_overlap({"a"}, {"a", "b", "c", "d"}) == pytest.approx(0.25)


def test_ac_9_3_no_match_is_zero():
    assert required_skill_overlap({"x"}, {"a", "b", "c"}) == 0.0


def test_ac_9_3_empty_required_is_zero_not_one():
    """AC-2.6 removes these at ingestion; the guard must not award free points."""
    assert required_skill_overlap({"python"}, set()) == 0.0


def test_ac_9_3_bounded():
    assert 0.0 <= required_skill_overlap({"a", "b", "z"}, {"a", "b"}) <= 1.0


# ---------------------------------------------------------------- AC-9.4

def test_ac_9_4_empty_preferred_returns_none_not_one():
    """
    The Stage 2 AI defect: `if preferred_skills else 1.0`. Measured on this
    corpus, that would have awarded a perfect score to 93% of postings.
    """
    result = preferred_skill_overlap({"python"}, set())
    assert result is None
    assert result != 1.0 and result != 0.0


def test_ac_9_4_present_preferred_scores_normally():
    assert preferred_skill_overlap({"spark"}, {"spark", "airflow"}) == pytest.approx(0.5)


def test_ac_9_4_no_damping_on_preferred():
    """Damping is an AC-9.3 rule; preferred is 8% and fires on only 7% of jobs."""
    assert preferred_skill_overlap({"spark"}, {"spark"}) == 1.0
