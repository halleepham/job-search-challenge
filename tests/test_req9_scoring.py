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


# ============================================================ AC-9.5 - AC-9.14

import pandas as pd  # noqa: E402

from src.scoring import (  # noqa: E402
    WEIGHTS,
    education_alignment,
    experience_alignment,
    location_proximity,
    score_job,
    tier_for,
    title_match,
)
from src.profiles import UserProfile  # noqa: E402


# ---------------------------------------------------------------- AC-9.5

@pytest.mark.parametrize("user,job,expected", [
    (4.5, 5.0, 0.75),           # gap 0.5  -> close
    (3.0, 5.0, 0.50),           # gap 2.0  -> under
    (0.0, 8.0, 0.15),           # gap 8.0  -> max(0.15, 0.5-0.8)
    (5.0, 5.0, 1.0),            # exact
    (7.0, 5.0, 1.0),            # 2 yrs surplus, inside the plateau
    (8.0, 5.0, 1.0),            # 3 yrs surplus, edge of plateau
])
def test_ac_9_5_experience_bands(user, job, expected):
    assert experience_alignment(user, job) == pytest.approx(expected, abs=0.01)


def test_ac_9_5_overqualification_is_penalised_gently():
    """
    D7: under-qualification is a screening barrier, over-qualification only a
    preference mismatch — so the penalty is asymmetric and floored.
    """
    ten_over = experience_alignment(15.0, 5.0)
    one_under = experience_alignment(4.0, 5.0)
    assert 0.60 <= ten_over < 1.0
    assert ten_over < one_under, "10 years over should score below 1 year under"


def test_ac_9_5_overqualification_floor_holds():
    assert experience_alignment(40.0, 0.0) == pytest.approx(0.60)


def test_ac_9_5_unknown_requirement_is_neutral():
    """AC-9.13: missing input gets an explicit neutral, never a silent 1.0."""
    assert experience_alignment(3.0, None) == 0.5


# ---------------------------------------------------------------- AC-9.6

@pytest.mark.parametrize("completed,in_progress,required,expected", [
    (3, None, 3, 1.0),          # bachelor's, bachelor's required
    (3, None, 4, 0.6),          # bachelor's, master's required — one step
    (3, None, 5, 0.2),          # bachelor's, PhD required
    (4, None, 3, 1.0),          # exceeds
    (3, 4, 3, 1.0),             # mid-master's (3.5) vs bachelor's required
    (3, 4, 4, 0.6),             # mid-master's (3.5) vs master's required
    (3, None, None, 0.5),       # job states no requirement — neutral
])
def test_ac_9_6_education_ladder(completed, in_progress, required, expected):
    assert education_alignment(completed, in_progress, required) == pytest.approx(expected)


def test_ac_9_6_in_progress_is_the_decisive_case():
    """
    D6: the reason education is graded rather than gated. A student mid-master's
    must not be eliminated from master's-required roles.
    """
    assert education_alignment(3, 4, 4) > 0.0


# ---------------------------------------------------------------- AC-9.7

@pytest.mark.parametrize("preferred,job_title,expected", [
    (["data engineer"], "Senior Data Engineer", 1.0),
    (["data engineer"], "Data Engineer II", 1.0),
    (["data scientist"], "Data Engineer", 0.5),
    (["data engineer"], "Registered Nurse", 0.0),
    ([], "Data Engineer", 0.5),
])
def test_ac_9_7_title_match(preferred, job_title, expected):
    assert title_match(preferred, job_title) == pytest.approx(expected)


def test_ac_9_7_best_preferred_title_wins():
    assert title_match(["nurse", "data engineer"], "Data Engineer") == 1.0


# ---------------------------------------------------------------- AC-9.8

def test_ac_9_8_remote_scores_full():
    assert location_proximity(is_remote=True, accepts_remote=True, miles=None,
                              string_match=False) == 1.0


@pytest.mark.parametrize("miles,lo,hi", [(0, 0.99, 1.01), (10, 0.80, 0.90),
                                         (35, 0.50, 0.62), (100, 0.15, 0.23)])
def test_ac_9_8_distance_decay(miles, lo, hi):
    v = location_proximity(False, True, miles, False)
    assert lo <= v <= hi, f"{miles} miles -> {v:.2f}"


def test_ac_9_8_unresolved_branches():
    assert location_proximity(False, True, None, True) == 0.8
    assert location_proximity(False, True, None, False) == 0.3


# ---------------------------------------------------------------- AC-9.1/9.2

def test_ac_9_1_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("score,tier", [(95, "Strong"), (80, "Strong"), (79, "Good"),
                                        (65, "Good"), (64, "Moderate"), (45, "Moderate"),
                                        (44, "Low"), (0, "Low")])
def test_ac_9_2_tiers(score, tier):
    assert tier_for(score) == tier


# ---------------------------------------------------------------- AC-9.14

@pytest.fixture
def scored():
    profile = UserProfile(
        name="t", career_goals="data pipelines", resume_text="built ETL in Airflow",
        skills={"python", "sql", "airflow"}, years_experience=3.0,
        highest_completed_education=3, education_in_progress=4,
        preferred_titles=["data engineer"], preferred_location="Kansas City, MO",
    )
    job = pd.Series({
        "job_id": 1, "title": "Senior Data Engineer", "title_normalized": "data engineer",
        "required_skills": ["python", "sql", "airflow"], "preferred_skills": ["dbt"],
        "min_years_exp": 5.0, "education_required": 3, "is_remote": False,
        "location_raw": "Kansas City, MO", "work_setting": "On-site",
    })
    return score_job(job, profile, goals_similarity=0.6, evidence_similarity=0.7)


def test_ac_9_14_breakdown_sums_to_score(scored):
    """AC-9.14: the displayed contributions must add up to the displayed score."""
    total = sum(c["weighted"] for c in scored["components"])
    assert round(total * 100) == scored["score"]


def test_ac_9_14_every_component_present(scored):
    names = {c["name"] for c in scored["components"]}
    assert names == set(WEIGHTS)


def test_ac_9_1_score_is_a_bounded_integer(scored):
    assert isinstance(scored["score"], int) and 0 <= scored["score"] <= 100


def test_ac_9_4_renormalises_when_preferred_absent():
    """
    AC-9.4: with no preferred skills the component drops and the remaining seven
    weights renormalize to 1.0 — measured, that is 93% of postings.
    """
    profile = UserProfile(name="t", career_goals="g", resume_text="r",
                          skills={"python"}, preferred_titles=["data engineer"])
    job = pd.Series({
        "job_id": 1, "title": "Data Engineer", "title_normalized": "data engineer",
        "required_skills": ["python"], "preferred_skills": [], "min_years_exp": None,
        "education_required": None, "is_remote": True, "location_raw": None,
        "work_setting": "Remote",
    })
    out = score_job(job, profile, 0.5, 0.5)
    weights = {c["name"]: c["weight"] for c in out["components"]}
    assert weights["preferred_skills"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert round(sum(c["weighted"] for c in out["components"]) * 100) == out["score"]


def test_ac_9_13_missing_everything_still_scores(scored):
    """No component may crash or silently default to 1.0 on missing input."""
    profile = UserProfile(name="t", career_goals="", resume_text="", skills=set())
    job = pd.Series({"job_id": 1, "title": "X", "title_normalized": "x",
                     "required_skills": [], "preferred_skills": [], "min_years_exp": None,
                     "education_required": None, "is_remote": False, "location_raw": None,
                     "work_setting": "Unknown"})
    out = score_job(job, profile, 0.0, 0.0)
    assert 0 <= out["score"] <= 100
