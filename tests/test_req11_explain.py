"""AC-derived tests for REQ-11 v1.1 — the explainable-match panels."""

import pandas as pd
import pytest

from src.explain import (
    component_bars,
    evidence_table,
    gaps_and_unknowns,
    verdict,
)
from src.profiles import UserProfile


def make_result(**job_overrides):
    job = pd.Series({
        "job_id": 1, "title": "Data Engineer", "company": "Acme",
        "location_raw": "Kansas City, MO", "work_setting": "On-site",
        "work_setting_inferred": False, "employment_type": "Full-time",
        "salary_listed": True, "salary_min": 100000.0, "salary_max": 130000.0,
        "min_years_exp": 3.0, "min_years_exp_source": "description_regex",
        "education_required": 3, "required_skills": ["python", "sql", "airflow", "dbt"],
        "description": "Build pipelines.",
        **job_overrides})
    return {
        "job_id": 1, "score": 68, "tier": "Good", "job": job,
        "matched_skills": ["python", "sql"], "missing_skills": ["airflow", "dbt"],
        # Weights are AC-9.4-renormalized (preferred dropped, remaining /0.92) and
        # `weighted` = sub_score x weight, so the fixture obeys AC-11.3 by
        # construction rather than by hand-tuned numbers that happen to look right.
        "components": [
            {"name": "required_skills", "sub_score": 0.5, "weight": 0.3261, "weighted": 0.1630},
            {"name": "preferred_skills", "sub_score": None, "weight": 0.0, "weighted": 0.0},
            {"name": "experience", "sub_score": 1.0, "weight": 0.1630, "weighted": 0.1630},
            {"name": "education", "sub_score": 1.0, "weight": 0.0435, "weighted": 0.0435},
            {"name": "title", "sub_score": 1.0, "weight": 0.0543, "weighted": 0.0543},
            {"name": "location", "sub_score": 1.0, "weight": 0.0870, "weighted": 0.0870},
            {"name": "career_goals_similarity", "sub_score": 0.4, "weight": 0.1413, "weighted": 0.0565},
            {"name": "resume_evidence_similarity", "sub_score": 0.6, "weight": 0.1848, "weighted": 0.1109},
        ],
        "evidence": [{"text": "Built ETL pipelines in Airflow.", "section": "experience",
                      "char_span": (0, 30), "similarity": 0.71}],
    }


PROFILE = UserProfile(name="t", career_goals="pipelines", resume_text="r",
                      skills={"python", "sql"}, preferred_location="Kansas City, MO",
                      min_salary=90000)


# ---------------------------------------------------------------- AC-11.10

def test_ac_11_10_bars_show_points_out_of_max():
    bars = component_bars(make_result())
    skills = bars[bars.Component == "Skills"].iloc[0]
    assert skills["Earned"] == pytest.approx(16.3, abs=0.1)
    assert skills["Max"] == pytest.approx(32.6, abs=0.1)


def test_ac_11_10_dropped_component_shows_zero_max_not_a_failure():
    """AC-9.4: a dropped component must not read as a component scored zero."""
    bars = component_bars(make_result())
    pref = bars[bars.Component == "Preferred skills"].iloc[0]
    assert pref["Dropped"] and pref["Max"] == 0.0


def test_ac_11_3_bars_still_sum_to_the_score():
    result = make_result()
    assert round(component_bars(result)["Earned"].sum()) == result["score"]


# ---------------------------------------------------------------- AC-11.9

def test_ac_11_9_evidence_is_typed_with_a_source():
    table = evidence_table(make_result(), PROFILE)
    assert list(table.columns) == ["Type", "Evidence", "Source"]
    assert (table["Source"] != "").all()


def test_ac_11_9_resume_evidence_is_labelled_by_section():
    table = evidence_table(make_result(), PROFILE)
    assert "Experience" in table["Type"].values
    assert "Résumé" in table["Source"].values


def test_ac_11_9_includes_job_and_profile_rows():
    table = evidence_table(make_result(), PROFILE)
    assert "Job description" in table["Source"].values
    assert "Profile" in table["Source"].values


def test_ac_11_9_marks_an_inferred_work_setting():
    table = evidence_table(make_result(work_setting_inferred=True), PROFILE)
    location = table[table["Type"] == "Location"].iloc[0]["Evidence"]
    assert "inferred" in location


# ---------------------------------------------------------------- AC-11.11

def test_ac_11_11_columns():
    gaps = gaps_and_unknowns(make_result(), PROFILE)
    assert list(gaps.columns) == ["Issue", "Kind", "Details", "Impact"]


def test_ac_11_11_missing_skills_are_a_gap():
    gaps = gaps_and_unknowns(make_result(), PROFILE)
    row = gaps[gaps.Issue == "Missing skills"].iloc[0]
    assert row["Kind"] == "Gap" and "airflow" in row["Details"]


@pytest.mark.parametrize("override,issue", [
    ({"salary_listed": False, "salary_max": None}, "Salary not listed"),
    ({"work_setting_inferred": True}, "Work setting inferred"),
    ({"education_required": None}, "Education requirement not stated"),
    ({"min_years_exp_source": "none", "min_years_exp": None},
     "Experience requirement not stated"),
    ({"min_years_exp_source": "seniority_label"}, "Experience inferred"),
    ({"required_skills": ["python"]}, "Few requirements listed"),
])
def test_ac_11_11_each_unknown_is_surfaced(override, issue):
    """
    Every missing-data policy the pipeline applies must be visible to the user.
    A score built partly on neutral defaults that does not say so implies a
    confidence the data cannot support.
    """
    gaps = gaps_and_unknowns(make_result(**override), PROFILE)
    row = gaps[gaps.Issue == issue]
    assert len(row) == 1 and row.iloc[0]["Kind"] == "Unknown"


def test_ac_11_11_complete_posting_reports_no_unknowns():
    gaps = gaps_and_unknowns(make_result(), PROFILE)
    assert (gaps["Kind"] == "Unknown").sum() == 0


def test_ac_11_11_half_missing_is_medium_not_high():
    """Exactly half unmet is a real gap but not disqualifying."""
    gaps = gaps_and_unknowns(make_result(), PROFILE)
    assert gaps[gaps.Issue == "Missing skills"].iloc[0]["Impact"] == "Medium"


def test_ac_11_11_high_impact_when_most_skills_missing():
    result = make_result(required_skills=["a", "b", "c", "d"])
    result["missing_skills"] = ["a", "b", "c"]
    gaps = gaps_and_unknowns(result, PROFILE)
    assert gaps[gaps.Issue == "Missing skills"].iloc[0]["Impact"] == "High"


# ---------------------------------------------------------------- AC-11.13

def test_ac_11_13_verdict_names_what_to_verify():
    result = make_result(salary_listed=False, salary_max=None)
    headline, advice = verdict(result, gaps_and_unknowns(result, PROFILE))
    assert headline == "Good candidate"
    assert "salary" in advice.lower()


def test_ac_11_13_verdict_leads_with_a_blocking_gap():
    result = make_result(required_skills=["a", "b", "c", "d"], salary_listed=False,
                         salary_max=None)
    result["missing_skills"] = ["a", "b", "c"]
    _, advice = verdict(result, gaps_and_unknowns(result, PROFILE))
    assert "missing skills" in advice.lower()


def test_ac_11_13_verdict_says_so_when_nothing_is_unstated():
    result = make_result()
    _, advice = verdict(result, gaps_and_unknowns(result, PROFILE))
    assert "nothing material" in advice.lower()


@pytest.mark.parametrize("tier,expected", [
    ("Strong", "Strong candidate"), ("Good", "Good candidate"),
    ("Moderate", "Possible candidate"), ("Low", "Weak candidate"),
])
def test_ac_11_13_headline_per_tier(tier, expected):
    result = make_result()
    result["tier"] = tier
    assert verdict(result, gaps_and_unknowns(result, PROFILE))[0] == expected


def test_ac_11_9_section_heading_not_repeated():
    """
    Paragraph chunking keeps the heading inside the chunk, so without stripping
    it the row reads "Skills | Skills Python, SQL…". Display only — the stored
    char_span still covers the full verbatim chunk.
    """
    result = make_result()
    result["evidence"] = [{"text": "Skills\nPython, SQL, Airflow.", "section": "skills",
                           "char_span": (0, 27), "similarity": 0.6}]
    row = evidence_table(result, PROFILE).iloc[0]
    assert row["Type"] == "Skills"
    assert row["Evidence"] == "Python, SQL, Airflow."


def test_ac_11_9_body_without_a_heading_is_untouched():
    result = make_result()
    result["evidence"] = [{"text": "Built ETL pipelines in Airflow.", "section": "experience",
                           "char_span": (0, 31), "similarity": 0.7}]
    assert evidence_table(result, PROFILE).iloc[0]["Evidence"] == "Built ETL pipelines in Airflow."


def test_ac_11_3_displayed_points_sum_exactly_to_the_score():
    """
    Rounding each component then adding does not equal rounding the total —
    eight values at 0.1 precision drifted to 78.5 against a card reading 79.
    Largest-remainder allocation makes the displayed breakdown add up exactly.
    """
    result = make_result()
    assert component_bars(result)["Earned"].sum() == pytest.approx(result["score"], abs=0.051)


@pytest.mark.parametrize("score", [0, 37, 49, 50, 73, 99, 100])
def test_ac_11_3_holds_across_scores(score):
    result = make_result()
    result["score"] = score
    total = sum(c["weighted"] for c in result["components"]) or 1
    for c in result["components"]:                      # rescale to the target score
        c["weighted"] = c["weighted"] / total * (score / 100)
    assert component_bars(result)["Earned"].sum() == pytest.approx(score, abs=0.051)


@pytest.mark.parametrize("nudge", [0.0, 0.004, -0.004, 0.0049, -0.0049])
def test_ac_11_3_holds_when_the_exact_total_rounds_either_way(nudge):
    """
    Regression: the allocation only handled a surplus. When the exact total
    rounded *down* the residual was negative, nothing was subtracted, and the
    table read 81.1 against a card saying 81.
    """
    result = make_result()
    result["components"][0]["weighted"] += nudge
    result["score"] = round(sum(c["weighted"] for c in result["components"]) * 100)
    assert component_bars(result)["Earned"].sum() == pytest.approx(result["score"], abs=0.051)


def test_ac_11_3_dropped_components_cannot_swallow_a_shortfall():
    """
    Regression: a dropped component sits at 0 points and has the smallest
    remainder, so it was selected first to absorb a decrement it could not give.
    The residual went unconsumed and the table read 81.1 against a card of 81.
    """
    result = make_result()
    for c in result["components"][2:]:          # leave only skills + a dropped row
        c["sub_score"], c["weight"], c["weighted"] = None, 0.0, 0.0
    result["components"][0]["weighted"] = 0.8147
    result["score"] = round(sum(c["weighted"] for c in result["components"]) * 100)
    bars = component_bars(result)
    assert bars["Earned"].sum() == pytest.approx(result["score"], abs=0.051)
    assert (bars["Earned"] >= 0).all(), "no component may render negative points"
