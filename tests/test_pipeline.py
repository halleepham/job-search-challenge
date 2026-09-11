"""End-to-end pipeline tests — the composition of REQ-6, 7, 9 and 4."""

import pandas as pd
import pytest

from src.indexing import JobIndex
from src.personal_kb import PersonalKB
from src.pipeline import search
from src.profiles import UserProfile

JOBS = pd.DataFrame([
    {"job_id": 1, "title": "Data Engineer", "title_normalized": "data engineer",
     "description": "Build ETL pipelines in Airflow and dbt on AWS.",
     "required_skills": ["python", "airflow", "sql"], "preferred_skills": ["dbt"],
     "location_raw": "Kansas City, MO", "is_remote": False, "work_setting": "On-site",
     "employment_type": "Full-time", "salary_min": 100000.0, "salary_max": 130000.0,
     "salary_listed": True, "min_years_exp": 3.0, "education_required": 3},
    {"job_id": 2, "title": "Senior Data Engineer", "title_normalized": "data engineer",
     "description": "Lead pipeline work with Spark and Snowflake.",
     "required_skills": ["python", "spark"], "preferred_skills": [],
     "location_raw": "Remote", "is_remote": True, "work_setting": "Remote",
     "employment_type": "Full-time", "salary_min": 150000.0, "salary_max": 180000.0,
     "salary_listed": True, "min_years_exp": 7.0, "education_required": None},
    {"job_id": 3, "title": "Registered Nurse", "title_normalized": "registered nurse",
     "description": "Patient care on the cardiac ward.",
     "required_skills": ["excel"], "preferred_skills": [],
     "location_raw": "Kansas City, MO", "is_remote": False, "work_setting": "On-site",
     "employment_type": "Full-time", "salary_min": 60000.0, "salary_max": 70000.0,
     "salary_listed": True, "min_years_exp": None, "education_required": 3},
])


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    return JobIndex.build(JOBS, index_dir=tmp_path_factory.mktemp("pipe"))


@pytest.fixture(scope="module")
def profile():
    return UserProfile(
        name="t", career_goals="I want to build data pipelines feeding dashboards.",
        resume_text="Education\nM.S. Data Science in progress.\n\n"
                    "Experience\nBuilt ETL pipelines in Airflow moving 40M rows daily.",
        skills={"python", "sql", "airflow"}, years_experience=2.0,
        highest_completed_education=3, education_in_progress=4,
        preferred_titles=["data engineer"], preferred_location="Kansas City, MO",
        min_salary=90000)


@pytest.fixture(scope="module")
def kb(profile):
    return PersonalKB.build(profile.resume_text, profile.career_goals)


@pytest.fixture(scope="module")
def result(index, profile, kb):
    return search(JOBS, index, profile, kb)


def test_returns_ranked_results(result):
    assert result.results
    scores = [r["score"] for r in result.results]
    assert scores == sorted(scores, reverse=True)


def test_filters_eliminate_before_scoring(result):
    """The nurse posting shares no skills and must never be scored at all."""
    assert 3 not in {r["job_id"] for r in result.results}


def test_scores_are_bounded_integers(result):
    for r in result.results:
        assert isinstance(r["score"], int) and 0 <= r["score"] <= 100


def test_ac_9_14_breakdown_sums_to_score_end_to_end(result):
    for r in result.results:
        assert round(sum(c["weighted"] for c in r["components"]) * 100) == r["score"]


def test_ac_11_2_funnel_reported(result):
    for key in ["start", "location", "salary", "skill_floor", "retrieved", "returned"]:
        assert key in result.funnel


def test_ac_13_3_per_stage_timings(result):
    for stage in ["filter", "retrieve", "score", "rank", "evidence", "total"]:
        assert stage in result.timings_ms and result.timings_ms[stage] >= 0


def test_ac_11_5_evidence_is_verbatim(result, profile):
    """
    AC-11.5 / AC-4.2: every quote shown must slice back out of the source
    document — the property that survived removing the LLM.
    """
    for r in result.results:
        for ev in r["evidence"]:
            source = profile.career_goals if ev["section"] == "career_goals" else profile.resume_text
            start, end = ev["char_span"]
            assert source[start:end] == ev["text"]


def test_matched_and_missing_skills_partition(result):
    for r in result.results:
        assert not set(r["matched_skills"]) & set(r["missing_skills"])


def test_empty_result_is_graceful(index, kb):
    impossible = UserProfile(name="t", career_goals="g", resume_text="r",
                             skills={"python"}, min_salary=10_000_000)
    out = search(JOBS, index, impossible, kb)
    assert out.results == []
    assert out.funnel["limiting_filter"] == "salary"


def test_retrieval_mode_is_selectable(index, profile, kb):
    """AC-7.3: REQ-13 compares modes through this same entry point."""
    for mode in ("bm25", "dense", "hybrid"):
        assert search(JOBS, index, profile, kb, mode=mode).results


def test_d13_v2_retrieval_always_runs(index, profile, kb):
    """
    D13 v2: retrieval is never skipped — `k` is sized by measured recall instead.
    v1 skipped the stage below a threshold, which solved a k-sizing problem by
    deleting the stage and left the report's comparison evaluating something the
    application bypassed.
    """
    from src.pipeline import search

    out = search(JOBS, index, profile, kb)
    assert out.funnel["retrieval"].startswith("hybrid top-")


def test_d13_v2_k_has_a_floor(index, profile, kb):
    """Hybrid reaches 100% recall of the true top-20 at k=400, so 400 is the floor."""
    from src.pipeline import MIN_RETRIEVE_K, search

    out = search(JOBS, index, profile, kb)
    assert f"top-{MIN_RETRIEVE_K}" in out.funnel["retrieval"]


def test_d13_explicit_mode_still_uses_retrieval(index, profile, kb):
    """AC-7.3: REQ-13 must still be able to exercise each retriever."""
    from src.pipeline import search

    out = search(JOBS, index, profile, kb, mode="bm25")
    assert out.funnel["retrieval"].startswith("bm25")


def test_d13_exact_path_cannot_lose_a_scored_job(index, profile, kb):
    """Every job that passes the filters is scored when the set is small."""
    from src.filters import apply_filters
    from src.pipeline import search

    kept, _ = apply_filters(JOBS, profile)
    out = search(JOBS, index, profile, kb, top_n=99)
    assert len(out.results) == len(kept)
