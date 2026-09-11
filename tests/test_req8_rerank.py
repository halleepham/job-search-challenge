"""
AC-derived tests for REQ-8 v2.0 — optional cross-encoder reranking.

The stage is **off by default**; these assert that it works when asked for, that
it never leaks into the score, and — most importantly — that it cannot turn
itself on.
"""

import pandas as pd
import pytest

from src.indexing import JobIndex
from src.profiles import UserProfile

JOBS = pd.DataFrame([
    {"job_id": 1, "title": "Data Engineer", "title_normalized": "data engineer",
     "description": "Build ETL pipelines in Airflow and dbt on AWS.",
     "required_skills": ["python", "airflow", "sql"], "preferred_skills": ["dbt"],
     "location_raw": "Kansas City, MO", "is_remote": False, "work_setting": "On-site",
     "work_setting_inferred": False, "employment_type": "Full-time",
     "salary_min": 100000.0, "salary_max": 130000.0, "salary_listed": True,
     "min_years_exp": 3.0, "min_years_exp_source": "description_regex",
     "education_required": 3, "posting_url": "https://example.com/1",
     "expiry_date": "2024-05-07"},
    {"job_id": 2, "title": "Security Analyst", "title_normalized": "security analyst",
     "description": "SOC monitoring, SIEM tuning and incident response.",
     "required_skills": ["python", "splunk"], "preferred_skills": [],
     "location_raw": "Kansas City, MO", "is_remote": False, "work_setting": "On-site",
     "work_setting_inferred": False, "employment_type": "Full-time",
     "salary_min": 100000.0, "salary_max": 130000.0, "salary_listed": True,
     "min_years_exp": 3.0, "min_years_exp_source": "description_regex",
     "education_required": 3, "posting_url": "https://example.com/2",
     "expiry_date": "2024-05-07"},
])

PROFILE = UserProfile(
    name="t", career_goals="I want to build data pipelines feeding dashboards.",
    resume_text="Built ETL pipelines in Airflow.", skills={"python", "airflow", "sql"},
    years_experience=2.0, preferred_titles=["data engineer"],
    preferred_location="Kansas City, MO")


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    return JobIndex.build(JOBS, index_dir=tmp_path_factory.mktemp("rr"))


@pytest.fixture(scope="module")
def kb():
    from src.personal_kb import PersonalKB

    return PersonalKB.build(PROFILE.resume_text, PROFILE.career_goals)


# ---------------------------------------------------------------- AC-8.1

def test_ac_8_1_model_is_pinned():
    from src.rerank import RERANK_MODEL

    assert RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_ac_8_1_reranking_reorders_and_truncates(index):
    from src.rerank import rerank_candidates

    hits = [(0, 0.0), (1, 0.0)]
    out = rerank_candidates(index, JOBS, PROFILE, hits, top_k=1)
    assert len(out) == 1
    assert out[0][0] in (0, 1)


def test_ac_8_1_empty_input_is_not_an_error(index):
    from src.rerank import rerank_candidates

    assert rerank_candidates(index, JOBS, PROFILE, [], top_k=5) == []


def test_ac_8_1_ranks_the_relevant_job_first(index):
    """A data profile against a data job and a security job."""
    from src.rerank import rerank_candidates

    out = rerank_candidates(index, JOBS, PROFILE, [(0, 0.0), (1, 0.0)], top_k=2)
    assert JOBS.iloc[out[0][0]]["job_id"] == 1


# ---------------------------------------------------------------- AC-8.2

def test_ac_8_2_scores_are_not_propagated(index):
    """
    AC-8.2: the cross-encoder decides *which* jobs are considered, never *why*
    one scored what it did. Its scores are zeroed so nothing downstream can read
    them into an explanation.
    """
    from src.rerank import rerank_candidates

    out = rerank_candidates(index, JOBS, PROFILE, [(0, 0.0), (1, 0.0)], top_k=2)
    assert all(score == 0.0 for _, score in out)


def test_ac_8_2_no_rerank_component_in_the_breakdown(index, kb):
    from src.pipeline import search
    from src.scoring import WEIGHTS

    result = search(JOBS, index, PROFILE, kb, rerank=True)
    for r in result.results:
        assert {c["name"] for c in r["components"]} == set(WEIGHTS)
        assert not any("rerank" in c["name"] or "cross" in c["name"] for c in r["components"])


# ---------------------------------------------------------------- default-off

def test_req8_is_off_by_default(index, kb):
    """
    At k=400 hybrid retrieval already recovers 100% of the true top-20, so this
    stage has nothing to recover and would add seconds to a sub-second search.
    It must never enable itself.
    """
    from src.pipeline import search

    assert "rerank" not in search(JOBS, index, PROFILE, kb).funnel["retrieval"]


def test_req8_reports_itself_when_enabled(index, kb):
    from src.pipeline import search

    funnel = search(JOBS, index, PROFILE, kb, rerank=True).funnel
    assert "rerank" in funnel["retrieval"], "an enabled stage must be visible in the funnel"


def test_req8_enabled_still_produces_a_valid_breakdown(index, kb):
    """AC-9.14 must hold on the reranked path too."""
    from src.pipeline import search

    for r in search(JOBS, index, PROFILE, kb, rerank=True).results:
        assert round(sum(c["weighted"] for c in r["components"]) * 100) == r["score"]
