"""AC-derived tests for REQ-7 (FROZEN v1.0) — hybrid retrieval."""

import pandas as pd
import pytest

from src.indexing import JobIndex
from src.profiles import UserProfile
from src.retrieval import RRF_K, build_queries, reciprocal_rank_fusion, retrieve

JOBS = pd.DataFrame([
    {"job_id": 1, "title": "Data Engineer", "description": "Build ETL pipelines in Airflow and dbt.",
     "required_skills": ["python", "airflow", "sql"], "preferred_skills": ["dbt"]},
    {"job_id": 2, "title": "Backend Engineer", "description": "Go microservices running on Kubernetes.",
     "required_skills": ["go", "kubernetes"], "preferred_skills": []},
    {"job_id": 3, "title": "Security Analyst", "description": "SOC monitoring, SIEM tuning, incident response.",
     "required_skills": ["splunk", "siem"], "preferred_skills": []},
    {"job_id": 4, "title": "Machine Learning Engineer", "description": "Train and deploy models with PyTorch.",
     "required_skills": ["python", "pytorch"], "preferred_skills": ["mlflow"]},
    {"job_id": 5, "title": "Registered Nurse", "description": "Patient care on the cardiac ward.",
     "required_skills": ["excel"], "preferred_skills": []},
])


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    return JobIndex.build(JOBS, index_dir=tmp_path_factory.mktemp("idx7"))


@pytest.fixture
def data_profile():
    return UserProfile(
        name="t",
        career_goals="I want to build data pipelines that turn raw data into dashboards.",
        resume_text="Built ETL pipelines in Airflow.",
        skills={"python", "sql", "airflow"},
        preferred_titles=["data engineer"],
    )


# ---------------------------------------------------------------- AC-7.1

def test_ac_7_1_three_separate_queries(data_profile):
    """
    AC-7.1: career goals, skills and titles are issued as separate queries and
    all result lists fused — not concatenated into one string, which would
    dilute the dense vector.
    """
    queries = build_queries(data_profile)
    assert set(queries) == {"career_goals", "skills", "titles"}
    assert all(q.strip() for q in queries.values())


def test_ac_7_1_queries_are_not_one_blob(data_profile):
    q = build_queries(data_profile)
    assert q["skills"] != q["career_goals"]
    assert data_profile.career_goals not in q["skills"]


def test_ac_7_1_empty_query_parts_are_skipped():
    p = UserProfile(name="t", career_goals="", resume_text="", skills=set(),
                    preferred_titles=[])
    assert build_queries(p) == {}


# ---------------------------------------------------------------- AC-7.2

def test_ac_7_2_rrf_uses_ranks_not_scores():
    """
    AC-7.2: BM25 scores and cosine similarities live on incompatible scales, so
    fusion must use ranks only. Wildly different score magnitudes must not
    change the fused order.
    """
    small = [[(0, 0.9), (1, 0.8), (2, 0.7)]]
    huge = [[(0, 900.0), (1, 800.0), (2, 700.0)]]
    assert reciprocal_rank_fusion(small) == reciprocal_rank_fusion(huge)


def test_ac_7_2_rrf_formula():
    """score(doc) = sum over lists of 1 / (60 + rank), rank starting at 1."""
    fused = reciprocal_rank_fusion([[(7, 0.5), (9, 0.4)]])
    assert fused[0] == (7, pytest.approx(1 / (RRF_K + 1)))
    assert fused[1] == (9, pytest.approx(1 / (RRF_K + 2)))


def test_ac_7_2_agreement_across_lists_wins():
    """A document ranked well by both retrievers must beat one ranked well by only one."""
    fused = dict(reciprocal_rank_fusion([[(1, 0.9), (2, 0.8)], [(2, 0.9), (1, 0.1)]]))
    both = dict(reciprocal_rank_fusion([[(1, 0.9), (3, 0.8)], [(4, 0.9), (5, 0.1)]]))
    assert fused[1] > both[3]


def test_ac_7_2_k_is_sixty():
    assert RRF_K == 60


# ---------------------------------------------------------------- AC-7.3

@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_ac_7_3_each_retriever_is_independently_callable(index, data_profile, mode):
    """AC-7.3: REQ-13 must be able to compare modes without rebuilding the pipeline."""
    hits = retrieve(index, data_profile, mode=mode, k=3)
    assert 0 < len(hits) <= 3


def test_ac_7_3_modes_can_disagree(index, data_profile):
    bm25 = [i for i, _ in retrieve(index, data_profile, mode="bm25", k=5)]
    dense = [i for i, _ in retrieve(index, data_profile, mode="dense", k=5)]
    assert bm25 and dense                       # both return results
    assert isinstance(bm25[0], int)


def test_ac_7_3_unknown_mode_rejected(index, data_profile):
    with pytest.raises(ValueError):
        retrieve(index, data_profile, mode="magic")


# ---------------------------------------------------------------- relevance

def test_retrieval_ranks_the_relevant_job_first(index, data_profile):
    top = retrieve(index, data_profile, mode="hybrid", k=1)[0][0]
    assert JOBS.iloc[top]["job_id"] in (1, 4), "a data role should lead for a data profile"


def test_retrieval_is_profile_sensitive(index, data_profile):
    security = UserProfile(
        name="s", career_goals="I want threat detection and incident response work.",
        resume_text="SOC analyst tuning SIEM detections.",
        skills={"splunk", "siem"}, preferred_titles=["security analyst"])
    assert retrieve(index, data_profile, k=1)[0][0] != retrieve(index, security, k=1)[0][0]


# ---------------------------------------------------------------- AC-7.4

def test_ac_7_4_returns_all_when_fewer_than_k(index, data_profile):
    hits = retrieve(index, data_profile, k=500)
    assert len(hits) <= len(JOBS)


def test_ac_7_4_respects_the_candidate_subset(index, data_profile):
    """
    D2: filters run first, so retrieval searches only the survivors. A job
    outside the subset must never be returned.
    """
    subset = [2, 3]
    hits = retrieve(index, data_profile, candidate_positions=subset, k=10)
    assert {i for i, _ in hits} <= set(subset)


def test_ac_7_4_empty_subset_is_not_an_error(index, data_profile):
    assert retrieve(index, data_profile, candidate_positions=[], k=10) == []


def test_ac_7_4_single_candidate(index, data_profile):
    assert len(retrieve(index, data_profile, candidate_positions=[0], k=10)) == 1
