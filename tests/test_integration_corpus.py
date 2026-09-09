"""
Integration tests against the real built corpus and index.

Skipped when the corpus has not been built, so the unit suite stays runnable on
a fresh clone. These pin the properties that only appear at real scale - two of
them were defects that every unit test passed straight through.
"""

import numpy as np
import pandas as pd
import pytest

CORPUS = "data/processed/jobs_tech.parquet"
pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(CORPUS).exists()
    or not __import__("pathlib").Path("data/index/manifest.json").exists(),
    reason="real corpus/index not built (run: python -m src.ingest, then build the index)",
)


@pytest.fixture(scope="module")
def env():
    from src.filters import warm_caches
    from src.indexing import JobIndex

    warm_caches()
    jobs = pd.read_parquet(CORPUS)
    return jobs, JobIndex.load_or_build(jobs)


@pytest.fixture(scope="module")
def searches(env):
    from src.personal_kb import PersonalKB
    from src.pipeline import search
    from src.profiles import PRESETS

    jobs, index = env
    out = {}
    for key, profile in PRESETS.items():
        kb = PersonalKB.build(profile.resume_text, profile.career_goals)
        out[key] = search(jobs, index, profile, kb)
    return out


def _sub(result, name):
    return [c["sub_score"] for r in result.results for c in r["components"] if c["name"] == name]


@pytest.mark.parametrize("key", ["data_science_student", "senior_backend_engineer",
                                 "security_analyst"])
def test_every_profile_returns_results(searches, key):
    assert len(searches[key].results) == 5


@pytest.mark.parametrize("key", ["data_science_student", "senior_backend_engineer",
                                 "security_analyst"])
def test_ac_9_11a_semantic_scores_discriminate(searches, key):
    """
    AC-9.11a: calibrated sub-scores must vary across a real top-5.

    This is the defect the unit tests could not see. Calibrating on 2,000 random
    corpus jobs put the p95 anchor below the p5 of the retrieved set, so every
    scored candidate clipped to 1.0 and 30% of the weight became a constant.
    """
    for name in ("career_goals_similarity", "resume_evidence_similarity"):
        values = _sub(searches[key], name)
        assert len(set(round(v, 2) for v in values)) > 1, (
            f"{key}/{name} is constant across the top-5: {values}")


@pytest.mark.parametrize("key", ["data_science_student", "senior_backend_engineer",
                                 "security_analyst"])
def test_ac_9_12_semantic_components_are_not_one_signal(searches, key):
    """
    AC-9.12: the two semantic components must not be the same measurement.

    Before AC-9.10 excluded the career-goals chunk from the evidence max, the
    goals chunk won that max and both components returned identical values on
    most results.
    """
    goals = _sub(searches[key], "career_goals_similarity")
    evidence = _sub(searches[key], "resume_evidence_similarity")
    identical = sum(abs(g - e) < 0.01 for g, e in zip(goals, evidence))
    assert identical < len(goals), f"{key}: all {len(goals)} results have goals == evidence"


@pytest.mark.parametrize("key", ["data_science_student", "senior_backend_engineer",
                                 "security_analyst"])
def test_ac_9_14_breakdown_sums_on_real_data(searches, key):
    for r in searches[key].results:
        assert round(sum(c["weighted"] for c in r["components"]) * 100) == r["score"]


def test_ac_9_11_scores_are_pool_independent(env):
    """
    D8: a job's score must not depend on which other jobs were retrieved.
    Scoring the same job against a larger corpus slice must not move its score.
    """
    from src.personal_kb import PersonalKB
    from src.pipeline import search
    from src.profiles import PRESETS

    jobs, index = env
    p = PRESETS["data_science_student"]
    kb = PersonalKB.build(p.resume_text, p.career_goals)
    a = search(jobs, index, p, kb, retrieve_k=200)
    b = search(jobs, index, p, kb, retrieve_k=400)
    common = {r["job_id"]: r["score"] for r in a.results} | {}
    for r in b.results:
        if r["job_id"] in common:
            assert r["score"] == common[r["job_id"]], "score moved when the pool changed"


def test_ac_13_3_end_to_end_latency(searches):
    """AC-13.3: a full search stays interactive."""
    for key, res in searches.items():
        assert res.timings_ms["total"] < 5000, f"{key} took {res.timings_ms['total']:.0f} ms"


def test_results_are_domain_appropriate(searches):
    """A sanity check no unit fixture can give: does the ranking make sense?"""
    titles = " ".join(r["job"]["title"].lower() for r in searches["security_analyst"].results)
    assert any(w in titles for w in ("security", "cyber", "threat", "soc", "information"))
