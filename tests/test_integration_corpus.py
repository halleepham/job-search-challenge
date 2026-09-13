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


@pytest.fixture(scope="module")
def wide_searches(env):
    """Top-20, the population AC-9.11a v1.1 measures spread over."""
    from src.personal_kb import PersonalKB
    from src.pipeline import search
    from src.profiles import PRESETS

    jobs, index = env
    return {key: search(jobs, index, p, PersonalKB.build(p.resume_text, p.career_goals),
                        top_n=20)
            for key, p in PRESETS.items()}


def _sub(result, name):
    return [c["sub_score"] for r in result.results for c in r["components"] if c["name"] == name]


@pytest.mark.parametrize("key", ["data_science_student", "senior_backend_engineer",
                                 "security_analyst"])
def test_every_profile_returns_results(searches, key):
    assert len(searches[key].results) == 5


@pytest.mark.parametrize("key", ["data_science_student", "senior_backend_engineer",
                                 "security_analyst"])
def test_ac_9_11a_semantic_scores_discriminate(searches, wide_searches, key):
    """
    AC-9.11a v1.1: calibrated sub-scores must vary across the scored population.

    Measured over the top 20 rather than the top 5: with 229 survivors, a top-5
    sitting entirely above p95 *is* the top 2.2%, so identical values there are a
    correct outcome. Within the top 5 we require only that at least one of the two
    components varies — a flat pair would mean 30% of the weight is constant,
    which is the defect this AC exists to catch.
    """
    for name in ("career_goals_similarity", "resume_evidence_similarity"):
        wide = _sub(wide_searches[key], name)
        assert len(set(round(v, 2) for v in wide)) > 1, (
            f"{key}/{name} is constant across the top-20: {wide}")

    top5 = [len(set(round(v, 2) for v in _sub(searches[key], n)))
            for n in ("career_goals_similarity", "resume_evidence_similarity")]
    assert max(top5) > 1, f"{key}: both semantic components are flat across the top-5"


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


def test_ac_5_4_cached_index_loads_within_budget(env):
    """AC-5.4: under 10s — the justification for caching rather than re-embedding."""
    import time

    from src.indexing import JobIndex

    jobs, _ = env
    start = time.perf_counter()
    reloaded = JobIndex.load_or_build(jobs)
    elapsed = time.perf_counter() - start
    assert reloaded.from_cache, "index must load from disk, not rebuild"
    assert elapsed < 10, f"cached load took {elapsed:.1f}s"


def test_ac_6_4_geocoding_resolution_is_measured(env):
    """
    AC-6.4: the location filter is the most consequential one, so its coverage is
    measured rather than assumed. Roughly 69% of rows resolve; the rest fall to
    AC-6.2's string-equality branch by design.
    """
    from src.filters import geocode, parse_location

    jobs, _ = env
    locations = jobs["location_raw"].dropna().unique()
    resolved = {loc: bool(geocode(*parse_location(loc))) for loc in locations}
    rate = jobs["location_raw"].map(resolved).fillna(False).mean()
    assert 0.5 < rate < 1.0, f"geocoding resolves {rate:.0%} of rows"


def test_ac_7_5_retrieval_is_interactive(env):
    """AC-7.5: under 2s on CPU."""
    import time

    from src.filters import apply_filters
    from src.profiles import PRESETS
    from src.retrieval import retrieve

    jobs, index = env
    p = PRESETS["data_science_student"]
    kept, _ = apply_filters(jobs, p)
    position_of = {j: i for i, j in enumerate(jobs["job_id"])}
    candidates = [position_of[j] for j in kept["job_id"]]

    start = time.perf_counter()
    retrieve(index, p, candidate_positions=candidates, mode="hybrid", k=400)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"retrieval took {elapsed:.2f}s"


def test_ac_1_11_every_job_links_to_its_source(env):
    """AC-1.11: 'Applied' needs somewhere to go, and expiry must be disclosable."""
    jobs, _ = env
    assert jobs["posting_url"].notna().all()
    assert jobs["posting_url"].str.startswith("http").all()
    assert jobs["expiry_date"].notna().all()


def test_ac_2_7_coverage_stats_are_published(env):
    """AC-2.7: extraction is measured, not assumed — and the report cites these."""
    import json
    from pathlib import Path

    stats = json.loads(Path("data/processed/coverage_stats.json").read_text())
    skills = stats["skills"]
    for field in ["pct_with_required_section", "pct_with_preferred_section",
                  "pct_with_preferred_skills", "median_required_skills",
                  "pct_with_one_required_skill"]:
        assert field in skills
    assert stats["funnel"]["after_zero_skill_drop"] == len(env[0])


def test_ac_11_7_no_profile_field_is_collected_and_ignored():
    """
    AC-11.7: every field the form collects must reach a filter or a score
    component. Storing education and salary and then reading neither is the exact
    defect in the Stage 2 AI code (`agent-exercise:src/models.py:17-18`).
    """
    import inspect

    from src import filters, pipeline, scoring
    from src.profiles import UserProfile

    # Whole modules, not three entry points: `preferred_location` is consumed by
    # `_location_mask` and `_distance_for`, so inspecting only the callers would
    # report a field as unused that the pipeline plainly reads.
    consumed = " ".join(inspect.getsource(m) for m in (filters, scoring, pipeline))
    # Reaches scoring through the knowledge base rather than by attribute name.
    indirect = {"name", "resume_text"}
    for field in UserProfile.__dataclass_fields__:
        if field not in indirect:
            assert field in consumed, f"UserProfile.{field} is collected but never read"
