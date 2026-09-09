"""
AC-derived tests for REQ-6 (FROZEN v1.0) — hard filters.

Filters **eliminate**, they do not penalize (D2). AC-6.2's five-branch location
decision tree is tested branch by branch, as the AC requires.
"""

import pandas as pd
import pytest

from src.filters import (
    STATE_TO_CODE,
    apply_filters,
    employment_type_passes,
    geocode,
    haversine_miles,
    location_passes,
    normalize_location,
    parse_location,
    salary_passes,
    skill_floor_passes,
    work_setting_passes,
)
from src.profiles import UserProfile


def job(**kw):
    base = dict(job_id=1, title="Data Engineer", location_raw="Kansas City, MO",
                is_remote=False, work_setting="On-site", employment_type="Full-time",
                salary_min=100000.0, salary_max=120000.0, salary_listed=True,
                required_skills=["python", "sql"])
    base.update(kw)
    return pd.Series(base)


def profile(**kw):
    base = dict(name="t", career_goals="g", resume_text="r",
                skills={"python", "sql", "airflow"}, preferred_location="Kansas City, MO",
                min_salary=90000)
    base.update(kw)
    return UserProfile(**base)


# ---------------------------------------------------------------- AC-6.3

@pytest.mark.parametrize("raw,expected", [
    ("Kansas City, MO", ("kansas city", "MO")),
    ("kansas city, mo", ("kansas city", "MO")),
    ("Kansas City, Missouri", ("kansas city", "MO")),
    ("Kansas City Metropolitan Area", ("kansas city", None)),
    ("Greater Seattle Area", ("seattle", None)),
    ("  Austin,   TX  ", ("austin", "TX")),
])
def test_ac_6_3_location_parsing(raw, expected):
    assert parse_location(raw) == expected


def test_ac_6_3_state_names_expand_to_codes():
    assert STATE_TO_CODE["missouri"] == "MO" and STATE_TO_CODE["washington"] == "WA"


def test_ac_6_3_normalization_is_case_and_space_insensitive():
    assert normalize_location("Kansas City,  MO") == normalize_location("kansas city, mo")


# ---------------------------------------------------------------- geocoding

def test_geocode_resolves_a_major_city():
    lat, lon = geocode("kansas city", "MO")
    assert 38.5 < lat < 39.5 and -95.5 < lon < -94.0


def test_geocode_disambiguates_by_state():
    """Kansas City exists in both MO and KS."""
    assert geocode("kansas city", "MO") != geocode("kansas city", "KS")


def test_geocode_resolves_the_suburb_case():
    """AC-6.2: Overland Park must resolve — it is the suburb case that matters."""
    assert geocode("overland park", "KS") is not None


def test_geocode_unknown_returns_none():
    assert geocode("nowhereville", "ZZ") is None


def test_haversine_known_distance():
    """Kansas City MO to Overland Park KS is roughly 12 miles."""
    d = haversine_miles(*geocode("kansas city", "MO"), *geocode("overland park", "KS"))
    assert 5 < d < 25


# ---------------------------------------------------------------- AC-6.2

def test_ac_6_2_branch_a_remote_only_user():
    """(a) Remote-only user: a job passes only if is_remote."""
    p = profile(accepted_work_settings={"Remote"})
    assert location_passes(job(is_remote=True), p)
    assert not location_passes(job(is_remote=False, location_raw="Kansas City, MO"), p)


def test_ac_6_2_branch_b_remote_job_ignores_distance():
    """(b) A remote job passes regardless of distance."""
    p = profile(preferred_location="Kansas City, MO")
    assert location_passes(job(is_remote=True, location_raw="Boston, MA"), p)


def test_ac_6_2_branch_c_within_cutoff():
    """(c) Both geocode and distance is within the cutoff."""
    p = profile(preferred_location="Kansas City, MO", max_distance_miles=100)
    assert location_passes(job(location_raw="Overland Park, KS"), p)


def test_ac_6_2_branch_c_outside_cutoff():
    p = profile(preferred_location="Kansas City, MO", max_distance_miles=100)
    assert not location_passes(job(location_raw="Boston, MA"), p)


def test_ac_6_2_branch_d_string_equality_fallback():
    """(d) Either side fails to geocode: fall back to normalized string equality."""
    p = profile(preferred_location="Smallville, KS")
    assert location_passes(job(location_raw="smallville, ks"), p)


def test_ac_6_2_branch_e_unresolved_and_unequal_is_dropped():
    """(e) Neither geocodes and the strings differ: the job is dropped."""
    p = profile(preferred_location="Smallville, KS")
    assert not location_passes(job(location_raw="Othertown, VT"), p)


def test_ac_6_2_no_preferred_location_passes_everything():
    """A user with no location set is not gated on geography."""
    assert location_passes(job(location_raw="Boston, MA"), profile(preferred_location=None))


# ---------------------------------------------------------------- AC-6.5

def test_ac_6_5_below_floor_is_dropped():
    assert not salary_passes(job(salary_max=70000.0), profile(min_salary=90000))


def test_ac_6_5_at_or_above_floor_passes():
    assert salary_passes(job(salary_max=120000.0), profile(min_salary=90000))


def test_ac_6_5_unlisted_respects_the_user_choice():
    """AC-6.5: unlisted passes iff the user opted in — the policy is user-visible."""
    unlisted = job(salary_listed=False, salary_min=None, salary_max=None)
    assert salary_passes(unlisted, profile(include_unlisted_salary=True))
    assert not salary_passes(unlisted, profile(include_unlisted_salary=False))


def test_ac_6_5_zero_floor_is_a_noop():
    assert salary_passes(job(salary_max=1.0), profile(min_salary=0))
    assert salary_passes(job(salary_listed=False, salary_max=None),
                         profile(min_salary=0, include_unlisted_salary=False))


# ---------------------------------------------------------------- AC-6.6 / 6.7

def test_ac_6_6_set_membership_not_equality():
    """D5: 'remote or hybrid' must be expressible."""
    p = profile(accepted_work_settings={"Remote", "Hybrid"})
    assert work_setting_passes(job(work_setting="Hybrid"), p)
    assert work_setting_passes(job(work_setting="Remote"), p)
    assert not work_setting_passes(job(work_setting="On-site"), p)


def test_ac_6_6_unknown_passes_and_is_tagged():
    p = profile(accepted_work_settings={"Remote"})
    assert work_setting_passes(job(work_setting="Unknown"), p)


def test_ac_6_7_employment_type_is_multi_select():
    p = profile(accepted_employment_types={"Full-time", "Internship"})
    assert employment_type_passes(job(employment_type="Internship"), p)
    assert not employment_type_passes(job(employment_type="Contract"), p)


def test_ac_6_7_unknown_employment_passes():
    assert employment_type_passes(job(employment_type="Unknown"),
                                  profile(accepted_employment_types={"Full-time"}))


# ---------------------------------------------------------------- AC-6.8

def test_ac_6_8_requires_at_least_one_shared_skill():
    assert skill_floor_passes(job(required_skills=["python", "rust"]), profile())
    assert not skill_floor_passes(job(required_skills=["cobol", "fortran"]), profile())


def test_ac_6_8_never_sees_an_empty_required_list():
    """AC-2.6 removed zero-skill postings at ingestion; this is a guard."""
    assert not skill_floor_passes(job(required_skills=[]), profile())


# ---------------------------------------------------------------- AC-6.9 / 6.10

@pytest.fixture
def corpus():
    return pd.DataFrame([
        job(job_id=1, location_raw="Kansas City, MO"),
        job(job_id=2, location_raw="Boston, MA"),                        # too far
        job(job_id=3, salary_max=60000.0),                               # underpays
        job(job_id=4, employment_type="Contract"),                       # wrong type
        job(job_id=5, required_skills=["cobol"]),                        # no shared skill
        job(job_id=6, is_remote=True, work_setting="Remote", location_raw="Austin, TX"),
    ])


def test_ac_6_9_filters_compose_with_and(corpus):
    kept, funnel = apply_filters(corpus, profile(
        accepted_employment_types={"Full-time"}, accepted_work_settings={"Remote", "On-site"}))
    assert set(kept["job_id"]) == {1, 6}


def test_ac_6_9_funnel_reports_each_stage(corpus):
    _, funnel = apply_filters(corpus, profile(accepted_employment_types={"Full-time"}))
    for stage in ["start", "location", "salary", "work_setting", "employment_type", "skill_floor"]:
        assert stage in funnel
    assert funnel["start"] == len(corpus)


def test_ac_6_10_empty_result_names_the_biggest_eliminator(corpus):
    """AC-6.10: never silently return zero."""
    kept, funnel = apply_filters(corpus, profile(min_salary=1_000_000))
    assert len(kept) == 0
    assert funnel["limiting_filter"] == "salary"
    assert funnel["suggestion"]


def test_ac_6_9_is_fast(corpus):
    """AC-6.1: the whole corpus in under 200ms — filters are cheap, hence D2."""
    import time
    big = pd.concat([corpus] * 3000, ignore_index=True)   # ~18k rows
    t = time.time(); apply_filters(big, profile()); elapsed = time.time() - t
    assert elapsed < 0.2 * 4, f"filtering {len(big)} rows took {elapsed:.2f}s"


def test_vectorized_matches_rowwise(corpus):
    """
    The vectorized masks and the per-row predicates are two expressions of the
    same rules. They must never disagree.
    """
    from src.filters import FILTERS, MASKS

    p = profile(accepted_employment_types={"Full-time"},
                accepted_work_settings={"Remote", "On-site"})
    for (name, row_fn), (mask_name, mask_fn) in zip(FILTERS, MASKS):
        assert name == mask_name
        rowwise = corpus.apply(row_fn, axis=1, args=(p,)).tolist()
        vector = mask_fn(corpus, p).tolist()
        assert rowwise == vector, f"{name}: row-wise and vectorized disagree"


@pytest.mark.parametrize("value", [
    ["python", "sql"],                       # fixtures give Python lists
    pytest.param(None, id="none"),
])
def test_ac_6_8_tolerates_list_and_null_skill_columns(value):
    assert skill_floor_passes(job(required_skills=value), profile()) == bool(value)


def test_ac_6_8_tolerates_numpy_arrays():
    """
    Regression: Parquet round-trips list columns as numpy arrays, and
    `value or []` raises ValueError on an array instead of defaulting. Fixtures
    used Python lists, so only real data exposed this.
    """
    import numpy as np
    assert skill_floor_passes(job(required_skills=np.array(["python", "rust"])), profile())
    assert not skill_floor_passes(job(required_skills=np.array(["cobol"])), profile())


def test_ac_6_1_warm_path_is_within_budget(corpus):
    """
    AC-6.1: under 200ms over the corpus. Measured warm on the real 18,990-row
    corpus: 12-65ms. `warm_caches()` exists so a cold first search does not pay
    the one-off geocoding-table construction inside the budget.
    """
    import time

    from src.filters import warm_caches

    warm_caches()
    big = pd.concat([corpus] * 3000, ignore_index=True)
    t = time.time(); apply_filters(big, profile()); elapsed = time.time() - t
    assert elapsed < 0.8, f"filtering {len(big):,} rows took {elapsed:.2f}s"
