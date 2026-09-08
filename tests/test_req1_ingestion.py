"""
AC-derived tests for REQ-1 (FROZEN v1.0) — phase 1a only.

Covers AC-1.1 (tech scoping), AC-1.2 (scoping fixture behaviour), AC-1.3 (dedupe).
AC-1.4 through AC-1.10 are phases 1b/1c and are not exercised here.
"""

from pathlib import Path

import pytest

from src.ingest import (
    TECH_CODES,
    is_tech_title,
    normalize_key,
    normalize_title,
    scope_and_dedupe,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def scoped():
    """Run phase 1a over the 20-row fixture (AC-1.2)."""
    return scope_and_dedupe(
        postings_path=FIXTURES / "postings_sample.csv",
        job_skills_path=FIXTURES / "job_skills_sample.csv",
    )


# ---------------------------------------------------------------- AC-1.1

def test_ac_1_1_tech_codes_are_a_named_constant():
    """AC-1.1: TECH_CODES is declared as a named constant, not inlined."""
    assert TECH_CODES == frozenset({"IT", "ENG", "ANLS", "QA", "SCI"})


def test_ac_1_1_join_does_not_fan_out(scoped):
    """
    AC-1.1: job_skills.csv holds ~1.69 rows per job, so the join must be a
    semi-join. A naive inner join would emit job 1 twice (IT and ENG) and
    job 10 twice (SCI and IT).
    """
    df = scoped.frame
    assert df["job_id"].is_unique, "join fanned out — use a semi-join, not an inner join"
    assert (df["job_id"] == 1).sum() == 1, "job 1 has two skill codes but must appear once"
    assert (df["job_id"] == 10).sum() == 1, "job 10 has two skill codes but must appear once"


def test_ac_1_1_retained_by_code_alone(scoped):
    """AC-1.1: a posting whose skill_abr is in TECH_CODES is retained."""
    ids = set(scoped.frame["job_id"])
    assert 3 in ids, "ANLS-coded Data Analyst must be retained"
    assert 8 in ids, "QA-coded QA Automation Engineer must be retained"
    assert 10 in ids, "SCI-coded ML Scientist must be retained"


def test_ac_1_1_retained_by_title_alone(scoped):
    """
    AC-1.1 / AC-1.2 v1.2: retention is code OR title. Job 21 is a genuinely
    technical role coded MGMT — exactly what the title branch exists to recover.
    """
    assert "MGMT" not in TECH_CODES
    assert 21 in set(scoped.frame["job_id"]), "title-regex branch of the OR is not firing"


# ---------------------------------------------------------------- AC-1.2

def test_ac_1_2_non_tech_postings_dropped(scoped):
    """AC-1.2: postings in non-tech job functions with non-tech titles are dropped."""
    ids = set(scoped.frame["job_id"])
    for job_id, what in [(4, "Registered Nurse"), (5, "Warehouse Associate"),
                         (9, "Worship Leader"), (13, "Account Executive"),
                         (17, "Physical Therapist"), (19, "Barista")]:
        assert job_id not in ids, f"{what} must not survive tech scoping"


def test_ac_1_2_branches_have_different_postures(scoped):
    """
    AC-1.2 v1.2: the two branches of the OR are deliberately asymmetric.
    TECH_CODES is recall-oriented (ENG covers all engineering, so a Mechanical
    Engineer survives and AC-2.6 removes it later). The title branch is
    precision-oriented (compounds only), so a Sales Engineer coded SALE is dropped.
    """
    ids = set(scoped.frame["job_id"])
    assert 7 in ids, "Mechanical Engineer must survive — ENG branch is recall-oriented"
    assert 6 not in ids, "Sales Engineer must be dropped — title branch is precision-oriented"
    assert 15 not in ids, "Construction PM: MGMT code and no title-regex match — correctly dropped"


def test_ac_1_2_scoping_counts_are_reported(scoped):
    """AC-1.2 / AC-1.8: the funnel records what happened at each stage."""
    assert scoped.n_raw == 21
    assert scoped.n_after_scoping < scoped.n_raw
    assert scoped.n_after_dedupe <= scoped.n_after_scoping


# ---------------------------------------------------------------- AC-1.3

def test_ac_1_3_normalization_precedes_comparison():
    """AC-1.3: all three dedupe keys are lowercased and whitespace-collapsed."""
    assert normalize_key("ACME CORP") == normalize_key("Acme Corp")
    assert normalize_key("Kansas City,  MO") == normalize_key("kansas city, mo")
    assert normalize_key("  Beta   LLC  ") == "beta llc"


def test_ac_1_3_case_and_whitespace_duplicate_removed(scoped):
    """
    AC-1.3: job 11 is job 1 with different casing across all three key fields.
    After normalization they collide and exactly one survives.
    """
    ids = set(scoped.frame["job_id"])
    assert not (1 in ids and 11 in ids), "case-variant duplicate was not removed"
    assert 1 in ids or 11 in ids, "both copies were dropped — dedupe is too aggressive"


def test_ac_1_3_distinct_titles_at_same_company_are_kept(scoped):
    """
    AC-1.3: dedupe is on the full triple. Job 12 (Data Engineer) shares company
    and location with job 1 (Senior Data Engineer). They are distinct openings, so
    the dedupe key must preserve seniority (AC-1.3 v1.1) even though
    title_normalized strips it for AC-1.1.
    """
    ids = set(scoped.frame["job_id"])
    assert 12 in ids, "distinct title at same company must not be merged away"


def test_ac_1_3_dedupe_is_deterministic():
    """AC-1.3 / AC-1.9: repeated runs keep the same survivor, not an arbitrary one."""
    a = scope_and_dedupe(FIXTURES / "postings_sample.csv", FIXTURES / "job_skills_sample.csv")
    b = scope_and_dedupe(FIXTURES / "postings_sample.csv", FIXTURES / "job_skills_sample.csv")
    assert list(a.frame["job_id"]) == list(b.frame["job_id"])


def test_ac_1_3_dropped_count_recorded(scoped):
    """AC-1.3: the dedupe drop count is recorded for coverage_stats.json."""
    assert scoped.n_dropped_duplicates == scoped.n_after_scoping - scoped.n_after_dedupe
    assert scoped.n_dropped_duplicates >= 1, "fixture contains a known duplicate pair"


# ------------------------------------------------- pure-function unit tests
# AC-1.10: per-row logic is pure Python, testable with no database connection.

@pytest.mark.parametrize("raw,expected", [
    ("Senior Data Engineer", "data engineer"),
    ("Sr. Software Engineer", "software engineer"),
    ("Software Engineer II", "software engineer"),
    ("Lead Data Scientist", "data scientist"),
    ("Principal Backend Developer", "backend developer"),
    ("Junior QA Analyst", "qa analyst"),
    ("  Data   Analyst  ", "data analyst"),
])
def test_normalize_title_strips_seniority(raw, expected):
    assert normalize_title(raw) == expected


def test_normalize_title_keeps_meaningful_words():
    """Stripping must not eat real title content."""
    assert normalize_title("Site Reliability Engineer") == "site reliability engineer"
    assert "data" in normalize_title("Staff Data Engineer")


@pytest.mark.parametrize("title", [
    "data engineer", "software engineer", "data analyst", "machine learning scientist",
    "qa automation engineer", "cloud infrastructure architect", "full stack developer",
    "security analyst", "database administrator", "software engineering manager",
])
def test_is_tech_title_accepts(title):
    assert is_tech_title(title)


@pytest.mark.parametrize("title", [
    "registered nurse", "warehouse associate", "worship leader",
    "physical therapist", "barista", "account executive",
    # AC-1.2 v1.2: bare role words are noise, measured on the real corpus
    "sales engineer", "financial analyst", "board certified behavior analyst",
    "office administrator", "data entry clerk", "technical writer",
    "process engineer", "mechanical engineer",
])
def test_is_tech_title_rejects(title):
    assert not is_tech_title(title)
