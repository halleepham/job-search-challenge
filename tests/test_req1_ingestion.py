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


# ============================================================ PHASE 1b
# AC-1.4 salary · AC-1.5 work setting · AC-1.6/1.6a experience + employment type
# · AC-1.7 education. All pure functions — no database connection (AC-1.10).

from src.ingest import (  # noqa: E402
    EDUCATION_LEVELS,
    SENIORITY_YEARS,
    normalize_employment_type,
    normalize_salary,
    parse_education,
    parse_min_years,
    derive_work_setting,
)


# ---------------------------------------------------------------- AC-1.4

def test_ac_1_4_prefers_normalized_salary():
    """AC-1.4: normalized_salary is authoritative when non-null."""
    lo, hi, listed, src = normalize_salary(38480.0, 17.0, 20.0, "HOURLY", "USD")
    assert listed and src == "normalized_salary"
    assert (lo, hi) == (17.0 * 2080, 20.0 * 2080), "range annualized from min/max"


def test_ac_1_4_normalized_without_range():
    """AC-1.4: 6,280 real rows carry normalized_salary but no min/max."""
    lo, hi, listed, src = normalize_salary(90000.0, None, None, "YEARLY", "USD")
    assert (lo, hi, listed, src) == (90000.0, 90000.0, True, "normalized_salary")


@pytest.mark.parametrize("period,mult", [("HOURLY", 2080), ("MONTHLY", 12), ("YEARLY", 1)])
def test_ac_1_4_derivation_fallback(period, mult):
    """AC-1.4: with normalized_salary null, derive from min/max x pay_period."""
    lo, hi, listed, src = normalize_salary(None, 10.0, 20.0, period, "USD")
    assert (lo, hi) == (10.0 * mult, 20.0 * mult)
    assert listed and src == "derived"


@pytest.mark.parametrize("period", [None, "WEEKLY", "BIWEEKLY", "PER_PROJECT"])
def test_ac_1_4_unhandled_period_yields_none(period):
    """AC-1.4: any other or missing pay_period yields None / salary_listed False."""
    assert normalize_salary(None, 10.0, 20.0, period, "USD") == (None, None, False, "none")


def test_ac_1_4_no_salary_at_all():
    assert normalize_salary(None, None, None, None, None) == (None, None, False, "none")


def test_ac_1_4_non_usd_rejected():
    """AC-1.4 says annual USD. 11 real rows are EUR/CAD/BBD and cannot be converted."""
    assert normalize_salary(90000.0, None, None, "YEARLY", "EUR") == (None, None, False, "non_usd")


def test_ac_1_4_every_case_has_a_rule():
    """AC-1.4: exactly one rule per input case, nothing falls through."""
    cases = [(38480.0, 17.0, 20.0, "HOURLY", "USD"), (90000.0, None, None, "YEARLY", "USD"),
             (None, 10.0, 20.0, "YEARLY", "USD"), (None, 10.0, 20.0, "WEEKLY", "USD"),
             (None, None, None, None, None), (90000.0, None, None, "YEARLY", "EUR")]
    sources = {normalize_salary(*c)[3] for c in cases}
    assert sources == {"normalized_salary", "derived", "none", "non_usd"}


# ---------------------------------------------------------------- AC-1.5

def test_ac_1_5_remote_flag_wins():
    assert derive_work_setting(1.0, "Data Engineer", "Work from anywhere") == ("Remote", False)


def test_ac_1_5_null_flag_means_not_remote():
    """
    AC-1.5: remote_allowed is a sparse flag (1.0 / null), not a nullable boolean.
    Null must NOT route to Unknown — 87.7% of the corpus is null, and under
    AC-6.6's pass-and-tag policy that would make the work-setting filter a no-op.
    """
    setting, inferred = derive_work_setting(None, "Data Engineer", "Onsite in our KC office")
    assert setting == "On-site" and inferred is True


@pytest.mark.parametrize("text", ["This is a hybrid role", "Hybrid schedule, 3 days onsite"])
def test_ac_1_5_hybrid_detected(text):
    assert derive_work_setting(None, "Data Engineer", text)[0] == "Hybrid"


def test_ac_1_5_hybrid_needs_word_boundary():
    """'hybridization' must not make a job Hybrid."""
    assert derive_work_setting(None, "Research Scientist",
                               "Experience with hybridization assays")[0] == "On-site"


def test_ac_1_5_inferred_flag_set_only_when_derived():
    assert derive_work_setting(1.0, "X", "y")[1] is False       # read from the flag
    assert derive_work_setting(None, "X", "hybrid role")[1] is True   # inferred
    assert derive_work_setting(None, "X", "y")[1] is True             # inferred


# ---------------------------------------------------------------- AC-1.6a

@pytest.mark.parametrize("raw,expected", [
    ("Full-time", "Full-time"), ("Contract", "Contract"), ("Part-time", "Part-time"),
    ("Temporary", "Temporary"), ("Internship", "Internship"),
])
def test_ac_1_6a_retained_types(raw, expected):
    assert normalize_employment_type(raw) == expected


@pytest.mark.parametrize("raw", ["Volunteer", "Other", None, ""])
def test_ac_1_6a_excluded_types(raw):
    """AC-1.6a: Volunteer and Other are excluded from the corpus entirely."""
    assert normalize_employment_type(raw) is None


# ---------------------------------------------------------------- AC-1.6

def test_ac_1_6_regex_takes_precedence_over_label():
    years, src = parse_min_years("Requires 5+ years of experience", "Entry level")
    assert (years, src) == (5.0, "description_regex")


@pytest.mark.parametrize("label,years", list(SENIORITY_YEARS.items()))
def test_ac_1_6_seniority_fallback(label, years):
    assert parse_min_years("No numeric requirement stated here", label) == (years, "seniority_label")


def test_ac_1_6_no_signal_is_none():
    assert parse_min_years("No requirement stated", None) == (None, "none")


def test_ac_1_6_implausible_values_ignored():
    """'100 years of combined leadership' is not an experience requirement."""
    assert parse_min_years("100 years of combined experience", None) == (None, "none")


def test_ac_1_6_first_match_wins():
    """Deterministic rule: the first stated figure, which is the requirement line."""
    assert parse_min_years("3 years required; 10 years preferred", None)[0] == 3.0


# ---------------------------------------------------------------- AC-1.7

@pytest.mark.parametrize("text,level", [
    ("High school diploma required", 1), ("GED or equivalent", 1),
    ("Associate degree in IT", 2), ("Bachelor's degree in CS", 3),
    ("BS/BA in a technical field", 3), ("Master's degree preferred", 4),
    ("MBA required", 4), ("PhD in Statistics", 5), ("Doctorate required", 5),
])
def test_ac_1_7_degree_keywords(text, level):
    assert parse_education(text) == level


def test_ac_1_7_lowest_wins():
    """
    AC-1.7: when several degrees appear, take the LOWEST — "Bachelor's required,
    Master's preferred" requires a bachelor's.
    """
    assert parse_education("Bachelor's degree required, Master's preferred") == 3
    assert parse_education("PhD or Master's or Bachelor's") == 3


def test_ac_1_7_no_match_is_none():
    assert parse_education("Five years of pipeline experience") is None


@pytest.mark.parametrize("text", [
    "Proficiency with MS Office and Excel",
    "Experience with MS Word",
    "Familiarity with BS 7799 security standards",
])
def test_ac_1_7_abbreviation_false_positives(text):
    """
    'MS Office' must not read as a Master's degree. Bare BS/MS/BA/MA are never
    matched — only explicit degree phrasings.
    """
    assert parse_education(text) is None


def test_ac_1_7_ladder_is_ordinal():
    assert EDUCATION_LEVELS["high school"] < EDUCATION_LEVELS["bachelor"] < EDUCATION_LEVELS["phd"]


@pytest.mark.parametrize("text,expected", [
    ("4-7 years related business experience", 4.0),
    ("3-5+ years of experience", 3.0),
    ("2 - 4 years in a similar role", 2.0),
    ("5–7 years experience", 5.0),          # en dash
    ("7+ years of experience", 7.0),
    ("10 years of increasingly responsible experience", 10.0),
    ("1+ years' experience in one or more", 1.0),
])
def test_ac_1_6_v1_3_range_takes_lower_bound(text, expected):
    """
    AC-1.6 v1.3: min_years_exp is a MINIMUM, so a range yields its lower bound.
    The original regex matched the upper bound because "4-" is not followed by
    "years", overstating every ranged requirement.
    """
    assert parse_min_years(text, None)[0] == expected


# ============================================================ PHASE 1c
# AC-1.8 coverage stats · AC-1.9 idempotency · AC-1.10 engine boundary

import json  # noqa: E402

import pandas as pd  # noqa: E402

from src.ingest import CORPUS_SCHEMA, build_corpus, normalize_fields  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    return build_corpus(FIXTURES / "postings_sample.csv",
                        FIXTURES / "job_skills_sample.csv",
                        out_dir=out), out


# ---------------------------------------------------------------- AC-1.8

def test_ac_1_8_stats_file_written(built):
    _, out = built
    assert (out / "coverage_stats.json").exists()


def test_ac_1_8_reports_required_coverage_rates(built):
    """AC-1.8: non-null rates for the five named fields."""
    (_, stats), _ = built
    for field in ["salary_min", "education_required", "min_years_exp",
                  "work_setting_known", "skills_desc"]:
        assert field in stats["coverage_pct"], f"{field} missing from coverage_pct"
        assert 0.0 <= stats["coverage_pct"][field] <= 100.0


def test_ac_1_8_reports_funnel_counts(built):
    """AC-1.8: row counts after each named stage."""
    (_, stats), _ = built
    f = stats["funnel"]
    assert f["raw_load"] == 21
    assert f["raw_load"] >= f["after_tech_scoping"] >= f["after_dedupe"] >= f["after_employment_filter"]
    assert "after_zero_skill_drop" in f, "AC-2.6's stage must be present even before REQ-2 fills it"
    assert f["after_zero_skill_drop"] is None, "unfilled until REQ-2 exists — not silently faked"


def test_ac_1_8_reports_salary_path_counts(built):
    """AC-1.4 amendment: how many rows took each of the three salary paths."""
    (_, stats), _ = built
    paths = stats["salary_source"]
    assert set(paths) <= {"normalized_salary", "derived", "non_usd", "none"}
    assert sum(paths.values()) == stats["funnel"]["after_employment_filter"]


def test_ac_1_8_non_usd_is_counted_not_hidden(built):
    """The fixture contains one EUR posting; it must appear as its own path."""
    (_, stats), _ = built
    assert stats["salary_source"].get("non_usd", 0) == 1


# ---------------------------------------------------------------- AC-1.9

def test_ac_1_9_idempotent(tmp_path):
    """AC-1.9: two runs over the same inputs produce identical output."""
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    (df_a, stats_a) = build_corpus(FIXTURES / "postings_sample.csv",
                                   FIXTURES / "job_skills_sample.csv", out_dir=a_dir)
    (df_b, stats_b) = build_corpus(FIXTURES / "postings_sample.csv",
                                   FIXTURES / "job_skills_sample.csv", out_dir=b_dir)
    assert len(df_a) == len(df_b)
    assert list(df_a["job_id"]) == list(df_b["job_id"])
    assert stats_a == stats_b
    assert json.loads((a_dir / "coverage_stats.json").read_text()) == \
           json.loads((b_dir / "coverage_stats.json").read_text())


def test_ac_1_9_parquet_roundtrips(built):
    """The written Parquet reloads to the same rows the pipeline produced."""
    (df, _), out = built
    reloaded = pd.read_parquet(out / "jobs_tech.parquet")
    assert len(reloaded) == len(df)
    assert list(reloaded["job_id"]) == list(df["job_id"])


def test_ac_1_9_schema_columns_present(built):
    """Output conforms to REQ-1's normalized schema for the fields phase 1 owns."""
    (df, _), _ = built
    for col in CORPUS_SCHEMA:
        assert col in df.columns, f"schema column {col} missing"


def test_ac_1_9_no_excluded_employment_types(built):
    """AC-1.6a: Volunteer (job 9) and Other (job 19) never reach the corpus."""
    (df, _), _ = built
    assert set(df["employment_type"]) <= {"Full-time", "Contract", "Part-time",
                                          "Temporary", "Internship"}


# ---------------------------------------------------------------- AC-1.10

def test_ac_1_10_per_row_logic_needs_no_database():
    """
    AC-1.10: the engine boundary. Every per-row rule is a pure function callable
    with no DuckDB connection, which is what keeps the engine choice (D9)
    reversible.
    """
    assert normalize_salary(90000.0, None, None, "YEARLY", "USD")[2] is True
    assert derive_work_setting(1.0, "t", "d")[0] == "Remote"
    assert normalize_employment_type("Full-time") == "Full-time"
    assert parse_min_years("5+ years", None)[0] == 5.0
    assert parse_education("Bachelor's degree") == 3
    assert normalize_title("Senior Data Engineer") == "data engineer"


def test_ac_1_10_normalize_fields_is_dataframe_only():
    """normalize_fields operates on a DataFrame — no connection, no SQL."""
    df = pd.DataFrame([{
        "title": "Data Engineer", "description": "3 years. Bachelor's degree.",
        "formatted_work_type": "Full-time", "formatted_experience_level": "Associate",
        "remote_allowed": None, "normalized_salary": 100000.0, "min_salary": None,
        "max_salary": None, "pay_period": "YEARLY", "currency": "USD",
    }])
    out, stats = normalize_fields(df)
    assert len(out) == 1
    assert out.iloc[0]["salary_listed"] and out.iloc[0]["education_required"] == 3


@pytest.mark.parametrize("smin,smax", [
    (float("nan"), float("nan")), (float("nan"), 120000.0), (100000.0, float("nan")),
])
def test_ac_1_4_nan_range_falls_back_to_normalized(smin, smax):
    """
    Regression: pandas yields NaN, not None, for a missing numeric cell, and
    `NaN is not None` is True. The original guard let NaN into the range branch
    and returned NaN * multiplier — 999 real rows came out salary_listed=True
    with a null salary, which AC-6.5's `salary_max >= min_salary` filter would
    then read as underpaying and drop.
    """
    lo, hi, listed, src = normalize_salary(95000.0, smin, smax, "YEARLY", "USD")
    assert (lo, hi, listed, src) == (95000.0, 95000.0, True, "normalized_salary")


def test_ac_1_4_nan_with_no_normalized_is_unlisted():
    assert normalize_salary(float("nan"), float("nan"), float("nan"), "YEARLY", "USD") \
        == (None, None, False, "none")


def test_ac_1_4_listed_implies_a_value():
    """Invariant: salary_listed=True must never come with a null salary."""
    cases = [(95000.0, float("nan"), float("nan"), "YEARLY", "USD"),
             (95000.0, 90000.0, 100000.0, "YEARLY", "USD"),
             (None, 40.0, 60.0, "HOURLY", "USD"),
             (float("nan"), None, None, None, None)]
    for c in cases:
        lo, hi, listed, _ = normalize_salary(*c)
        assert listed == (lo is not None and hi is not None)
