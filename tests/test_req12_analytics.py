"""AC-derived tests for REQ-12 (FROZEN v1.0) — dataset analytics."""

import pandas as pd
import pytest

from src.analytics import CORPUS_LABEL, compute, state_of, title_family

JOBS = pd.DataFrame([
    {"job_id": 1, "title": "Data Engineer", "company": "Acme", "location_raw": "Kansas City, MO",
     "work_setting": "On-site", "min_years_exp": 3.0, "salary_listed": True,
     "salary_min": 100000.0, "salary_max": 120000.0, "required_skills": ["python", "sql"]},
    {"job_id": 2, "title": "Data Engineer", "company": "Acme", "location_raw": "Austin, TX",
     "work_setting": "Remote", "min_years_exp": 5.0, "salary_listed": True,
     "salary_min": 130000.0, "salary_max": 150000.0, "required_skills": ["python", "spark"]},
    {"job_id": 3, "title": "Security Analyst", "company": "Beta", "location_raw": "Austin, TX",
     "work_setting": "Hybrid", "min_years_exp": None, "salary_listed": False,
     "salary_min": None, "salary_max": None, "required_skills": ["splunk"]},
])


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    return compute(JOBS, out_dir=tmp_path_factory.mktemp("analytics"))


# ---------------------------------------------------------------- AC-12.1

def test_ac_12_1_all_aggregations_present(tables):
    expected = {"top_titles", "top_skills", "geographic_distribution",
                "salary_by_title_family", "work_setting", "experience_distribution",
                "top_companies", "skill_cooccurrence"}
    assert expected <= set(tables)


def test_ac_12_5_top_titles_hand_computed(tables):
    """AC-12.5: verified against hand-computed expected values."""
    t = tables["top_titles"].set_index("title")["n"].to_dict()
    assert t == {"Data Engineer": 2, "Security Analyst": 1}


def test_ac_12_5_geographic_hand_computed(tables):
    g = tables["geographic_distribution"].set_index("state")["n"].to_dict()
    assert g == {"TX": 2, "MO": 1}


def test_ac_12_5_work_setting_hand_computed(tables):
    w = tables["work_setting"].set_index("work_setting")["n"].to_dict()
    assert w == {"On-site": 1, "Remote": 1, "Hybrid": 1}


def test_ac_12_5_top_skills_hand_computed(tables):
    s = tables["top_skills"].set_index("skill")["n"].to_dict()
    assert s["python"] == 2 and s["sql"] == 1 and s["splunk"] == 1


def test_ac_12_5_salary_excludes_unlisted(tables):
    """A posting with no salary must not drag the median toward zero."""
    row = tables["salary_by_title_family"].set_index("title_family").loc["Data Engineer"]
    assert row["n_jobs"] == 2 and row["n_with_salary"] == 2
    assert row["median_salary_max"] == pytest.approx(135000, abs=1)


def test_ac_12_5_experience_excludes_nulls(tables):
    assert tables["experience_distribution"]["n"].sum() == 2


# ---------------------------------------------------------------- AC-12.3

def test_ac_12_3_cooccurrence_pairs(tables):
    pairs = {(r.skill_a, r.skill_b): r.n for r in tables["skill_cooccurrence"].itertuples()}
    assert pairs.get(("python", "sql")) == 1
    assert pairs.get(("python", "spark")) == 1


def test_ac_12_3_single_skill_job_makes_no_pair(tables):
    """The Security Analyst posting has one skill and cannot contribute a pair."""
    assert ("splunk", "splunk") not in {
        (r.skill_a, r.skill_b) for r in tables["skill_cooccurrence"].itertuples()}


# ---------------------------------------------------------------- AC-12.4 / 12.2

def test_ac_12_4_csvs_written(tables, tmp_path_factory):
    out = tmp_path_factory.mktemp("csv_check")
    compute(JOBS, out_dir=out)
    for name in tables:
        assert (out / f"{name}.csv").exists()


def test_ac_12_2_corpus_label_is_honest():
    """
    AC-12.2 / D3: outputs must say what the corpus actually is. Calling it
    "tech jobs" would overstate a corpus that is ~25% strictly software/data.
    """
    assert "IT, engineering, analytics" in CORPUS_LABEL
    assert "tech jobs" not in CORPUS_LABEL.lower()


def test_ac_12_2_label_written_alongside(tmp_path):
    compute(JOBS, out_dir=tmp_path)
    assert CORPUS_LABEL in (tmp_path / "_corpus_label.txt").read_text()


# ---------------------------------------------------------------- helpers

@pytest.mark.parametrize("title,family", [
    ("Senior Data Engineer", "Data Engineer"),
    ("Machine Learning Scientist", "Data Scientist / ML"),
    ("Business Intelligence Analyst", "Data Analyst / BI"),
    ("Full Stack Developer", "Software Engineer"),
    ("Site Reliability Engineer", "DevOps / Cloud / SRE"),
    ("Cyber Threat Analyst", "Security"),
    ("QA Automation Engineer", "QA / Test"),
    ("Mechanical Engineer", "Other Engineering"),
    ("Registered Nurse", "Other"),
])
def test_title_family_first_match_wins(title, family):
    assert title_family(title) == family


@pytest.mark.parametrize("raw,state", [
    ("Kansas City, MO", "MO"), ("Austin, Texas", "TX"),
    ("United States", None), (None, None),
])
def test_state_extraction(raw, state):
    assert state_of(raw) == state
