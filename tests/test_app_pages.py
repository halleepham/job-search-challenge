"""
Runs the Streamlit pages with AppTest so a runtime exception fails CI.

An HTTP 200 from Streamlit proves only that the server started - the page script
executes over a websocket afterwards, so a crash still returns 200. These tests
execute the scripts.

Skipped when the corpus is unbuilt, so a fresh clone still runs the unit suite.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not Path("data/processed/jobs_tech.parquet").exists()
    or not Path("data/index/manifest.json").exists(),
    reason="corpus/index not built",
)

TIMEOUT = 120


def _run(path: str, **session_state) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    for key, value in session_state.items():
        at.session_state[key] = value
    return at.run()


def test_home_renders():
    at = _run("app.py")
    assert not at.exception, at.exception
    assert at.title[0].value == "Job Search & Matching"


def test_profile_page_renders():
    at = _run("pages/1_Profile.py")
    assert not at.exception, at.exception


def test_ac_3_1_profile_page_shows_no_results():
    """AC-3.1: the profile page is standalone — no scores or job cards on it."""
    at = _run("pages/1_Profile.py")
    text = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "Why this score" not in text


def test_ac_3_2_every_specified_control_exists():
    """AC-3.2: the widgets REQ-3 §page structure specifies are all present."""
    at = _run("pages/1_Profile.py")
    assert at.text_area, "career goals / résumé paste"
    assert at.multiselect, "skills and preferred titles"
    assert at.number_input, "years of experience and minimum salary"
    assert at.selectbox, "education ladders, location, preset"
    assert at.slider, "maximum distance"
    assert len(at.checkbox) >= 9, "3 work settings + 5 employment types + include-unlisted"


def test_ac_3_3_work_and_employment_are_checkbox_sets():
    at = _run("pages/1_Profile.py")
    labels = {c.label for c in at.checkbox}
    assert {"Remote", "Hybrid", "On-site"} <= labels
    assert {"Full-time", "Contract", "Part-time", "Temporary", "Internship"} <= labels


def test_ac_3_4_validation_blocks_an_empty_form():
    """With no résumé and no goals the submit button must be disabled."""
    at = _run("pages/1_Profile.py")
    assert at.warning, "expected validation warnings on an empty custom profile"
    assert at.button[0].disabled


def test_analytics_page_renders():
    at = _run("pages/3_Analytics.py")
    assert not at.exception, at.exception


def test_results_page_without_profile_is_graceful():
    at = _run("pages/2_Results.py")
    assert not at.exception, at.exception
    assert at.info, "expected a prompt to build a profile first"


def test_results_page_renders_a_real_search():
    from src.profiles import PRESETS

    at = _run("pages/2_Results.py", profile=PRESETS["data_science_student"])
    assert not at.exception, at.exception
    assert at.metric, "expected the AC-11.2 funnel metrics"


def test_ac_11_3_breakdown_sums_to_the_displayed_score():
    """
    AC-11.3 end to end: the points shown in the expander must total the score
    shown on the card. This is the property that makes the number explainable.
    """
    from src.personal_kb import PersonalKB
    from src.pipeline import search
    from src.profiles import PRESETS
    import pandas as pd
    from src.indexing import JobIndex

    jobs = pd.read_parquet("data/processed/jobs_tech.parquet")
    index = JobIndex.load_or_build(jobs)
    p = PRESETS["data_science_student"]
    res = search(jobs, index, p, PersonalKB.build(p.resume_text, p.career_goals))
    for r in res.results:
        shown = sum(round(c["weighted"] * 100, 1) for c in r["components"])
        assert abs(shown - r["score"]) < 0.5, f"table sums to {shown}, card says {r['score']}"
