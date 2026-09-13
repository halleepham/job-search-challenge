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

TIMEOUT = 300


def _click(at, label):
    """
    Click by index, not by key. A button created on a column
    (`col.button(...)`) gets an auto-generated key that does not round-trip
    through `at.button(key=...)` — the click registers but the branch never
    fires, which looks exactly like an application bug.
    """
    index = next(i for i, b in enumerate(at.button) if b.label == label)
    return at.button[index].click().run()


def _run(path: str, **session_state) -> AppTest:
    at = AppTest.from_file(path, default_timeout=TIMEOUT)
    for key, value in session_state.items():
        at.session_state[key] = value
    return at.run()


def test_profile_page_renders():
    at = _run("views/profile.py")
    assert not at.exception, at.exception
    assert at.title[0].value == "Profile"


def test_ac_3_1_search_page_shows_no_results():
    """AC-3.1: the search page is standalone — no scores or job cards on it."""
    at = _run("views/profile.py")
    text = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "Why this match" not in text and "Match" not in text


def test_ac_3_2_every_specified_control_exists():
    """AC-3.2: the widgets REQ-3 §page structure specifies are all present."""
    at = _run("views/profile.py")
    assert at.text_area, "career goals / résumé paste"
    assert at.multiselect, "skills and preferred titles"
    assert at.number_input, "years of experience and minimum salary"
    assert at.selectbox, "education ladders, location, preset"
    assert at.slider, "maximum distance"
    assert len(at.checkbox) >= 9, "3 work settings + 5 employment types + include-unlisted"


def test_ac_3_3_work_and_employment_are_checkbox_sets():
    at = _run("views/profile.py")
    labels = {c.label for c in at.checkbox}
    assert {"Remote", "Hybrid", "On-site"} <= labels
    assert {"Full-time", "Contract", "Part-time", "Temporary", "Internship"} <= labels


def test_validation_is_silent_until_submit():
    """
    Warnings about fields you have not reached yet are noise. Nothing is
    reported until the user actually presses Search.
    """
    at = _run("views/profile.py")
    assert not at.warning, "form warned before the user submitted anything"


def test_validation_appears_after_submit():
    at = _run("views/profile.py")
    _click(at, "Search jobs")
    assert at.warning, "expected validation messages once Search was pressed"


def test_job_titles_come_from_the_corpus_not_a_hardcoded_list():
    """
    The original list was 15 titles written by hand, so "data science engineer"
    simply did not exist. Options now come from the corpus, and free text is
    accepted alongside for anything it lacks.
    """
    from views.profile import title_choices

    choices = title_choices()
    assert len(choices) > 100, "expected corpus-derived titles, not a short hand-written list"
    assert any(c[0].isupper() for c in choices), "titles must display capitalised"


def test_form_controls_display_capitalised():
    at = _run("views/profile.py")
    skills = next(m for m in at.multiselect if m.label == "Your skills")
    assert any(o[0].isupper() for o in skills.options), "skills must display capitalised"


def test_insights_page_renders():
    at = _run("views/insights.py")
    assert not at.exception, at.exception


def test_results_page_without_profile_is_graceful():
    at = _run("views/results.py")
    assert not at.exception, at.exception
    assert at.info, "expected a prompt to build a profile first"


def test_results_page_renders_a_real_search():
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
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
    from src.explain import component_bars

    res = search(jobs, index, p, PersonalKB.build(p.resume_text, p.career_goals))
    for r in res.results:
        # Assert on what the page renders, not on a second implementation of the
        # same arithmetic - that is how the 78.5-vs-79 drift stayed invisible.
        shown = component_bars(r)["Earned"].sum()
        assert abs(shown - r["score"]) < 0.051, \
            f"table sums to {shown}, card says {r['score']}"


def test_ac_11_8_to_11_13_panels_render():
    """REQ-11 v1.1: the explainable-match panels the course guidance specifies."""
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    assert not at.exception, at.exception

    text = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "**This job**" in text and "**You**" in text, "AC-11.8 job/candidate comparison"
    assert any(w in text for w in ("candidate.", "candidate,")), "AC-11.13 verdict"

    # AC-11.9 / AC-11.11 now live in per-job sub-tabs rather than inline headings.
    tab_labels = {label for group in at.tabs for label in [getattr(group, "label", "")]}
    assert any("Evidence" in t for t in tab_labels), "AC-11.9 evidence tab"
    assert any("Gaps & unknowns" in t for t in tab_labels), "AC-11.11 gaps tab"
    # AppTest does not expose st.progress, so assert on the caption the bars emit —
    # which also proves AC-11.3, that the points shown total the displayed score.
    captions = " ".join(c.value for c in at.caption if isinstance(c.value, str))
    assert "Points total" in captions, "AC-11.10 component bars"
    assert any("Your decision" == r.label for r in at.radio), "AC-11.12 decision control"


def test_ac_11_13_v1_2_no_methodology_takeaway():
    """
    AC-11.13 v1.2: the app does not explain itself to its own user. The
    per-result verdict stays (it is about the job); the page-level essay on what
    an explainable match is *for* belongs in the report.
    """
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    everything = " ".join(
        str(x.value) for group in (at.caption, at.markdown, at.info) for x in group
        if isinstance(getattr(x, "value", None), str))
    assert "Takeaway" not in everything


def test_results_is_a_list_plus_detail():
    """
    The job-board pattern: a scannable list of results, and a detail pane for the
    one you clicked. Every listed job needs a way to open it.
    """
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    view_buttons = [b for b in at.button if b.label == "View details"]
    assert view_buttons, "list rows must be openable"
    labels = [getattr(t, "label", "") for t in at.tabs]
    assert "Job description" in labels, "detail pane leads with the posting itself"
    assert "Why this match" in labels, "…and explains the score behind a click"


def test_clicking_a_job_changes_the_detail_pane():
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    first = at.session_state["selected_job"]
    _click(at, "View details")
    assert at.session_state["selected_job"] != first


def test_ac_11_12_weight_sliders_present():
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    from src.scoring import WEIGHTS
    assert len(at.slider) == len(WEIGHTS), "one slider per score component"


def test_my_jobs_empty_state():
    at = _run("views/my_jobs.py")
    assert not at.exception, at.exception
    assert at.info, "expected a prompt when nothing is filed"


def test_my_jobs_lists_filed_decisions():
    at = _run("views/my_jobs.py", decisions={
        101: {"status": "Applied", "title": "Data Engineer", "company": "Acme",
              "location": "Kansas City, MO", "score": 74, "tier": "Good"},
        102: {"status": "Saved for later", "title": "Data Analyst", "company": "Beta",
              "location": "Remote", "score": 61, "tier": "Good"}})
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "Data Engineer" in text and "Data Analyst" in text


def test_decision_is_not_saved_until_confirmed():
    """
    Selecting a radio must not file the job — a stray click should not silently
    change what the user has decided.
    """
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    radio = next(r for r in at.radio if r.label == "Your decision")
    at.radio(key=radio.key).set_value("Applied").run()
    # `decisions` exists (setdefault) but must still be empty: selecting is not filing.
    assert dict(at.session_state["decisions"]) == {}, "radio alone filed the job"

    assert any(b.label == "Save" for b in at.button), "expected an explicit Save control"
    _click(at, "Save")
    filed = dict(at.session_state["decisions"])
    assert filed, "Save did not file the decision"
    assert next(iter(filed.values()))["status"] == "Applied"


def test_ac_3_11_profile_survives_navigation():
    """
    AC-3.11: the form is keyed, so returning to it shows what was submitted.
    Re-entering a résumé and twelve fields to change one salary figure is not an
    acceptable cost for adjusting a search.
    """
    at = _run("views/profile.py")
    at.text_area(key="pf_goals").set_value("I want data engineering work.").run()
    assert at.session_state["pf_goals"] == "I want data engineering work."

    again = AppTest.from_file("views/profile.py", default_timeout=TIMEOUT)
    for key, value in at.session_state.filtered_state.items():
        again.session_state[key] = value
    again.run()
    assert again.text_area(key="pf_goals").value == "I want data engineering work."


def test_ac_3_11_clear_form_resets():
    at = _run("views/profile.py")
    at.text_area(key="pf_goals").set_value("something").run()
    _click(at, "Clear form")
    assert at.session_state["pf_goals"] == ""


def test_ac_3_11_profile_can_be_saved_by_name():
    at = _run("views/profile.py")
    at.text_input(key="pf_save_name").set_value("Data roles KC").run()
    _click(at, "Save profile")
    assert "Data roles KC" in dict(at.session_state["saved_profiles"])


def test_ac_11_14_posting_link_and_expiry_disclosed():
    """
    AC-11.14: every posting in this historical corpus closed in 2024. Linking
    without saying so would imply an application is still possible.
    """
    from src.profiles import PRESETS

    at = _run("views/results.py", profile=PRESETS["data_science_student"])
    assert not at.exception, at.exception
    warnings = " ".join(w.value for w in at.warning if isinstance(w.value, str))
    assert "closed on" in warnings, "expiry must be disclosed beside the link"
    assert "historical" in warnings, "the corpus being historical must be stated, not implied"

    # AppTest does not expose st.link_button, so assert the data the link is
    # built from — a link rendered from a null URL would be the real failure.
    import pandas as pd

    jobs = pd.read_parquet("data/processed/jobs_tech.parquet",
                           columns=["posting_url", "expiry_date"])
    assert jobs["posting_url"].notna().all(), "every result must have a source link"
    assert jobs["expiry_date"].notna().all(), "every result must disclose when it closed"


def test_ac_3_11_v1_1_form_restores_after_navigation():
    """
    Streamlit drops widget state for controls not rendered on the current run, so
    keyed widgets alone did not survive a trip to Matches and back — the form came
    back empty. It is restored from the submitted profile instead.
    """
    from src.profiles import PRESETS

    at = _run("views/profile.py", profile=PRESETS["data_science_student"])
    assert not at.exception, at.exception
    assert at.text_area(key="pf_goals").value == PRESETS["data_science_student"].career_goals
    assert at.session_state["pf_salary"] == PRESETS["data_science_student"].min_salary


def test_ac_3_12_loaded_profile_offers_update_not_only_save_as_new():
    from src.profiles import PRESETS

    at = _run("views/profile.py",
              saved_profiles={"Data roles KC": PRESETS["data_science_student"]},
              loaded_name="Data roles KC")
    assert not at.exception, at.exception
    assert any("Update" in b.label for b in at.button), "expected an update-in-place action"
    assert any("Data roles KC" in b.label for b in at.button), "the button must name the profile"


def test_ac_3_12_update_overwrites_rather_than_duplicating():
    from dataclasses import replace

    from src.profiles import PRESETS

    original = PRESETS["data_science_student"]
    at = _run("views/profile.py",
              saved_profiles={"Data roles KC": original},
              loaded_name="Data roles KC")
    at.text_area(key="pf_goals").set_value("Now I want platform engineering work.").run()
    _click(at, 'Update “Data roles KC”')
    saved = dict(at.session_state["saved_profiles"])
    assert list(saved) == ["Data roles KC"], "update must not create a second entry"
    assert saved["Data roles KC"].career_goals == "Now I want platform engineering work."


def test_ac_3_12_unsaved_profile_offers_plain_save():
    at = _run("views/profile.py")
    assert any(b.label == "Save profile" for b in at.button)
    assert not any("Update" in b.label for b in at.button)
