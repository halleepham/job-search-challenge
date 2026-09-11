"""
Job search application — entry page.

This is the search form: the user describes themselves and their constraints,
and lands on results. There is no separate landing page explaining the method -
this is an application someone uses, not a demonstration of how it works.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.profiles import PRESETS, UserProfile
from src.skills import load_vocabulary
from src.validation import (
    EMPLOYMENT_TYPES,
    NOT_ENROLLED,
    WORK_SETTINGS,
    education_ordinal,
    validate_profile,
)

st.set_page_config(page_title="Job Search", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")

CORPUS = Path("data/processed/jobs_tech.parquet")
if not CORPUS.exists():
    st.error("Corpus not built. Run:\n\n```\npython -m src.download_data\npython -m src.ingest\n```")
    st.stop()

vocab = load_vocabulary()
SKILL_OPTIONS = sorted(vocab.canonical)
TITLE_OPTIONS = sorted({
    "data engineer", "data scientist", "data analyst", "software engineer",
    "backend engineer", "frontend engineer", "full stack developer", "devops engineer",
    "platform engineer", "machine learning engineer", "security analyst",
    "database administrator", "qa engineer", "cloud engineer", "business analyst",
})
EDU_LEVELS = ["High School", "Associate", "Bachelor's", "Master's", "PhD"]


@st.cache_data
def city_options() -> list[str]:
    """AC-3.5: locations validated against the geocoding table, not free text."""
    import geonamescache

    cities = geonamescache.GeonamesCache().get_cities().values()
    return [""] + sorted({f"{c['name']}, {c['admin1code']}" for c in cities
                          if c["countrycode"] == "US" and c["population"] >= 50_000})


st.title("Find your next job")
st.caption("Tell us about yourself and what you're looking for. "
           "Every match comes with the evidence behind it.")

preset_key = st.selectbox(
    "Start from an example profile", ["Custom"] + list(PRESETS),
    format_func=lambda k: "Custom — fill it in myself" if k == "Custom" else PRESETS[k].name)
preset = PRESETS.get(preset_key)

st.divider()
about, want = st.columns(2, gap="large")

with about:
    st.subheader("About you")

    paste = st.toggle("Paste résumé text instead of uploading", value=preset is not None)
    resume_text = ""
    if paste:
        resume_text = st.text_area("Résumé", value=preset.resume_text if preset else "",
                                   height=170, placeholder="Paste your résumé…",
                                   label_visibility="collapsed")
    else:
        uploaded = st.file_uploader("Résumé (PDF, TXT or MD)", type=["pdf", "txt", "md"])
        if uploaded:
            if uploaded.name.lower().endswith(".pdf"):
                from pypdf import PdfReader

                resume_text = "\n\n".join(p.extract_text() or ""
                                          for p in PdfReader(uploaded).pages)
            else:
                resume_text = uploaded.read().decode("utf-8", errors="ignore")
            st.success(f"Read {len(resume_text):,} characters")

    career_goals = st.text_area(
        "What kind of work are you looking for?",
        value=preset.career_goals if preset else "", height=80,
        placeholder="e.g. a data engineering role building pipelines that feed dashboards")

    skills = set(st.multiselect(
        "Your skills", SKILL_OPTIONS,
        default=sorted((preset.skills & set(SKILL_OPTIONS)) if preset else [])))
    other = st.text_input("Other skills (comma-separated)", placeholder="e.g. cobol, sas")
    skills |= {s.strip().lower() for s in other.split(",") if s.strip()}

    y, e1, e2 = st.columns([1, 1.3, 1.3])
    years = y.number_input("Years' experience", 0.0, 50.0,
                           float(preset.years_experience) if preset else 0.0, 0.5)
    completed = e1.selectbox("Education", EDU_LEVELS, index=2)
    pursuing = e2.selectbox("Studying for", [NOT_ENROLLED] + EDU_LEVELS[1:],
                            index=3 if (preset and preset.education_in_progress == 4) else 0)

    with st.expander("Additional documents (optional)"):
        extra_documents = st.text_area("Projects, coursework, certifications", height=90,
                                       label_visibility="collapsed")

with want:
    st.subheader("What you want")

    titles = st.multiselect("Job titles", TITLE_OPTIONS,
                            default=[t for t in (preset.preferred_titles if preset else [])
                                     if t in TITLE_OPTIONS])
    cities = city_options()
    default_city = preset.preferred_location if preset and preset.preferred_location in cities else ""
    location = st.selectbox("Location", cities, index=cities.index(default_city))

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Work setting**")
        work_settings = {s for s in WORK_SETTINGS
                         if st.checkbox(s, value=(s in preset.accepted_work_settings)
                                        if preset else True, key=f"ws_{s}")}
    with w2:
        st.markdown("**Employment type**")
        employment_types = {t for t in EMPLOYMENT_TYPES
                            if st.checkbox(t, value=(t in preset.accepted_employment_types)
                                           if preset else True, key=f"et_{t}")}

    s1, s2 = st.columns(2)
    min_salary = s1.number_input("Minimum salary (USD)", 0, 500_000,
                                 int(preset.min_salary) if preset else 0, 5_000)
    include_unlisted = s2.checkbox("Include jobs with no listed salary", value=True,
                                   help="Only 30% of postings list a salary.")

    physical = bool({"Hybrid", "On-site"} & work_settings)
    max_distance = st.slider("Maximum commute (miles)", 10, 250,
                             int(preset.max_distance_miles) if preset else 100,
                             disabled=not (location and physical),
                             help="Applies to on-site and hybrid roles. Remote ignores distance.")

st.divider()
errors = validate_profile(resume_text, career_goals, skills, work_settings,
                          employment_types, location)
for message in errors:
    st.warning(message)

if st.button("Search jobs", type="primary", disabled=bool(errors), use_container_width=True):
    st.session_state["profile"] = UserProfile(
        name=preset.name if preset else "You",
        career_goals=career_goals,
        resume_text=resume_text + ("\n\n" + extra_documents if extra_documents else ""),
        skills=skills, years_experience=years,
        highest_completed_education=education_ordinal(completed) or 3,
        education_in_progress=education_ordinal(pursuing),
        preferred_titles=titles, preferred_location=location or None,
        accepted_work_settings=work_settings, accepted_employment_types=employment_types,
        min_salary=min_salary, include_unlisted_salary=include_unlisted,
        max_distance_miles=max_distance)
    st.session_state.pop("selected_job", None)
    st.switch_page("pages/1_Results.py")
