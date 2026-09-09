"""REQ-3: profile input page. Its own page — nothing else renders here (AC-3.1)."""

from __future__ import annotations

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

st.set_page_config(page_title="Profile", page_icon="👤", layout="wide")
st.title("Your profile")

vocab = load_vocabulary()
SKILL_OPTIONS = sorted(vocab.canonical)
TITLE_OPTIONS = sorted({
    "data engineer", "data scientist", "data analyst", "software engineer",
    "backend engineer", "frontend engineer", "full stack developer", "devops engineer",
    "platform engineer", "machine learning engineer", "security analyst",
    "database administrator", "qa engineer", "cloud engineer", "business analyst",
})


@st.cache_data
def city_options() -> list[str]:
    """AC-3.5: locations are validated against the geocoding table, not free text."""
    import geonamescache

    cities = geonamescache.GeonamesCache().get_cities().values()
    return [""] + sorted(
        {f"{c['name']}, {c['admin1code']}" for c in cities
         if c["countrycode"] == "US" and c["population"] >= 50_000}
    )


# --- Section D (first, so it can populate everything below) -----------------
preset_key = st.selectbox(
    "Load a preset profile", ["Custom"] + list(PRESETS),
    format_func=lambda k: "Custom" if k == "Custom" else PRESETS[k].name,
    help="AC-3.7 — presets also drive the evaluation notebook's per-profile comparison.",
)
preset = PRESETS.get(preset_key)

st.divider()

# --- Section A: career documents --------------------------------------------
st.subheader("A · Career documents")
st.caption("These become the evidence shown next to each match — quoted verbatim, never generated.")

paste_mode = st.toggle("Paste résumé text instead of uploading", value=preset is not None)
resume_text = ""
if paste_mode:
    resume_text = st.text_area("Résumé text", value=preset.resume_text if preset else "",
                               height=200, placeholder="Paste your résumé…")
else:
    uploaded = st.file_uploader("Résumé", type=["pdf", "txt", "md"])
    if uploaded:
        if uploaded.name.lower().endswith(".pdf"):
            from pypdf import PdfReader

            resume_text = "\n\n".join(p.extract_text() or "" for p in PdfReader(uploaded).pages)
        else:
            resume_text = uploaded.read().decode("utf-8", errors="ignore")
        st.success(f"Read {len(resume_text):,} characters from {uploaded.name}")

extra_documents = st.text_area(
    "Additional documents (optional)", height=90,
    placeholder="Projects, coursework, certifications…")

career_goals = st.text_area(
    "Career goals **(required)**", value=preset.career_goals if preset else "", height=90,
    placeholder="What kind of work do you want, and why?")

st.divider()

# --- Section B: qualifications ----------------------------------------------
st.subheader("B · Qualifications")
c1, c2 = st.columns([2, 1])
with c1:
    skills = set(st.multiselect(
        "Skills", SKILL_OPTIONS,
        default=sorted((preset.skills & set(SKILL_OPTIONS)) if preset else []),
        help="Options come from the extraction vocabulary, so they match what we parse from postings."))
    other = st.text_input("Other skills (comma-separated)", placeholder="e.g. cobol, sas")
    skills |= {s.strip().lower() for s in other.split(",") if s.strip()}
with c2:
    years = st.number_input("Years of experience", 0.0, 50.0,
                            float(preset.years_experience) if preset else 0.0, 0.5)

c3, c4 = st.columns(2)
completed = c3.selectbox("Highest completed education",
                         ["High School", "Associate", "Bachelor's", "Master's", "PhD"],
                         index=2)
pursuing = c4.selectbox("Currently pursuing",
                        [NOT_ENROLLED, "Associate", "Bachelor's", "Master's", "PhD"],
                        index=3 if (preset and preset.education_in_progress == 4) else 0,
                        help="Counts as a half-step below the degree — so a student mid-master's "
                             "is not eliminated from master's-required roles.")

st.divider()

# --- Section C: job preferences ---------------------------------------------
st.subheader("C · Job preferences")
st.caption("These are **hard filters**: a job failing one is removed, not merely down-ranked.")

titles = st.multiselect("Preferred job titles", TITLE_OPTIONS,
                        default=[t for t in (preset.preferred_titles if preset else [])
                                 if t in TITLE_OPTIONS])

cities = city_options()
default_city = preset.preferred_location if preset and preset.preferred_location in cities else ""
location = st.selectbox("Preferred location", cities, index=cities.index(default_city))

c5, c6 = st.columns(2)
with c5:
    st.markdown("**Work setting** — check all you would accept")
    work_settings = {s for s in WORK_SETTINGS
                     if st.checkbox(s, value=(s in preset.accepted_work_settings) if preset else True,
                                    key=f"ws_{s}")}
with c6:
    st.markdown("**Employment type** — check all you would accept")
    employment_types = {t for t in EMPLOYMENT_TYPES
                        if st.checkbox(t, value=(t in preset.accepted_employment_types)
                                       if preset else True, key=f"et_{t}")}

c7, c8 = st.columns(2)
min_salary = c7.number_input("Minimum salary (USD)", 0, 500_000,
                            int(preset.min_salary) if preset else 0, 5_000)
include_unlisted = c8.checkbox(
    "Include jobs with no listed salary", value=True,
    help="Only 30% of postings list a salary. Unchecking this removes most of the corpus.")

distance_enabled = bool(location) and bool({"Hybrid", "On-site"} & work_settings)
max_distance = st.slider("Maximum distance (miles)", 10, 250,
                         int(preset.max_distance_miles) if preset else 100,
                         disabled=not distance_enabled,
                         help="Applies to on-site and hybrid roles. Remote jobs ignore distance.")

st.divider()

# --- Submit ------------------------------------------------------------------
errors = validate_profile(resume_text, career_goals, skills, work_settings,
                          employment_types, location)
if errors:
    for e in errors:
        st.warning(e)

if st.button("Find matches", type="primary", disabled=bool(errors), use_container_width=True):
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
        max_distance_miles=max_distance,
    )
    st.switch_page("pages/2_Results.py")
