"""Profile — the form describing the person and what they are looking for."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.display import display_map, pretty, to_canonical
from src.profiles import PRESETS, UserProfile
from src.skills import load_vocabulary
from src.validation import (
    EMPLOYMENT_TYPES,
    NOT_ENROLLED,
    WORK_SETTINGS,
    education_ordinal,
    validate_profile,
)

EDU_LEVELS = ["High School", "Associate", "Bachelor's", "Master's", "PhD"]


@st.cache_data
def skill_choices() -> dict[str, str]:
    return display_map(sorted(load_vocabulary().canonical))


@st.cache_data
def title_choices() -> dict[str, str]:
    """
    Job titles taken from the corpus itself — the 300 most common normalized
    titles — rather than a hand-written list. Free text is accepted alongside,
    so a title the corpus happens not to contain is never a dead end.
    """
    jobs = pd.read_parquet("data/processed/jobs_tech.parquet",
                           columns=["title_normalized"])
    common = jobs["title_normalized"].value_counts().head(300).index
    return display_map(sorted(t for t in common if t and len(t) > 3))


@st.cache_data
def city_choices() -> list[str]:
    import geonamescache

    cities = geonamescache.GeonamesCache().get_cities().values()
    return [""] + sorted({f"{c['name']}, {c['admin1code']}" for c in cities
                          if c["countrycode"] == "US" and c["population"] >= 50_000})


st.title("Profile")
st.caption("Tell us about yourself and what you're looking for. "
           "Every match comes with the evidence behind it.")

preset_key = st.selectbox(
    "Start from an example profile", ["Custom"] + list(PRESETS),
    format_func=lambda k: "Custom — fill it in myself" if k == "Custom" else PRESETS[k].name)
preset = PRESETS.get(preset_key)

skills_map, titles_map = skill_choices(), title_choices()
about, want = st.columns(2, gap="large")

# ── About you ───────────────────────────────────────────────────────────────
with about:
    st.subheader("About you")
    st.caption("Who you are and what you've done")

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

    picked = st.multiselect("Your skills", list(skills_map),
                            default=[pretty(s) for s in sorted(preset.skills)
                                     if pretty(s) in skills_map] if preset else [])
    typed = st.text_input("Other skills, comma-separated",
                          placeholder="e.g. Cobol, SAS, Snowflake")
    skills = to_canonical(picked, skills_map) | to_canonical(typed.split(","), {})

    st.markdown("**Education**")
    e1, e2 = st.columns(2)
    completed = e1.selectbox("Highest completed", EDU_LEVELS, index=2)
    pursuing = e2.selectbox("Currently studying for", [NOT_ENROLLED] + EDU_LEVELS[1:],
                            index=3 if (preset and preset.education_in_progress == 4) else 0,
                            help="Counts as a half-step below the degree, so being part-way "
                                 "through does not rule you out of roles that ask for it.")

    st.markdown("**Experience**")
    years = st.number_input("Years of professional experience", 0.0, 50.0,
                            float(preset.years_experience) if preset else 0.0, 0.5)

    with st.expander("Projects, coursework and certifications (optional)"):
        st.caption("Free text. This is searched alongside your résumé, so anything here can "
                   "become the evidence quoted next to a match.")
        extra_documents = st.text_area("Anything else worth knowing", height=110,
                                       label_visibility="collapsed")

# ── Job preferences ─────────────────────────────────────────────────────────
with want:
    st.subheader("Job preferences")
    st.caption("What you want")

    picked_titles = st.multiselect(
        "Job titles", list(titles_map),
        default=[pretty(t) for t in (preset.preferred_titles if preset else [])
                 if pretty(t) in titles_map])
    typed_titles = st.text_input("Other job titles, comma-separated",
                                 placeholder="e.g. Data Science Engineer, Data Pipeline Engineer")
    titles = sorted(to_canonical(picked_titles, titles_map)
                    | to_canonical(typed_titles.split(","), {}))

    cities = city_choices()
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
                                   help="Only about 30% of postings state a salary.")

    physical = bool({"Hybrid", "On-site"} & work_settings)
    max_distance = st.slider("Maximum commute (miles)", 10, 250,
                             int(preset.max_distance_miles) if preset else 100,
                             disabled=not (location and physical),
                             help="Applies to on-site and hybrid roles. Remote ignores distance.")

st.divider()

# Validation runs on submit, not while the user is still typing.
if st.button("Search jobs", type="primary", use_container_width=True):
    errors = validate_profile(resume_text, career_goals, skills, work_settings,
                              employment_types, location)
    if errors:
        st.session_state["profile_errors"] = errors
    else:
        st.session_state.pop("profile_errors", None)
        st.session_state["profile"] = UserProfile(
            name=preset.name if preset else "You",
            career_goals=career_goals,
            resume_text=resume_text + ("\n\n" + extra_documents if extra_documents else ""),
            skills=skills, years_experience=years,
            highest_completed_education=education_ordinal(completed) or 3,
            education_in_progress=education_ordinal(pursuing),
            preferred_titles=titles, preferred_location=location or None,
            accepted_work_settings=work_settings,
            accepted_employment_types=employment_types,
            min_salary=min_salary, include_unlisted_salary=include_unlisted,
            max_distance_miles=max_distance)
        st.session_state.pop("selected_job", None)
        st.switch_page("views/results.py")

for message in st.session_state.get("profile_errors", []):
    st.warning(message)
