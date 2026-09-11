"""
Profile — the form describing the person and what they are looking for.

Every control is keyed, and the keys are the single source of truth. Loading a
preset or a saved profile writes into those keys and reruns; the form therefore
survives navigation (AC-3.11) instead of discarding a résumé and twelve fields
because the user wanted to change one salary figure.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.display import display_map, pretty, to_canonical
from src.profiles import PRESETS, UserProfile
from src.skills import load_vocabulary, match_skills
from src.validation import (
    EMPLOYMENT_TYPES,
    NOT_ENROLLED,
    WORK_SETTINGS,
    education_ordinal,
    validate_profile,
)

EDU_LEVELS = ["High School", "Associate", "Bachelor's", "Master's", "PhD"]
EDU_FROM_ORDINAL = {1: "High School", 2: "Associate", 3: "Bachelor's",
                    4: "Master's", 5: "PhD"}
FORM_KEYS = ["pf_resume", "pf_goals", "pf_skills", "pf_other_skills", "pf_years",
             "pf_completed", "pf_pursuing", "pf_extra", "pf_titles", "pf_other_titles",
             "pf_location", "pf_salary", "pf_unlisted", "pf_distance"]


@st.cache_data
def skill_choices() -> dict[str, str]:
    return display_map(sorted(load_vocabulary().canonical))


@st.cache_data
def title_choices() -> dict[str, str]:
    """
    Titles come from the corpus — the 300 most common normalized titles — rather
    than a hand-written list. Free text is accepted alongside, so a title the
    corpus happens not to contain is never a dead end.
    """
    jobs = pd.read_parquet("data/processed/jobs_tech.parquet", columns=["title_normalized"])
    common = jobs["title_normalized"].value_counts().head(300).index
    return display_map(sorted(t for t in common if t and len(t) > 3))


@st.cache_data
def city_choices() -> list[str]:
    import geonamescache

    cities = geonamescache.GeonamesCache().get_cities().values()
    return [""] + sorted({f"{c['name']}, {c['admin1code']}" for c in cities
                          if c["countrycode"] == "US" and c["population"] >= 50_000})


skills_map, titles_map, cities = skill_choices(), title_choices(), city_choices()


def load_into_form(p: UserProfile) -> None:
    """Write a profile into the widget keys, which are the form's state."""
    st.session_state.update({
        "pf_resume": p.resume_text, "pf_goals": p.career_goals,
        "pf_skills": [pretty(x) for x in sorted(p.skills) if pretty(x) in skills_map],
        "pf_other_skills": ", ".join(pretty(x) for x in sorted(p.skills)
                                     if pretty(x) not in skills_map),
        "pf_years": float(p.years_experience),
        "pf_completed": EDU_FROM_ORDINAL.get(p.highest_completed_education, "Bachelor's"),
        "pf_pursuing": EDU_FROM_ORDINAL.get(p.education_in_progress, NOT_ENROLLED),
        "pf_titles": [pretty(t) for t in p.preferred_titles if pretty(t) in titles_map],
        "pf_other_titles": ", ".join(pretty(t) for t in p.preferred_titles
                                     if pretty(t) not in titles_map),
        "pf_location": p.preferred_location if p.preferred_location in cities else "",
        "pf_salary": int(p.min_salary), "pf_unlisted": bool(p.include_unlisted_salary),
        "pf_distance": int(p.max_distance_miles),
    })
    for s in WORK_SETTINGS:
        st.session_state[f"ws_{s}"] = s in p.accepted_work_settings
    for t in EMPLOYMENT_TYPES:
        st.session_state[f"et_{t}"] = t in p.accepted_employment_types


def clear_form() -> None:
    for key in FORM_KEYS:
        st.session_state.pop(key, None)
    for s in WORK_SETTINGS:
        st.session_state[f"ws_{s}"] = True
    for t in EMPLOYMENT_TYPES:
        st.session_state[f"et_{t}"] = True
    st.session_state.pop("profile_errors", None)


# First visit only: nothing to restore, so seed sensible defaults.
if "pf_resume" not in st.session_state:
    clear_form()

st.title("Profile")
st.caption("Tell us about yourself and what you're looking for. "
           "Every match comes with the evidence behind it.")

saved_profiles = st.session_state.setdefault("saved_profiles", {})
load_col, clear_col = st.columns([3, 1])
options = ["—"] + [f"Saved · {n}" for n in saved_profiles] + \
          [f"Example · {PRESETS[k].name}" for k in PRESETS]
chosen = load_col.selectbox("Load a profile", options,
                            help="Your own saved profiles, then the built-in examples.")
if chosen != "—" and load_col.button("Load", use_container_width=True):
    if chosen.startswith("Saved · "):
        load_into_form(saved_profiles[chosen.removeprefix("Saved · ")])
    else:
        name = chosen.removeprefix("Example · ")
        load_into_form(next(p for p in PRESETS.values() if p.name == name))
    st.rerun()
if clear_col.button("Clear form", use_container_width=True):
    clear_form()
    st.rerun()

st.divider()
about, want = st.columns(2, gap="large")

with about:
    st.subheader("About you")
    st.caption("Who you are and what you've done")

    upload = st.file_uploader("Upload a résumé (PDF, TXT or MD)", type=["pdf", "txt", "md"])
    if upload:
        if upload.name.lower().endswith(".pdf"):
            from pypdf import PdfReader

            text = "\n\n".join(pg.extract_text() or "" for pg in PdfReader(upload).pages)
        else:
            text = upload.read().decode("utf-8", errors="ignore")
        if text.strip() and text != st.session_state.get("pf_resume"):
            st.session_state["pf_resume"] = text
            st.rerun()

    resume_text = st.text_area("Résumé", key="pf_resume", height=170,
                               placeholder="Paste your résumé, or upload one above…")
    career_goals = st.text_area("What kind of work are you looking for?", key="pf_goals",
                                height=80,
                                placeholder="e.g. a data engineering role building pipelines")

    # AC-3.10: the extractor that reads job postings reads the user's own text too,
    # so skills they have but did not think to list are not scored as absent.
    found = match_skills(f"{resume_text} {career_goals}")
    new_found = [pretty(x) for x in sorted(found)
                 if pretty(x) in skills_map and pretty(x) not in st.session_state["pf_skills"]]
    if new_found and st.button(f"➕ Add {len(new_found)} skill(s) found in your text"):
        st.session_state["pf_skills"] = st.session_state["pf_skills"] + new_found
        st.rerun()

    picked = st.multiselect("Your skills", list(skills_map), key="pf_skills")
    typed = st.text_input("Other skills, comma-separated", key="pf_other_skills",
                          placeholder="e.g. Cobol, SAS")
    skills = to_canonical(picked, skills_map) | to_canonical(typed.split(","), {})

    st.markdown("**Education**")
    e1, e2 = st.columns(2)
    completed = e1.selectbox("Highest completed", EDU_LEVELS, key="pf_completed")
    pursuing = e2.selectbox("Currently studying for", [NOT_ENROLLED] + EDU_LEVELS[1:],
                            key="pf_pursuing",
                            help="Counts as a half-step below the degree, so being part-way "
                                 "through does not rule you out of roles that ask for it.")

    st.markdown("**Experience**")
    years = st.number_input("Years of professional experience", 0.0, 50.0, step=0.5,
                            key="pf_years")

    with st.expander("Projects, coursework and certifications (optional)"):
        st.caption("Searched alongside your résumé, so anything here can become the "
                   "evidence quoted next to a match.")
        extra_documents = st.text_area("Anything else worth knowing", key="pf_extra",
                                       height=110, label_visibility="collapsed")

with want:
    st.subheader("Job preferences")
    st.caption("What you want")

    picked_titles = st.multiselect("Job titles", list(titles_map), key="pf_titles")
    typed_titles = st.text_input("Other job titles, comma-separated", key="pf_other_titles",
                                 placeholder="e.g. Data Science Engineer, Data Pipeline Engineer")
    titles = sorted(to_canonical(picked_titles, titles_map)
                    | to_canonical(typed_titles.split(","), {}))

    location = st.selectbox("Location", cities, key="pf_location")

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Work setting**")
        work_settings = {s for s in WORK_SETTINGS if st.checkbox(s, key=f"ws_{s}")}
    with w2:
        st.markdown("**Employment type**")
        employment_types = {t for t in EMPLOYMENT_TYPES if st.checkbox(t, key=f"et_{t}")}

    s1, s2 = st.columns(2)
    min_salary = s1.number_input("Minimum salary (USD)", 0, 500_000, step=5_000, key="pf_salary")
    include_unlisted = s2.checkbox("Include jobs with no listed salary", key="pf_unlisted",
                                   help="Only about 30% of postings state a salary.")

    physical = bool({"Hybrid", "On-site"} & work_settings)
    max_distance = st.slider("Maximum commute (miles)", 10, 250, key="pf_distance",
                             disabled=not (location and physical),
                             help="Applies to on-site and hybrid roles. Remote ignores distance.")

st.divider()


def build() -> UserProfile:
    return UserProfile(
        name="You", career_goals=career_goals,
        resume_text=resume_text + ("\n\n" + extra_documents if extra_documents else ""),
        skills=skills, years_experience=years,
        highest_completed_education=education_ordinal(completed) or 3,
        education_in_progress=education_ordinal(pursuing),
        preferred_titles=titles, preferred_location=location or None,
        accepted_work_settings=work_settings, accepted_employment_types=employment_types,
        min_salary=min_salary, include_unlisted_salary=include_unlisted,
        max_distance_miles=max_distance)


search_col, save_col, name_col = st.columns([2, 1, 2])
# Validation runs on submit, not while the user is still filling the form.
if search_col.button("Search jobs", type="primary", use_container_width=True):
    errors = validate_profile(resume_text, career_goals, skills, work_settings,
                              employment_types, location)
    if errors:
        st.session_state["profile_errors"] = errors
    else:
        st.session_state.pop("profile_errors", None)
        st.session_state["profile"] = build()
        st.session_state.pop("selected_job", None)
        st.switch_page("views/results.py")

save_name = name_col.text_input("Name", key="pf_save_name", placeholder="e.g. Data roles, KC",
                                label_visibility="collapsed")
if save_col.button("Save profile", use_container_width=True, disabled=not save_name.strip()):
    saved_profiles[save_name.strip()] = build()
    st.success(f"Saved as **{save_name.strip()}** — load it from the dropdown above.")

for message in st.session_state.get("profile_errors", []):
    st.warning(message)
