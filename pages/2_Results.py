"""REQ-11: results page — top 5 with a fully decomposed score and verbatim evidence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Results", page_icon="🎯", layout="wide")
st.title("Your matches")

if "profile" not in st.session_state:
    st.info("Build a profile first — open **Profile** in the sidebar.")
    st.stop()

profile = st.session_state["profile"]


@st.cache_resource(show_spinner="Loading corpus and indexes…")
def load_engine():
    from src.filters import warm_caches
    from src.indexing import JobIndex

    jobs = pd.read_parquet("data/processed/jobs_tech.parquet")
    index = JobIndex.load_or_build(jobs)
    warm_caches()
    return jobs, index


@st.cache_data(show_spinner="Matching…")
def run(_profile, mode: str):
    from src.personal_kb import PersonalKB
    from src.pipeline import search

    jobs, index = load_engine()
    kb = PersonalKB.build(_profile.resume_text, _profile.career_goals)
    return search(jobs, index, _profile, kb, mode=mode)


mode = st.radio("Retrieval method", ["hybrid", "bm25", "dense"], horizontal=True,
                help="Hybrid fuses both. The other two are here because the evaluation "
                     "notebook compares them through this same entry point.")
result = run(profile, mode)

# --- AC-11.2: the funnel, so the pipeline is legible -------------------------
f = result.funnel
st.subheader("How the corpus narrowed")
stages = [("Corpus", f["start"]), ("Location", f["location"]), ("Salary", f["salary"]),
          ("Work setting", f["work_setting"]), ("Employment", f["employment_type"]),
          ("Shared skill", f["skill_floor"]), ("Retrieved", f.get("retrieved", 0)),
          ("Shown", f.get("returned", 0))]
cols = st.columns(len(stages))
for col, (label, n) in zip(cols, stages):
    col.metric(label, f"{n:,}")
st.caption(f"Search took {result.timings_ms['total']:.0f} ms · " +
           " · ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items() if k != "total"))

if not result.results:
    st.error(f.get("suggestion", "No jobs matched."))
    st.stop()

st.divider()

TIER_COLOR = {"Strong": "🟢", "Good": "🔵", "Moderate": "🟡", "Low": "⚪"}
LABELS = {
    "required_skills": "Required skills", "preferred_skills": "Preferred skills",
    "experience": "Experience", "education": "Education", "title": "Job title",
    "location": "Location", "career_goals_similarity": "Career goals ~ description",
    "resume_evidence_similarity": "Résumé evidence ~ description",
}

for rank, r in enumerate(result.results, start=1):
    job = r["job"]
    salary = (f"${job['salary_min']:,.0f} – ${job['salary_max']:,.0f}"
              if job["salary_listed"] else "Salary not listed")

    with st.container(border=True):
        head, score_col = st.columns([5, 1])
        head.markdown(f"### {rank}. {job['title']}")
        head.caption(
            f"**{job['company']}** · {job['location_raw']} · {job['employment_type']} · "
            f"{job['work_setting']}{' *(inferred)*' if job.get('work_setting_inferred') else ''} · {salary}"
        )
        score_col.metric(f"{TIER_COLOR[r['tier']]} {r['tier']}", r["score"])

        c1, c2 = st.columns(2)
        c1.success(f"**Matched ({len(r['matched_skills'])})** · " +
                   (", ".join(r["matched_skills"]) or "—"))
        c2.warning(f"**Missing ({len(r['missing_skills'])})** · " +
                   (", ".join(r["missing_skills"][:10]) or "—"))

        with st.expander("Why this score?"):
            # AC-11.3: contributions must sum to the displayed score.
            rows = [{"Component": LABELS[c["name"]],
                     "Sub-score": "—" if c["sub_score"] is None else round(c["sub_score"], 2),
                     "Weight": f"{c['weight']:.0%}",
                     "Points": round(c["weighted"] * 100, 1)} for c in r["components"]]
            table = pd.DataFrame(rows)
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption(f"Contributions total **{table['Points'].sum():.0f}** = the score above. "
                       "A component showing “—” was dropped for this job and its weight "
                       "redistributed across the rest.")

            if r["evidence"]:
                st.markdown("**Supporting text from your résumé** — quoted verbatim:")
                for ev in r["evidence"]:
                    st.markdown(f"> {ev['text'].strip()}")
                    st.caption(f"from your *{ev['section']}* section · similarity {ev['similarity']}")

# --- AC-11.6: score distribution --------------------------------------------
st.divider()
st.subheader("Score distribution across your matches")
chart = pd.DataFrame({
    "Job": [f"{i}. {r['job']['title'][:32]}" for i, r in enumerate(result.results, 1)],
    "Score": [r["score"] for r in result.results],
    "Tier": [r["tier"] for r in result.results]})
st.plotly_chart(
    px.bar(chart, x="Score", y="Job", color="Tier", orientation="h", range_x=[0, 100],
           color_discrete_map={"Strong": "#2e7d32", "Good": "#1565c0",
                               "Moderate": "#ef6c00", "Low": "#757575"}),
    use_container_width=True)
