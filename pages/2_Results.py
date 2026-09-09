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


from src.explain import component_bars, evidence_table, gaps_and_unknowns, verdict
from src.scoring import WEIGHTS

top_l, top_r = st.columns([2, 3])
mode = top_l.radio("Retrieval method", ["hybrid", "bm25", "dense"], horizontal=True,
                   help="Hybrid fuses both. Below 2,500 survivors retrieval is skipped "
                        "entirely and every eligible job is scored exactly (D13).")

# AC-11.12: explicit weight adjustment. Not the learned feedback loop cut in
# Non-Goals - the user sets these directly and sees the consequence.
with top_r.expander("Adjust what matters to you"):
    st.caption("Re-scores the same candidates. Values are relative; they are renormalised.")
    overrides, cols = {}, st.columns(4)
    for i, (name, default) in enumerate(WEIGHTS.items()):
        overrides[name] = cols[i % 4].slider(
            name.replace("_", " ").replace(" similarity", ""),
            0.0, 0.40, float(default), 0.01, key=f"w_{name}")
    if st.button("Re-score with these weights"):
        st.session_state["weight_overrides"] = overrides
        st.cache_data.clear()
    if st.session_state.get("weight_overrides"):
        if st.button("Reset to the designed weights"):
            del st.session_state["weight_overrides"]
            st.cache_data.clear()

if st.session_state.get("weight_overrides"):
    WEIGHTS.update(st.session_state["weight_overrides"])
    st.info("Scoring with **your** weights, not the designed ones.")

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

        gaps = gaps_and_unknowns(r, profile)
        headline, advice = verdict(r, gaps)                       # AC-11.13

        # AC-11.8: job and candidate side by side — the comparison the score is about.
        jc1, jc2 = st.columns(2)
        with jc1:
            st.markdown("**Job**")
            st.markdown(f"- {job['company']}\n- {job['location_raw']} ({job['work_setting']})\n"
                        f"- {job['employment_type']}\n- {salary}\n"
                        f"- Requires: {', '.join(list(job['required_skills'])[:6]) or '—'}")
        with jc2:
            st.markdown("**You**")
            edu = {1: "High school", 2: "Associate", 3: "Bachelor's", 4: "Master's", 5: "PhD"}
            pursuing = (f" (pursuing {edu.get(profile.education_in_progress, '')})"
                        if profile.education_in_progress else "")
            st.markdown(
                f"- {edu.get(profile.highest_completed_education, '—')}{pursuing}\n"
                f"- {profile.years_experience:g} years experience\n"
                f"- {profile.preferred_location or 'Any location'}\n"
                f"- Accepts: {', '.join(sorted(profile.accepted_work_settings))}\n"
                f"- Skills: {', '.join(sorted(profile.skills)[:6])}")

        st.markdown(f"**{headline}.** {advice}")

        # AC-11.10: bars showing earned points out of maximum.
        st.markdown("**Overall match**")
        bars = component_bars(r)
        for _, row in bars.iterrows():
            bar_col, num_col = st.columns([5, 1])
            bar_col.progress(min(1.0, row["Fraction"]), text=row["Component"])
            num_col.markdown(
                "—" if row["Dropped"] else f"**{row['Earned']:.1f}** / {row['Max']:.1f}")
        st.caption(f"Points total **{bars['Earned'].sum():.0f}** — the score above. "
                   "“—” means the component was dropped for this job (the posting gave it "
                   "nothing to measure) and its weight was redistributed across the rest.")

        ev_col, gap_col = st.columns(2)
        with ev_col:                                              # AC-11.9
            st.markdown("**Evidence retrieved**")
            st.dataframe(evidence_table(r, profile), hide_index=True,
                         use_container_width=True)
            st.caption("Résumé rows are exact spans of your document — never paraphrased "
                       "or generated.")
        with gap_col:                                             # AC-11.11
            st.markdown("**Gaps & unknowns**")
            if len(gaps):
                st.dataframe(gaps, hide_index=True, use_container_width=True)
                st.caption("*Gap* = something you lack. *Unknown* = something the posting "
                           "never stated, where the score used a neutral default.")
            else:
                st.success("No gaps, and the posting stated everything the score needs.")

        # AC-11.12: per-result action.
        act_col, _ = st.columns([2, 3])
        act_col.radio("Decision", ["Undecided", "Apply", "Save for later", "Dismiss"],
                      key=f"decision_{r['job_id']}", horizontal=False)

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

decisions = {k.removeprefix("decision_"): v for k, v in st.session_state.items()
             if k.startswith("decision_") and v != "Undecided"}
if decisions:
    st.markdown("**Your decisions:** " +
                " · ".join(f"{v} ({k})" for k, v in decisions.items()))

st.divider()
st.info(
    "**Takeaway.** An explainable match shows the score, the evidence behind it, and the "
    "gaps and unknowns around it — so the decision stays yours. Every number above is "
    "produced by a stated rule, every résumé quote is an exact span of your own document, "
    "and every component the posting could not support is shown as dropped rather than "
    "quietly guessed."
)
