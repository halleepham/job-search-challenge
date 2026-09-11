"""
REQ-11 v1.1: results page.

Laid out as one tab per matched job rather than a vertical stack of cards: five
full breakdowns stacked required a lot of scrolling to compare two jobs, which is
the thing a match list is for. Each tab is a self-contained panel grid.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.explain import component_bars, evidence_table, gaps_and_unknowns, verdict
from src.scoring import WEIGHTS

st.set_page_config(page_title="Results", page_icon="🎯", layout="wide",
                   initial_sidebar_state="collapsed")

if "profile" not in st.session_state:
    st.title("Your matches")
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
def run(_profile, mode: str, _weights_key: str):
    from src.personal_kb import PersonalKB
    from src.pipeline import search

    jobs, index = load_engine()
    kb = PersonalKB.build(_profile.resume_text, _profile.career_goals)
    return search(jobs, index, _profile, kb, mode=mode)


# ── controls ────────────────────────────────────────────────────────────────
head, ctrl = st.columns([3, 2])
head.title("Your matches")
mode = ctrl.radio("Retrieval method", ["hybrid", "bm25", "dense"], horizontal=True,
                  label_visibility="collapsed",
                  help="Below 2,500 survivors retrieval is skipped and every eligible "
                       "job is scored exactly (D13).")

overrides = st.session_state.get("weight_overrides")
if overrides:
    WEIGHTS.update(overrides)
result = run(profile, mode, str(sorted((overrides or {}).items())))

if not result.results:
    st.error(result.funnel.get("suggestion", "No jobs matched."))
    st.stop()

# ── at-a-glance summary: all five without scrolling ─────────────────────────
summary = pd.DataFrame([{
    "#": i, "Score": r["score"], "Tier": r["tier"], "Job": r["job"]["title"],
    "Company": r["job"]["company"], "Location": r["job"]["location_raw"],
    "Matched": len(r["matched_skills"]), "Missing": len(r["missing_skills"]),
} for i, r in enumerate(result.results, 1)])
st.dataframe(summary, hide_index=True, use_container_width=True)

with st.expander("Pipeline · how the corpus narrowed, and timings"):
    f = result.funnel
    stages = [("Corpus", f["start"]), ("Location", f["location"]), ("Salary", f["salary"]),
              ("Setting", f["work_setting"]), ("Type", f["employment_type"]),
              ("Shared skill", f["skill_floor"]), ("Shown", f.get("returned", 0))]
    for col, (label, n) in zip(st.columns(len(stages)), stages):
        col.metric(label, f"{n:,}")
    st.caption(f"Retrieval: {f.get('retrieval', '—')} · total "
               f"{result.timings_ms['total']:.0f} ms · " +
               " · ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items() if k != "total"))
    st.plotly_chart(
        px.bar(summary, x="Score", y="Job", color="Tier", orientation="h", range_x=[0, 100],
               height=260, color_discrete_map={"Strong": "#2e7d32", "Good": "#1565c0",
                                               "Moderate": "#ef6c00", "Low": "#757575"}),
        use_container_width=True)

with st.expander("Adjust what matters to you — re-scores the same candidates"):
    st.caption("Values are relative and get renormalised. This is you setting the weights "
               "directly, not the system learning them from clicks.")
    new, cols = {}, st.columns(4)
    for i, (name, default) in enumerate(WEIGHTS.items()):
        new[name] = cols[i % 4].slider(name.replace("_", " ").replace(" similarity", ""),
                                       0.0, 0.40, float(default), 0.01, key=f"w_{name}")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Re-score", type="primary"):
        st.session_state["weight_overrides"] = new
        st.rerun()
    if overrides and b2.button("Reset"):
        del st.session_state["weight_overrides"]
        st.rerun()
if overrides:
    st.caption("⚖️ Scoring with **your** weights, not the designed ones.")

EDU = {1: "High school", 2: "Associate", 3: "Bachelor's", 4: "Master's", 5: "PhD"}

# ── one tab per job ─────────────────────────────────────────────────────────
tabs = st.tabs([f"{i} · {r['score']} · {r['job']['title'][:26]}"
                for i, r in enumerate(result.results, 1)])

for tab, r in zip(tabs, result.results):
    with tab:
        job = r["job"]
        gaps = gaps_and_unknowns(r, profile)
        headline, advice = verdict(r, gaps)
        salary = (f"${job['salary_min']:,.0f} – ${job['salary_max']:,.0f}"
                  if job["salary_listed"] else "not listed")

        title_col, score_col = st.columns([5, 1])
        title_col.subheader(job["title"])
        title_col.caption(f"**{job['company']}** · {job['location_raw']} · "
                          f"{job['employment_type']} · {job['work_setting']}"
                          f"{' *(inferred)*' if job.get('work_setting_inferred') else ''} · {salary}")
        score_col.metric(r["tier"], r["score"])
        st.markdown(f"**{headline}.** {advice}")

        left, right = st.columns(2)

        with left:                                              # AC-11.8, AC-11.10
            st.markdown("**Job & you**")
            jc1, jc2 = st.columns(2)
            jc1.markdown(
                f"Job\n\n- {', '.join(list(job['required_skills'])[:6]) or '—'}\n"
                f"- {job['min_years_exp']:.0f}+ yrs\n" if job.get("min_years_exp") is not None
                else f"Job\n\n- {', '.join(list(job['required_skills'])[:6]) or '—'}\n- yrs unstated\n")
            pursuing = (f" (pursuing {EDU.get(profile.education_in_progress, '')})"
                        if profile.education_in_progress else "")
            jc2.markdown(
                f"You\n\n- {', '.join(sorted(profile.skills)[:6])}\n"
                f"- {profile.years_experience:g} yrs\n"
                f"- {EDU.get(profile.highest_completed_education, '—')}{pursuing}\n"
                f"- {profile.preferred_location or 'Any location'}")

            st.markdown("**Overall match**")
            bars = component_bars(r)
            for _, row in bars.iterrows():
                b, n = st.columns([4, 1])
                b.progress(min(1.0, row["Fraction"]), text=row["Component"])
                n.markdown("—" if row["Dropped"] else f"**{row['Earned']:.1f}**/{row['Max']:.0f}")
            st.caption(f"Points total **{bars['Earned'].sum():.0f}** — the score above. "
                       "“—” means the posting gave that component nothing to measure, so it was "
                       "dropped and its weight redistributed.")

        with right:                                             # AC-11.9, AC-11.11
            ev_tab, gap_tab, skill_tab = st.tabs(
                ["Evidence", f"Gaps & unknowns ({len(gaps)})", "Skills"])
            with ev_tab:
                st.dataframe(evidence_table(r, profile), hide_index=True,
                             use_container_width=True, height=260)
                st.caption("Résumé rows are exact spans of your document — never paraphrased.")
            with gap_tab:
                if len(gaps):
                    st.dataframe(gaps, hide_index=True, use_container_width=True, height=260)
                    st.caption("*Gap* = something you lack. *Unknown* = something the posting "
                               "never stated, where the score used a neutral default.")
                else:
                    st.success("No gaps, and the posting stated everything the score needs.")
            with skill_tab:
                s1, s2 = st.columns(2)
                s1.success(f"**Matched ({len(r['matched_skills'])})**\n\n" +
                           ("\n".join(f"- {s}" for s in r["matched_skills"]) or "—"))
                s2.warning(f"**Missing ({len(r['missing_skills'])})**\n\n" +
                           ("\n".join(f"- {s}" for s in r["missing_skills"][:12]) or "—"))

        with st.expander("Full job description"):
            st.write(job["description"])

        st.radio("Decision", ["Undecided", "Apply", "Save for later", "Dismiss"],
                 key=f"decision_{r['job_id']}", horizontal=True)

# ── footer ──────────────────────────────────────────────────────────────────
decisions = {k.removeprefix("decision_"): v for k, v in st.session_state.items()
             if k.startswith("decision_") and v != "Undecided"}
if decisions:
    st.success("**Your decisions:** " + " · ".join(f"{v} (job {k})" for k, v in decisions.items()))

st.caption(
    "**Takeaway.** An explainable match shows the score, the evidence behind it, and the gaps and "
    "unknowns around it — so the decision stays yours. Every number is produced by a stated rule, "
    "every résumé quote is an exact span of your own document, and every component the posting "
    "could not support is shown as dropped rather than quietly guessed."
)
