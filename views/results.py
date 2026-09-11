"""
Matches — a job list beside a detail pane, the pattern job boards use.

The list shows what you scan by (title, company, location, salary, match); the
detail pane shows everything about the job you clicked, including how its match
was calculated. Scanning and reading are different tasks and get different space.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explain import component_bars, evidence_table, gaps_and_unknowns, verdict
from src.scoring import WEIGHTS

if "profile" not in st.session_state:
    st.title("Matches")
    st.info("Build a profile first.")
    if st.button("Go to Profile"):
        st.switch_page("views/profile.py")
    st.stop()

profile = st.session_state["profile"]
EDU = {1: "High school", 2: "Associate", 3: "Bachelor's", 4: "Master's", 5: "PhD"}
TIER_ICON = {"Strong": "🟢", "Good": "🔵", "Moderate": "🟡", "Low": "⚪"}


@st.cache_resource(show_spinner="Loading jobs…")
def load_engine():
    from src.filters import warm_caches
    from src.indexing import JobIndex

    jobs = pd.read_parquet("data/processed/jobs_tech.parquet")
    index = JobIndex.load_or_build(jobs)
    warm_caches()
    return jobs, index


@st.cache_data(show_spinner="Finding matches…")
def run(_profile, weights_key: str):
    from src.personal_kb import PersonalKB
    from src.pipeline import search

    jobs, index = load_engine()
    kb = PersonalKB.build(_profile.resume_text, _profile.career_goals)
    return search(jobs, index, _profile, kb, top_n=10)


overrides = st.session_state.get("weight_overrides")
if overrides:
    WEIGHTS.update(overrides)
result = run(profile, str(sorted((overrides or {}).items())))

bar_l, bar_r = st.columns([3, 1])
bar_l.title("Matches")
bar_l.caption(f"{result.funnel['skill_floor']:,} jobs met your requirements · "
              f"showing the {len(result.results)} best")
if bar_r.button("Edit profile", use_container_width=True):
    st.switch_page("views/profile.py")

if not result.results:
    st.error(result.funnel.get("suggestion", "No jobs matched your requirements."))
    st.stop()

if "selected_job" not in st.session_state:
    st.session_state["selected_job"] = result.results[0]["job_id"]

listing, detail = st.columns([2, 3], gap="medium")

# ── the list: what you scan ─────────────────────────────────────────────────
with listing:
    for r in result.results:
        job = r["job"]
        selected = job["job_id"] == st.session_state["selected_job"]
        salary = (f"${job['salary_min']:,.0f}–${job['salary_max']:,.0f}"
                  if job["salary_listed"] else "Salary not listed")
        with st.container(border=True):
            head, score = st.columns([4, 1])
            head.markdown(f"{'▸ ' if selected else ''}**{job['title']}**")
            head.caption(f"{job['company']}  \n{job['location_raw']} · {job['work_setting']}  \n"
                         f"{job['employment_type']} · {salary}")
            score.markdown(f"### {r['score']}")
            score.caption(f"{TIER_ICON[r['tier']]} {r['tier']}")
            st.caption(f"✓ {len(r['matched_skills'])} of "
                       f"{len(r['matched_skills']) + len(r['missing_skills'])} skills matched")
            if not selected:
                if st.button("View details", key=f"sel_{job['job_id']}",
                             use_container_width=True):
                    st.session_state["selected_job"] = job["job_id"]
                    st.rerun()

# ── the detail: what you read ───────────────────────────────────────────────
with detail:
    r = next(x for x in result.results if x["job_id"] == st.session_state["selected_job"])
    job = r["job"]
    gaps = gaps_and_unknowns(r, profile)
    headline, advice = verdict(r, gaps)
    salary = (f"${job['salary_min']:,.0f} – ${job['salary_max']:,.0f}"
              if job["salary_listed"] else "Salary not listed")
    exp = (f"{job['min_years_exp']:.0f}+ years"
           if job.get("min_years_exp") is not None else "not stated")

    with st.container(border=True):
        t, s = st.columns([4, 1])
        t.subheader(job["title"])
        t.markdown(f"**{job['company']}** · {job['location_raw']}")
        t.caption(f"{job['employment_type']} · {job['work_setting']}"
                  f"{' (inferred from the posting)' if job.get('work_setting_inferred') else ''}"
                  f" · {salary}")
        s.metric("Match", r["score"])
        s.caption(f"{TIER_ICON[r['tier']]} {r['tier']}")
        st.markdown(f"**{headline}.** {advice}")

        # AC-11.14: every posting in this historical corpus has closed. Linking
        # without saying so would imply an application is still possible.
        expiry = job.get("expiry_date")
        url = job.get("posting_url")
        link_col, warn_col = st.columns([1, 3])
        if url:
            link_col.link_button("View original posting ↗", url, use_container_width=True)
        warn_col.warning(
            f"This posting closed on **{expiry}**. The dataset is historical "
            "(LinkedIn, Dec 2023 – Apr 2024), so it is a record of a real job, not a live "
            "opening. “Applied” here means *you applied*, not that you can.", icon="🕗")

        saved = st.session_state.setdefault("decisions", {})
        current = saved.get(job["job_id"], {}).get("status", "Undecided")
        choices = ["Undecided", "Applied", "Saved for later", "Not interested"]
        d1, d2 = st.columns([3, 1])
        choice = d1.radio("Your decision", choices, index=choices.index(current),
                          key=f"decision_{job['job_id']}", horizontal=True)
        # Selecting a radio changes nothing until it is confirmed - a stray click
        # should not silently file a job.
        if d2.button("Save", key=f"save_{job['job_id']}", use_container_width=True,
                     disabled=choice == current):
            if choice == "Undecided":
                saved.pop(job["job_id"], None)
            else:
                saved[job["job_id"]] = {
                    "status": choice, "title": job["title"], "company": job["company"],
                    "location": job["location_raw"], "score": r["score"], "tier": r["tier"],
                    "url": job.get("posting_url"), "expiry": str(job.get("expiry_date"))}
            st.rerun()
        if current != "Undecided":
            st.caption(f"✓ Filed under **{current}** — see *My jobs*.")

    overview, why, evidence, gaps_tab = st.tabs(
        ["Job description", "Why this match", "Evidence", f"Gaps & unknowns ({len(gaps)})"])

    with overview:
        m1, m2 = st.columns(2)
        m1.markdown("**Requires**\n\n" +
                    ("\n".join(f"- {s}" for s in list(job["required_skills"])[:10]) or "—"))
        m2.markdown(f"**Experience**  \n{exp}\n\n**Education**  \n"
                    f"{EDU.get(job.get('education_required'), 'not stated')}")
        st.markdown("---")
        st.write(job["description"])

    with why:
        c1, c2 = st.columns(2)
        c1.markdown("**This job**\n\n"
                    f"- Needs: {', '.join(list(job['required_skills'])[:6]) or '—'}\n"
                    f"- {exp}\n- {job['location_raw']}")
        pursuing = (f" (studying for {EDU.get(profile.education_in_progress, '')})"
                    if profile.education_in_progress else "")
        c2.markdown("**You**\n\n"
                    f"- Have: {', '.join(sorted(profile.skills)[:6])}\n"
                    f"- {profile.years_experience:g} years\n"
                    f"- {EDU.get(profile.highest_completed_education, '—')}{pursuing}")
        st.markdown("---")
        bars = component_bars(r)
        for _, row in bars.iterrows():
            b, n = st.columns([4, 1])
            b.progress(min(1.0, row["Fraction"]), text=row["Component"])
            n.markdown("—" if row["Dropped"] else f"**{row['Earned']:.1f}**/{row['Max']:.0f}")
        st.caption(f"Points total **{bars['Earned'].sum():.0f}** — your match score. "
                   "“—” means this posting gave that factor nothing to measure, so it was "
                   "left out and its share spread across the others.")
        s1, s2 = st.columns(2)
        s1.success(f"**You have ({len(r['matched_skills'])})**\n\n" +
                   ("\n".join(f"- {s}" for s in r["matched_skills"]) or "—"))
        s2.warning(f"**You'd need ({len(r['missing_skills'])})**\n\n" +
                   ("\n".join(f"- {s}" for s in r["missing_skills"][:12]) or "—"))

    with evidence:
        st.dataframe(evidence_table(r, profile), hide_index=True, use_container_width=True)
        st.caption("Lines from your résumé are quoted exactly as you wrote them.")

    with gaps_tab:
        if len(gaps):
            st.dataframe(gaps, hide_index=True, use_container_width=True)
            st.caption("*Gap* — something you'd need. *Unknown* — something this posting "
                       "never said, so the match used a neutral default.")
        else:
            st.success("Nothing missing, and this posting stated everything we needed.")

with st.expander("Adjust what matters to you"):
    st.caption("Re-ranks the same jobs using your priorities instead of the defaults.")
    new, cols = {}, st.columns(4)
    for i, (name, default) in enumerate(WEIGHTS.items()):
        new[name] = cols[i % 4].slider(name.replace("_", " ").replace(" similarity", ""),
                                       0.0, 0.40, float(default), 0.01, key=f"w_{name}")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Re-rank", type="primary"):
        st.session_state["weight_overrides"] = new
        st.rerun()
    if overrides and b2.button("Reset"):
        del st.session_state["weight_overrides"]
        st.rerun()

