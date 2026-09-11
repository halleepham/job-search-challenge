"""REQ-12: dataset analytics page — analysis of the corpus, not of one search."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import CORPUS_LABEL

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")
st.title("What is in the job corpus")
st.caption(f"**Corpus:** {CORPUS_LABEL}")

DIR = Path("data/processed/analytics")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(DIR / f"{name}.csv")


if not (DIR / "top_skills.csv").exists():
    st.error("Analytics not computed. Run:\n\n```\npython -m src.ingest\n```")
    st.stop()

skills, titles = table("top_skills"), table("top_titles")
geo, salary = table("geographic_distribution"), table("salary_by_title_family")
setting, experience = table("work_setting"), table("experience_distribution")
companies, cooccur = table("top_companies"), table("skill_cooccurrence")

c1, c2, c3 = st.columns(3)
c1.metric("Distinct skills tracked", f"{len(skills):,}+")
c2.metric("States represented", f"{len(geo):,}")
c3.metric("Remote share",
          f"{setting.set_index('work_setting')['n'].get('Remote', 0) / setting['n'].sum():.0%}")

tabs = st.tabs(["Skills", "Titles & salary", "Geography", "Work setting & experience", "Companies"])

with tabs[0]:
    st.plotly_chart(
        px.bar(skills.head(20).sort_values("n"), x="n", y="skill", orientation="h",
               labels={"n": "postings", "skill": ""}, title="Top 20 requested skills"),
        use_container_width=True)
    st.info(
        "**Worth reading carefully.** `excel`, `word` and `powerpoint` rank near the top. "
        "That is not a bug — it is what a corpus of IT/engineering/analytics/QA/science job "
        "functions actually contains. It is also why the match score, not the corpus "
        "definition, is what makes irrelevant jobs rank low."
    )
    st.subheader("Which skills are asked for together")
    st.dataframe(cooccur.head(25).rename(
        columns={"skill_a": "Skill A", "skill_b": "Skill B", "n": "Postings"}),
        hide_index=True, use_container_width=True)

with tabs[1]:
    st.plotly_chart(
        px.bar(titles.sort_values("n"), x="n", y="title", orientation="h",
               labels={"n": "postings", "title": ""}, title="Top 20 job titles"),
        use_container_width=True)
    st.subheader("Median salary by title family")
    st.caption("Only postings that state a salary — about 30% of the corpus — are included.")
    plot = salary[salary["n_with_salary"] > 30].sort_values("median_salary_max")
    st.plotly_chart(
        px.bar(plot, x="median_salary_max", y="title_family", orientation="h",
               labels={"median_salary_max": "median top of range (USD)", "title_family": ""}),
        use_container_width=True)
    st.dataframe(salary, hide_index=True, use_container_width=True)

with tabs[2]:
    st.plotly_chart(
        px.choropleth(geo, locations="state", locationmode="USA-states", color="n",
                      scope="usa", color_continuous_scale="Blues",
                      labels={"n": "postings"}, title="Postings by state"),
        use_container_width=True)
    st.caption("States are parsed from the posting's location text; roughly 69% of rows resolve.")

with tabs[3]:
    a, b = st.columns(2)
    a.plotly_chart(px.pie(setting, values="n", names="work_setting",
                          title="Remote vs hybrid vs on-site"), use_container_width=True)
    b.plotly_chart(
        px.bar(experience[experience["years_required"] <= 15],
               x="years_required", y="n",
               labels={"years_required": "years of experience required", "n": "postings"},
               title="Experience requirements"),
        use_container_width=True)

with tabs[4]:
    st.plotly_chart(
        px.bar(companies.sort_values("n"), x="n", y="company", orientation="h",
               labels={"n": "postings", "company": ""}, title="Top 20 hiring companies"),
        use_container_width=True)
