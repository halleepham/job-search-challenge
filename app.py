"""
Job Search & Matching Application — entry point.

Three pages (REQ-3 §page structure): Profile, Results, Analytics. Shared,
expensive resources are cached once here so page switches are instant.

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

CORPUS = Path("data/processed/jobs_tech.parquet")

st.set_page_config(page_title="Job Search & Matching", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading corpus and indexes…")
def load_engine():
    """Corpus, indexes and geocoding tables — loaded once per session."""
    from src.filters import warm_caches
    from src.indexing import JobIndex

    jobs = pd.read_parquet(CORPUS)
    index = JobIndex.load_or_build(jobs)
    warm_caches()
    return jobs, index


st.title("Job Search & Matching")
st.caption(
    "CS 5542 Challenge 1 · Human-AI Co-Design — filters eliminate, scores rank, "
    "and every number is traceable to a stated rule."
)

if not CORPUS.exists():
    st.error(
        "Corpus not built. Run:\n\n"
        "```\npython -m src.download_data\npython -m src.ingest\n```"
    )
    st.stop()

jobs, index = load_engine()

col1, col2, col3 = st.columns(3)
col1.metric("Job postings", f"{len(jobs):,}")
col2.metric("With a listed salary", f"{jobs['salary_listed'].mean():.0%}")
col3.metric("Remote", f"{jobs['work_setting'].eq('Remote').mean():.0%}")

st.markdown(
    """
### How a match is produced

1. **Filter** — jobs failing a non-negotiable constraint are *eliminated*, not penalised
2. **Retrieve** — BM25 (keywords) and embeddings (meaning), fused by rank
3. **Score** — eight weighted components, each explainable on its own
4. **Rank** — top 5, with the résumé line supporting every matched skill

Start on **Profile** in the sidebar, or load a preset there to see results immediately.
"""
)

st.info(
    "**About this corpus.** LinkedIn IT, engineering, analytics, QA and science job "
    "functions — roughly a quarter is strictly software/data work. It is described that "
    "way deliberately rather than as \"tech jobs\", which would overstate it."
)
