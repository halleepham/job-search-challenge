"""
Job Matcher — application entry point and router.

Uses `st.navigation(position="hidden")` so the default sidebar nav is suppressed
and a top navigation bar is rendered instead: the product name stays in the
top-left on every page, the way a website behaves.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Job Matcher", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")

CORPUS = Path("data/processed/jobs_tech.parquet")

PAGES = [
    st.Page("views/profile.py", title="Profile", icon="👤", default=True),
    st.Page("views/results.py", title="Matches", icon="🎯"),
    st.Page("views/my_jobs.py", title="My jobs", icon="🔖"),
    st.Page("views/insights.py", title="Insights", icon="📊"),
]
nav = st.navigation(PAGES, position="hidden")

# ── top navigation bar ──────────────────────────────────────────────────────
brand, *links = st.columns([2.2, 1, 1, 1, 1, 2])
brand.markdown("## 🧭 Job Matcher")
for col, page in zip(links, PAGES):
    saved = len(st.session_state.get("decisions", {}))
    label = page.title
    if page.title == "My jobs" and saved:
        label = f"{page.title} ({saved})"
    col.page_link(page, label=label, icon=page.icon, use_container_width=True)
st.divider()

if not CORPUS.exists():
    st.error("Job data not built. Run:\n\n```\npython -m src.download_data\npython -m src.ingest\n```")
    st.stop()

nav.run()
