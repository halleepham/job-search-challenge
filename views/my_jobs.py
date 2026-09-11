"""My jobs — what the user filed from the matches list."""

from __future__ import annotations

import pandas as pd
import streamlit as st

STATUSES = ["Applied", "Saved for later", "Not interested"]
ICONS = {"Applied": "✅", "Saved for later": "🔖", "Not interested": "🚫"}

st.title("My jobs")
decisions = st.session_state.get("decisions", {})

if not decisions:
    st.info("Nothing filed yet. Open a match and choose **Applied**, **Saved for later** "
            "or **Not interested**, then press Save.")
    if st.button("Go to matches"):
        st.switch_page("views/results.py")
    st.stop()

st.caption(f"{len(decisions)} job(s) filed this session. "
           "Decisions are kept while the app is open.")

frame = pd.DataFrame([{"Status": v["status"], "Job": v["title"], "Company": v["company"],
                       "Location": v["location"], "Match": v["score"], "job_id": job_id}
                      for job_id, v in decisions.items()])

tabs = st.tabs([f"{ICONS[s]} {s} ({(frame.Status == s).sum()})" for s in STATUSES])
for tab, status in zip(tabs, STATUSES):
    with tab:
        subset = frame[frame.Status == status]
        if subset.empty:
            st.caption(f"No jobs marked *{status}*.")
            continue
        for row in subset.itertuples():
            with st.container(border=True):
                left, score, action = st.columns([5, 1, 1])
                left.markdown(f"**{row.Job}**")
                left.caption(f"{row.Company} · {row.Location}")
                score.metric("Match", row.Match)
                if action.button("Remove", key=f"rm_{row.job_id}", use_container_width=True):
                    st.session_state["decisions"].pop(row.job_id, None)
                    st.rerun()

st.divider()
st.download_button("Download as CSV", frame.drop(columns=["job_id"]).to_csv(index=False),
                   "my_jobs.csv", "text/csv")
