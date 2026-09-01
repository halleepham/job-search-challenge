"""
Streamlit UI helper components, card renderers, and interactive visualization charts.
"""

from typing import List, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.models import JobMatchResult, UserProfile


def inject_custom_css():
    """Injects custom CSS styling for modern, clean card and badge rendering."""
    st.markdown(
        """
        <style>
        /* Main Container Styling */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        
        /* Metric Badges */
        .metric-card {
            background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Tier Badges */
        .tier-strong {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }
        .tier-good {
            background-color: #E1EFFE;
            color: #1E429F;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }
        .tier-moderate {
            background-color: #FEF08A;
            color: #713F12;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }
        .tier-low {
            background-color: #F3F4F6;
            color: #4B5563;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.82rem;
            display: inline-block;
        }

        /* Skill Pills */
        .skill-pill-matched {
            background-color: #E6F4EA;
            color: #137333;
            border: 1px solid #A8DAB5;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
        }
        .skill-pill-missing {
            background-color: #FCE8E6;
            color: #C5221F;
            border: 1px solid #F6AEA9;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
        }
        .skill-pill-pref {
            background-color: #E8F0FE;
            color: #1A73E8;
            border: 1px solid #AECBFA;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tier_badge(tier: str) -> str:
    """Returns HTML for tier badge."""
    if tier == "Strong Match":
        return f'<span class="tier-strong">🌟 Strong Match</span>'
    elif tier == "Good Match":
        return f'<span class="tier-good">⚡ Good Match</span>'
    elif tier == "Moderate Match":
        return f'<span class="tier-moderate">⚖️ Moderate Match</span>'
    else:
        return f'<span class="tier-low">🔍 Low Match</span>'


def render_job_card(result: JobMatchResult, user_profile: UserProfile):
    """Renders a single ranked job card with interactive score breakdown and skill tags."""
    job = result.job
    bd = result.breakdown
    tier_badge = render_tier_badge(result.match_tier)

    # Score color
    if result.overall_score >= 80:
        score_color = "#10B981"
    elif result.overall_score >= 65:
        score_color = "#3B82F6"
    elif result.overall_score >= 45:
        score_color = "#F59E0B"
    else:
        score_color = "#6B7280"

    with st.container():
        # Top Card Header
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f"""
                <div style="display: flex; align-items: baseline; gap: 8px;">
                    <span style="font-size: 1.25rem; font-weight: 800; color: #4B5563;">#{result.rank}</span>
                    <span style="font-size: 1.25rem; font-weight: 700; color: #1E293B;">{job.title}</span>
                </div>
                <div style="font-size: 0.95rem; color: #475569; margin-top: 2px;">
                    🏢 <strong>{job.company}</strong> &nbsp;•&nbsp; 📍 {job.location} &nbsp;•&nbsp; 💼 <em>{job.work_type}</em> &nbsp;•&nbsp; 🏷️ <span style="background-color: #F1F5F9; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{job.field}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: 800; color: {score_color}; line-height: 1;">
                        {result.overall_score:.1f}%
                    </div>
                    <div style="margin-top: 4px;">{tier_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quick Highlights row
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            st.caption(f"💰 **Salary**: {job.salary_range}")
        with qcol2:
            st.caption(f"🎓 **Level**: {job.experience_level} ({job.min_years_exp}+ yrs req)")
        with qcol3:
            st.caption(f"💡 **Skills Matched**: {len(bd.matched_skills)}/{len(job.required_skills)}")

        # Match summary narrative
        st.info(f"📋 **Match Analysis**: {result.match_summary}")

        # Expandable Deep-Dive
        with st.expander(f"🔍 View Full Details & Skill Match Breakdown for {job.title}", expanded=False):
            # Score Sub-bars
            st.markdown("##### 📊 Match Score Components")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Technical Skills (40%)", f"{bd.skill_score:.1f}%")
                st.progress(bd.skill_score / 100.0)
            with sc2:
                st.metric("Text / Role Fit (30%)", f"{bd.text_score:.1f}%")
                st.progress(bd.text_score / 100.0)
            with sc3:
                st.metric("Experience (15%)", f"{bd.experience_score:.1f}%")
                st.progress(bd.experience_score / 100.0)
            with sc4:
                st.metric("Field & Mode (15%)", f"{bd.preference_score:.1f}%")
                st.progress(bd.preference_score / 100.0)

            st.divider()

            # Matched vs Missing Skills
            st.markdown("##### 🛠️ Skills Breakdown")
            sk_col1, sk_col2 = st.columns(2)
            with sk_col1:
                st.markdown("**✅ Your Matched Skills:**")
                if bd.matched_skills:
                    matched_html = "".join([f'<span class="skill-pill-matched">✓ {s}</span>' for s in bd.matched_skills])
                    st.markdown(matched_html, unsafe_allow_html=True)
                else:
                    st.caption("No exact skill matches with job requirements.")

                if bd.matched_preferred_skills:
                    st.markdown("**⭐ Matched Preferred / Bonus Skills:**")
                    pref_html = "".join([f'<span class="skill-pill-pref">★ {s}</span>' for s in bd.matched_preferred_skills])
                    st.markdown(pref_html, unsafe_allow_html=True)

            with sk_col2:
                st.markdown("**⚠️ Missing / Recommended Skills to Learn:**")
                if bd.missing_skills:
                    missing_html = "".join([f'<span class="skill-pill-missing">✕ {s}</span>' for s in bd.missing_skills])
                    st.markdown(missing_html, unsafe_allow_html=True)
                else:
                    st.success("🎉 You possess 100% of the required skills for this role!")

            st.divider()

            # Job Description & Responsibilities
            st.markdown("##### 📄 Job Overview & Responsibilities")
            st.write(job.description)
            st.markdown("**Key Responsibilities:**")
            for resp in job.responsibilities:
                st.markdown(f"- {resp}")

            st.caption(f"**Education Required**: {job.education_required} | **Job ID**: `{job.id}`")

        st.markdown("<hr style='margin: 1rem 0; border: 0.5px solid #E2E8F0;'>", unsafe_allow_html=True)


def plot_field_distribution(results: List[JobMatchResult]):
    """Creates an interactive bar chart of average and top match score by Field."""
    from src.matcher import get_field_match_summary
    field_summary = get_field_match_summary(results)
    
    if not field_summary:
        return None

    fields = list(field_summary.keys())
    avg_scores = [field_summary[f]["avg_score"] for f in fields]
    max_scores = [field_summary[f]["max_score"] for f in fields]
    counts = [field_summary[f]["count"] for f in fields]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fields,
        y=avg_scores,
        name="Average Match %",
        marker_color="#3B82F6",
        text=[f"{s}%" for s in avg_scores],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=fields,
        y=max_scores,
        name="Top Match %",
        marker_color="#10B981",
        text=[f"{s}%" for s in max_scores],
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Match Scores by Job Field</b>",
        barmode="group",
        xaxis_title="Job Field / Domain",
        yaxis_title="Match Score (%)",
        yaxis=dict(range=[0, 105]),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=380,
    )
    return fig


def plot_top_jobs_radar(results: List[JobMatchResult], top_n: int = 4):
    """Creates a multi-job radar chart comparing breakdown categories for the top N matches."""
    top_matches = results[:top_n]
    if not top_matches:
        return None

    categories = ["Technical Skills", "Text/Role Fit", "Experience", "Field & Preferences"]
    fig = go.Figure()

    colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B"]

    for idx, r in enumerate(top_matches):
        bd = r.breakdown
        values = [bd.skill_score, bd.text_score, bd.experience_score, bd.preference_score]
        # Close the loop for radar chart
        values.append(values[0])
        cats = categories + [categories[0]]

        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=f"#{r.rank} {r.job.title} ({r.overall_score:.0f}%)",
            line_color=color,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title=f"<b>Competency Fit Comparison (Top {len(top_matches)} Jobs)</b>",
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

