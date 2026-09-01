"""
Main Streamlit Application for Job Search & Profile Matching Engine.
"""

import streamlit as st
import pandas as pd
from typing import List

from src.models import UserProfile, JobListing
from src.data_loader import (
    load_jobs,
    save_jobs,
    get_available_fields,
    get_all_skills,
    PRESET_PROFILES,
)
from src.matcher import rank_jobs, get_field_match_summary
from src.ui_components import (
    inject_custom_css,
    render_job_card,
    plot_field_distribution,
    plot_top_jobs_radar,
)

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Job Search & Match Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


def init_session_state():
    """Initializes session state variables."""
    if "jobs" not in st.session_state:
        st.session_state.jobs = load_jobs()
    if "user_profile" not in st.session_state:
        # Default with the first preset profile
        st.session_state.user_profile = PRESET_PROFILES["Full-Stack Web Developer (Mid-Level)"]
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = "Full-Stack Web Developer (Mid-Level)"


def main():
    init_session_state()
    jobs: List[JobListing] = st.session_state.jobs
    available_fields = get_available_fields(jobs)
    all_known_skills = get_all_skills(jobs)

    # -------------------------------------------------------------
    # Sidebar: Profile Configuration & Presets
    # -------------------------------------------------------------
    st.sidebar.markdown("## 👤 Candidate Profile")
    st.sidebar.caption("Input your details or select a preset to match against job listings.")

    # Preset selection
    preset_names = ["Custom Profile..."] + list(PRESET_PROFILES.keys())
    
    # Preset callback / selector
    preset_choice = st.sidebar.selectbox(
        "⚡ Quick-Fill Preset Profile",
        options=preset_names,
        index=preset_names.index(st.session_state.selected_preset) if st.session_state.selected_preset in preset_names else 0,
        help="Select a realistic candidate profile preset to instantly test matching."
    )

    if preset_choice != "Custom Profile..." and preset_choice != st.session_state.selected_preset:
        st.session_state.selected_preset = preset_choice
        preset = PRESET_PROFILES[preset_choice]
        st.session_state.user_profile = UserProfile(
            name=preset.name,
            headline=preset.headline,
            target_fields=list(preset.target_fields),
            skills=list(preset.skills),
            years_of_experience=preset.years_of_experience,
            education_level=preset.education_level,
            preferred_work_type=preset.preferred_work_type,
            min_salary=preset.min_salary,
            summary=preset.summary,
        )
        st.rerun()

    current_profile = st.session_state.user_profile

    # Sidebar Profile Form
    with st.sidebar.form("profile_form"):
        name_input = st.text_input("Full Name", value=current_profile.name)
        headline_input = st.text_input("Professional Headline", value=current_profile.headline)

        target_fields_input = st.multiselect(
            "Target Industry / Job Fields",
            options=available_fields,
            default=[f for f in current_profile.target_fields if f in available_fields],
            help="Select one or more job fields you are targeting, or leave blank to search all."
        )

        # Merge existing user skills with all catalog skills for suggestions
        combined_skill_options = sorted(list(set(all_known_skills + current_profile.skills)))
        skills_input = st.multiselect(
            "Skills & Proficiencies",
            options=combined_skill_options,
            default=current_profile.skills,
            help="Select or add your technical and professional skills."
        )

        # Add manual custom skill box if not in list
        custom_skills_raw = st.text_input(
            "Add Additional Skills (comma-separated)",
            value="",
            placeholder="e.g. GraphQL, Tailwind, PyTorch"
        )

        col_exp, col_edu = st.columns(2)
        with col_exp:
            exp_input = st.number_input(
                "Years of Exp",
                min_value=0.0,
                max_value=30.0,
                value=float(current_profile.years_of_experience),
                step=0.5,
            )
        with col_edu:
            edu_options = ["Bootcamp / Self-taught", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Ph.D."]
            edu_index = edu_options.index(current_profile.education_level) if current_profile.education_level in edu_options else 2
            edu_input = st.selectbox("Education Level", options=edu_options, index=edu_index)

        col_wt, col_sal = st.columns(2)
        with col_wt:
            wt_options = ["Any", "Remote", "Hybrid", "On-site"]
            wt_index = wt_options.index(current_profile.preferred_work_type) if current_profile.preferred_work_type in wt_options else 0
            wt_input = st.selectbox("Preferred Work Mode", options=wt_options, index=wt_index)
        with col_sal:
            salary_input = st.number_input(
                "Min Salary ($)",
                min_value=0,
                max_value=400000,
                value=int(current_profile.min_salary),
                step=5000,
            )

        summary_input = st.text_area(
            "Bio / Resume Summary",
            value=current_profile.summary,
            height=120,
            help="Paste a short summary of your background, experience, or career goals."
        )

        submitted = st.form_submit_button("💾 Update Profile & Match", use_container_width=True)
        if submitted:
            # Parse additional skills
            extra_skills = [s.strip() for s in custom_skills_raw.split(",") if s.strip()]
            all_user_skills = list(dict.fromkeys(skills_input + extra_skills))

            st.session_state.user_profile = UserProfile(
                name=name_input,
                headline=headline_input,
                target_fields=target_fields_input,
                skills=all_user_skills,
                years_of_experience=exp_input,
                education_level=edu_input,
                preferred_work_type=wt_input,
                min_salary=salary_input,
                summary=summary_input,
            )
            st.session_state.selected_preset = "Custom Profile..."
            st.rerun()

    # -------------------------------------------------------------
    # Main Panel: Header & Controls
    # -------------------------------------------------------------
    st.markdown('<div class="main-header">🎯 AI Job Search & Profile Match Engine</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Matching candidate <strong>{st.session_state.user_profile.name}</strong> ({st.session_state.user_profile.headline}) against active market listings.</div>',
        unsafe_allow_html=True,
    )

    # Filter Toolbar
    with st.expander("🛠️ Filters & Sorting Controls", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            field_filter = st.selectbox("Filter by Field", options=["All Fields"] + available_fields)
        with fcol2:
            wt_filter = st.selectbox("Filter by Work Mode", options=["All Types", "Remote", "Hybrid", "On-site"])
        with fcol3:
            exp_filter = st.selectbox("Filter by Experience Level", options=["All Levels", "Entry-Level", "Mid-Level", "Senior", "Lead"])
        with fcol4:
            min_score_slider = st.slider("Minimum Match Score (%)", min_value=0, max_value=100, value=0, step=5)

    # Rank and calculate matches
    results = rank_jobs(
        user_profile=st.session_state.user_profile,
        jobs=jobs,
        min_score=float(min_score_slider),
        field_filter=field_filter,
        work_type_filter=wt_filter,
        experience_filter=exp_filter,
    )

    # -------------------------------------------------------------
    # KPI Metrics Overview Row
    # -------------------------------------------------------------
    top_match = results[0] if results else None
    avg_score = sum(r.overall_score for r in results) / len(results) if results else 0.0
    field_summary = get_field_match_summary(results)
    best_field = max(field_summary.items(), key=lambda x: x[1]["avg_score"])[0] if field_summary else "N/A"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Jobs Evaluated</div>
                <div class="metric-val">{len(results)} <span style="font-size: 1rem; color: #94A3B8;">/ {len(jobs)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        top_score_str = f"{top_match.overall_score:.1f}%" if top_match else "0%"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Top Match Score</div>
                <div class="metric-val" style="color: #10B981;">{top_score_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Top Matching Domain</div>
                <div class="metric-val" style="font-size: 1.25rem; color: #3B82F6;">{best_field}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-lbl">Average Match Score</div>
                <div class="metric-val" style="color: #6366F1;">{avg_score:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Ranked Job Matches",
        "📊 Fit Analytics & Charts",
        "⚖️ Side-by-Side Comparison",
        "📂 Job Catalog & Add Job",
    ])

    # -------------------------------------------------------------
    # TAB 1: Ranked Job Matches
    # -------------------------------------------------------------
    with tab1:
        if not results:
            st.warning("⚠️ No job listings matched your current filter criteria. Try lowering the minimum score or adjusting filters.")
        else:
            st.markdown(f"### 📋 Top Ranked Job Matches ({len(results)} listings found)")
            for r in results:
                render_job_card(r, st.session_state.user_profile)

    # -------------------------------------------------------------
    # TAB 2: Fit Analytics & Charts
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### 📊 Candidate Fit Analytics & Insights")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            field_chart = plot_field_distribution(results)
            if field_chart:
                st.plotly_chart(field_chart, use_container_width=True)
            else:
                st.info("Not enough data to render domain distribution chart.")

        with col_c2:
            radar_chart = plot_top_jobs_radar(results, top_n=4)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
            else:
                st.info("Not enough data to render radar chart.")

        st.divider()

        # Skill Gap Recommendation Analysis
        st.markdown("#### 💡 Key In-Demand Skills for Top Matches")
        st.caption("Common skills required across your top matching jobs that you may want to acquire:")
        
        # Calculate missing skills frequency among top 5 jobs
        missing_freq: dict = {}
        for r in results[:5]:
            for s in r.breakdown.missing_skills:
                missing_freq[s] = missing_freq.get(s, 0) + 1
        
        if missing_freq:
            sorted_gaps = sorted(missing_freq.items(), key=lambda x: x[1], reverse=True)
            gap_cols = st.columns(min(len(sorted_gaps), 4))
            for i, (skill, count) in enumerate(sorted_gaps[:4]):
                with gap_cols[i]:
                    st.metric(label=f"Skill: {skill}", value=f"Required in {count} top role(s)")
        else:
            st.success("✨ Outstanding! You match all required skills across your top job opportunities.")

    # -------------------------------------------------------------
    # TAB 3: Side-by-Side Job Comparison
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### ⚖️ Side-by-Side Job Comparator")
        st.caption("Compare up to 3 jobs directly against your candidate profile.")

        if len(results) < 2:
            st.info("Please ensure at least 2 jobs are available to compare.")
        else:
            job_options = {f"#{r.rank} - {r.job.title} ({r.job.company}) [{r.overall_score:.1f}%]": r for r in results}
            default_selection = list(job_options.keys())[:min(3, len(job_options))]
            
            selected_job_keys = st.multiselect(
                "Select 2 or 3 jobs to compare:",
                options=list(job_options.keys()),
                default=default_selection,
                max_selections=3
            )

            if selected_job_keys:
                selected_results = [job_options[k] for k in selected_job_keys]
                comp_cols = st.columns(len(selected_results))

                for idx, r in enumerate(selected_results):
                    job = r.job
                    bd = r.breakdown
                    with comp_cols[idx]:
                        st.markdown(f"#### #{r.rank} {job.title}")
                        st.markdown(f"**{job.company}**")
                        st.metric("Overall Match", f"{r.overall_score:.1f}%", delta=r.match_tier)
                        
                        st.divider()
                        st.markdown(f"**Field**: {job.field}")
                        st.markdown(f"**Location**: {job.location} ({job.work_type})")
                        st.markdown(f"**Salary**: {job.salary_range}")
                        st.markdown(f"**Experience Req**: {job.experience_level} ({job.min_years_exp}+ yrs)")
                        st.markdown(f"**Education**: {job.education_required}")
                        
                        st.divider()
                        st.markdown("**Score Breakdown:**")
                        st.write(f"- Skills: `{bd.skill_score:.1f}%`")
                        st.write(f"- Role/Text Fit: `{bd.text_score:.1f}%`")
                        st.write(f"- Experience: `{bd.experience_score:.1f}%`")
                        st.write(f"- Preferences: `{bd.preference_score:.1f}%`")

                        st.divider()
                        st.markdown(f"**Matched Skills ({len(bd.matched_skills)}):**")
                        st.write(", ".join(bd.matched_skills) if bd.matched_skills else "None")
                        
                        st.markdown(f"**Skill Gaps ({len(bd.missing_skills)}):**")
                        st.write(", ".join(bd.missing_skills) if bd.missing_skills else "None")

    # -------------------------------------------------------------
    # TAB 4: Job Catalog & Add New Job Listing
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### 📂 Complete Job Catalog")
        
        # Display DataFrame
        table_data = []
        for j in jobs:
            table_data.append({
                "ID": j.id,
                "Title": j.title,
                "Company": j.company,
                "Field": j.field,
                "Location": j.location,
                "Work Mode": j.work_type,
                "Level": j.experience_level,
                "Min Yrs": j.min_years_exp,
                "Salary Range": j.salary_range,
                "Required Skills": ", ".join(j.required_skills),
            })
        df_jobs = pd.DataFrame(table_data)
        st.dataframe(df_jobs, use_container_width=True, hide_index=True)

        st.divider()

        # Add New Job Form
        st.markdown("### ➕ Add a New Job Listing to Catalog")
        with st.form("add_job_form", clear_on_submit=True):
            aj_c1, aj_c2 = st.columns(2)
            with aj_c1:
                new_title = st.text_input("Job Title", placeholder="e.g. Senior Backend Engineer")
                new_company = st.text_input("Company Name", placeholder="e.g. Acme Cloud Corp")
                new_field = st.text_input("Job Field / Domain", placeholder="e.g. Software Engineering")
                new_location = st.text_input("Location", placeholder="e.g. San Francisco, CA")
                new_work_type = st.selectbox("Work Type", ["Remote", "Hybrid", "On-site"])
            with aj_c2:
                new_level = st.selectbox("Experience Level", ["Entry-Level", "Mid-Level", "Senior", "Lead"])
                new_min_years = st.number_input("Min Years Experience", min_value=0, max_value=20, value=2)
                new_salary = st.text_input("Salary Range", placeholder="e.g. $120,000 - $150,000")
                new_req_skills = st.text_input("Required Skills (comma-separated)", placeholder="e.g. Python, Docker, PostgreSQL")
                new_pref_skills = st.text_input("Preferred Skills (comma-separated)", placeholder="e.g. AWS, Kubernetes")

            new_desc = st.text_area("Job Description", placeholder="Detailed role overview...")
            new_resps = st.text_area("Responsibilities (one per line)", placeholder="Develop backend services\nOptimize databases")

            add_submitted = st.form_submit_button("🚀 Add Job Listing", use_container_width=True)
            if add_submitted:
                if new_title and new_company and new_field:
                    new_id = f"JOB-{len(jobs) + 1:03d}"
                    req_list = [s.strip() for s in new_req_skills.split(",") if s.strip()]
                    pref_list = [s.strip() for s in new_pref_skills.split(",") if s.strip()]
                    resp_list = [r.strip() for r in new_resps.split("\n") if r.strip()]

                    created_job = JobListing(
                        id=new_id,
                        title=new_title,
                        company=new_company,
                        field=new_field,
                        location=new_location or "Remote",
                        work_type=new_work_type,
                        experience_level=new_level,
                        min_years_exp=int(new_min_years),
                        salary_range=new_salary or "Competitive",
                        min_salary=100000,
                        max_salary=150000,
                        education_required="Bachelor's Degree",
                        required_skills=req_list,
                        preferred_skills=pref_list,
                        description=new_desc or f"Exciting opportunity for {new_title} at {new_company}.",
                        responsibilities=resp_list or ["Contribute to core company goals."],
                    )
                    st.session_state.jobs.append(created_job)
                    save_jobs(st.session_state.jobs)
                    st.success(f"✅ Successfully added job '{new_title}' (ID: {new_id}) to the catalog!")
                    st.rerun()
                else:
                    st.error("Please fill in at least Job Title, Company, and Field.")


if __name__ == "__main__":
    main()

