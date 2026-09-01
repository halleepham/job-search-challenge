"""
Unit tests for the job matching and scoring engine.
"""

import pytest
from src.models import UserProfile, JobListing, JobMatchResult
from src.matcher import (
    normalize_skill,
    calculate_skill_score,
    calculate_text_similarity,
    calculate_experience_score,
    calculate_preference_score,
    evaluate_job_match,
    rank_jobs,
    get_field_match_summary,
)
from src.data_loader import load_jobs, PRESET_PROFILES


@pytest.fixture
def sample_jobs():
    return [
        JobListing(
            id="JOB-TEST-1",
            title="Senior Python Backend Developer",
            company="Tech Corp",
            field="Software Engineering",
            location="Remote",
            work_type="Remote",
            experience_level="Senior",
            min_years_exp=5,
            salary_range="$140,000 - $180,000",
            min_salary=140000,
            max_salary=180000,
            education_required="Bachelor's Degree",
            required_skills=["Python", "PostgreSQL", "Docker", "REST APIs"],
            preferred_skills=["AWS", "Redis"],
            description="Looking for an experienced Python developer to build robust backend APIs and distributed systems.",
            responsibilities=["Design RESTful endpoints", "Optimize database queries"],
        ),
        JobListing(
            id="JOB-TEST-2",
            title="Junior Frontend React Developer",
            company="Web Studio",
            field="Software Engineering",
            location="New York, NY",
            work_type="Hybrid",
            experience_level="Entry-Level",
            min_years_exp=1,
            salary_range="$70,000 - $90,000",
            min_salary=70000,
            max_salary=90000,
            education_required="Bachelor's Degree",
            required_skills=["React", "JavaScript", "HTML5", "CSS3"],
            preferred_skills=["TypeScript", "Tailwind CSS"],
            description="Entry level role for front-end web development with React and modern JavaScript.",
            responsibilities=["Develop UI components in React", "Ensure responsive web layouts"],
        ),
        JobListing(
            id="JOB-TEST-3",
            title="Lead Data Scientist",
            company="AI Innovations",
            field="Data Science & AI",
            location="Boston, MA",
            work_type="On-site",
            experience_level="Lead",
            min_years_exp=6,
            salary_range="$170,000 - $220,000",
            min_salary=170000,
            max_salary=220000,
            education_required="Master's Degree",
            required_skills=["Python", "PyTorch", "Machine Learning", "Deep Learning"],
            preferred_skills=["Transformers", "MLOps"],
            description="Lead machine learning research and train advanced neural networks.",
            responsibilities=["Train PyTorch deep learning models", "Publish experimental research"],
        )
    ]


def test_normalize_skill():
    assert normalize_skill("React.js") == "react"
    assert normalize_skill(" NodeJS ") == "node.js"
    assert normalize_skill("Postgres") == "postgresql"
    assert normalize_skill("K8s") == "kubernetes"
    assert normalize_skill("Python3") == "python"


def test_calculate_skill_score_perfect():
    user_skills = ["Python", "PostgreSQL", "Docker", "REST APIs", "AWS", "Redis"]
    required = ["Python", "PostgreSQL", "Docker", "REST APIs"]
    preferred = ["AWS", "Redis"]
    
    score, matched_req, missing_req, matched_pref = calculate_skill_score(
        user_skills, required, preferred
    )
    assert score == 100.0
    assert len(matched_req) == 4
    assert len(missing_req) == 0
    assert len(matched_pref) == 2


def test_calculate_skill_score_partial():
    user_skills = ["Python", "PostgreSQL"]
    required = ["Python", "PostgreSQL", "Docker", "REST APIs"]
    preferred = ["AWS", "Redis"]
    
    score, matched_req, missing_req, matched_pref = calculate_skill_score(
        user_skills, required, preferred
    )
    # 2 out of 4 required = 50% of 85 = 42.5; 0 of 2 pref = 0; total = 42.5
    assert score == 42.5
    assert len(matched_req) == 2
    assert "Docker" in missing_req
    assert "REST APIs" in missing_req


def test_calculate_experience_score():
    # Optimal match
    score, status = calculate_experience_score(user_years=5.0, job_min_years=5)
    assert score == 100.0
    assert "Optimal" in status

    # Surplus experience
    score, status = calculate_experience_score(user_years=8.0, job_min_years=5)
    assert score == 90.0
    assert "Exceeds" in status

    # Gap of 1 year
    score, status = calculate_experience_score(user_years=4.0, job_min_years=5)
    assert score == 75.0
    assert "Close" in status

    # Gap of 3 years
    score, status = calculate_experience_score(user_years=2.0, job_min_years=5)
    assert score <= 50.0


def test_calculate_preference_score():
    profile = UserProfile(
        target_fields=["Software Engineering"],
        preferred_work_type="Remote"
    )
    job_remote = JobListing(
        id="1", title="Dev", company="A", field="Software Engineering",
        location="Remote", work_type="Remote", experience_level="Mid",
        min_years_exp=2, salary_range="$100k", min_salary=100000, max_salary=120000,
        education_required="BA", required_skills=[], preferred_skills=[],
        description="", responsibilities=[]
    )
    score, field_match, wt_match = calculate_preference_score(profile, job_remote)
    assert score == 100.0  # 60 (field) + 40 (work type)
    assert field_match is True
    assert wt_match is True

    # Mismatched field & work type
    job_other = JobListing(
        id="2", title="Analyst", company="B", field="Finance",
        location="NY", work_type="On-site", experience_level="Mid",
        min_years_exp=2, salary_range="$100k", min_salary=100000, max_salary=120000,
        education_required="BA", required_skills=[], preferred_skills=[],
        description="", responsibilities=[]
    )
    score_other, field_match_other, wt_match_other = calculate_preference_score(profile, job_other)
    assert score_other < 50.0
    assert field_match_other is False
    assert wt_match_other is False


def test_rank_jobs_order(sample_jobs):
    # Candidate specifically targeting Python backend
    profile = UserProfile(
        name="Senior Pythonist",
        headline="Senior Python Backend Developer",
        target_fields=["Software Engineering"],
        skills=["Python", "PostgreSQL", "Docker", "REST APIs", "AWS"],
        years_of_experience=5.5,
        education_level="Bachelor's Degree",
        preferred_work_type="Remote",
        min_salary=130000,
        summary="Senior backend developer with 5+ years writing Python APIs and Docker microservices."
    )

    ranked = rank_jobs(profile, sample_jobs)
    assert len(ranked) == 3
    # Top ranked should be Senior Python Backend Developer (JOB-TEST-1)
    assert ranked[0].job.id == "JOB-TEST-1"
    assert ranked[0].rank == 1
    assert ranked[0].overall_score > ranked[1].overall_score
    assert ranked[0].overall_score > 75.0


def test_rank_jobs_filtering(sample_jobs):
    profile = PRESET_PROFILES["Machine Learning & AI Specialist (Senior)"]
    
    # Filter by Field
    ai_only = rank_jobs(profile, sample_jobs, field_filter="Data Science & AI")
    assert len(ai_only) == 1
    assert ai_only[0].job.field == "Data Science & AI"

    # Filter by Min Score threshold
    high_matches = rank_jobs(profile, sample_jobs, min_score=60.0)
    for r in high_matches:
        assert r.overall_score >= 60.0


def test_field_match_summary(sample_jobs):
    profile = PRESET_PROFILES["Full-Stack Web Developer (Mid-Level)"]
    ranked = rank_jobs(profile, sample_jobs)
    summary = get_field_match_summary(ranked)
    assert "Software Engineering" in summary
    assert summary["Software Engineering"]["count"] == 2
    assert summary["Software Engineering"]["avg_score"] > 0


def test_tier_calculation():
    assert JobMatchResult.calculate_tier(95.0) == "Strong Match"
    assert JobMatchResult.calculate_tier(80.0) == "Strong Match"
    assert JobMatchResult.calculate_tier(79.9) == "Good Match"
    assert JobMatchResult.calculate_tier(65.0) == "Good Match"
    assert JobMatchResult.calculate_tier(55.0) == "Moderate Match"
    assert JobMatchResult.calculate_tier(30.0) == "Low Match"


def test_text_similarity_edge_cases():
    profile_empty = UserProfile(headline="", summary="", skills=[], target_fields=[])
    job_empty = JobListing(
        id="E1", title="", company="", field="", location="", work_type="Remote",
        experience_level="Mid", min_years_exp=0, salary_range="", min_salary=0,
        max_salary=0, education_required="", required_skills=[], preferred_skills=[],
        description="", responsibilities=[]
    )
    assert calculate_text_similarity(profile_empty, job_empty) == 0.0

    profile_dev = UserProfile(
        headline="Senior React Developer",
        summary="Building scalable modern web apps in React and TypeScript.",
        skills=["React", "TypeScript"],
        target_fields=["Software Engineering"]
    )
    job_dev = JobListing(
        id="E2", title="Senior React Developer", company="WebCo", field="Software Engineering",
        location="Remote", work_type="Remote", experience_level="Senior", min_years_exp=3,
        salary_range="$120k", min_salary=120000, max_salary=140000,
        education_required="BA", required_skills=["React", "TypeScript"], preferred_skills=[],
        description="We are looking for a Senior React Developer to build scalable web applications.",
        responsibilities=["Develop React components", "Optimize TypeScript code"]
    )
    sim_score = calculate_text_similarity(profile_dev, job_dev)
    assert sim_score > 60.0


def test_empty_user_skills():
    score, matched_req, missing_req, matched_pref = calculate_skill_score(
        user_skills=[],
        required_skills=["Python", "SQL"],
        preferred_skills=["AWS"]
    )
    assert score == 0.0
    assert len(matched_req) == 0
    assert len(missing_req) == 2


def test_load_dataset_integrity():
    jobs = load_jobs()
    assert len(jobs) >= 20
    for j in jobs:
        assert j.id.startswith("JOB-")
        assert len(j.title) > 0
        assert len(j.field) > 0
        assert len(j.required_skills) > 0
        assert j.min_years_exp >= 0


