"""
Reference profiles, used by two requirements.

AC-5.5/AC-5.6: the background distribution for semantic calibration is measured
against these, and **at least one must be materially unlike the author's own** so
the calibration is not tuned to a single person.

AC-3.7: the same profiles ship as the UI's preset profiles, and supply the
"top-5 for different profiles" evaluation in AC-13.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserProfile:
    name: str
    career_goals: str
    resume_text: str
    skills: set[str] = field(default_factory=set)
    years_experience: float = 0.0
    highest_completed_education: int = 3          # ordinal ladder, AC-9.6
    education_in_progress: int | None = None
    preferred_titles: list[str] = field(default_factory=list)
    preferred_location: str | None = None
    accepted_work_settings: set[str] = field(
        default_factory=lambda: {"Remote", "Hybrid", "On-site"})
    accepted_employment_types: set[str] = field(
        default_factory=lambda: {"Full-time", "Contract", "Part-time",
                                 "Temporary", "Internship"})
    min_salary: int = 0
    include_unlisted_salary: bool = True
    max_distance_miles: int = 100


PRESETS: dict[str, UserProfile] = {
    "data_science_student": UserProfile(
        name="Data science master's student",
        career_goals=("I want a data engineering or data science role building pipelines "
                      "that turn raw data into dashboards and models."),
        resume_text=("Education\nM.S. Data Science, expected 2026. B.S. Mathematics.\n\n"
                     "Skills\nPython, SQL, Spark, Airflow, AWS, Tableau, dbt.\n\n"
                     "Experience\nData Engineering Intern. Built ETL pipelines in Airflow "
                     "moving 40M rows/day into Snowflake. Wrote dbt models and tests.\n\n"
                     "Projects\nHousing price model with scikit-learn and XGBoost. "
                     "Streamlit dashboard deployed on AWS."),
        skills={"python", "sql", "spark", "airflow", "aws", "tableau", "dbt",
                "snowflake", "scikit-learn", "streamlit"},
        years_experience=1.0,
        highest_completed_education=3,
        education_in_progress=4,
        preferred_titles=["data engineer", "data scientist", "data analyst"],
        preferred_location="Kansas City, MO",
        min_salary=90000,
    ),
    # AC-5.6: materially unlike the author - different domain, seniority, geography.
    "senior_backend_engineer": UserProfile(
        name="Senior backend engineer",
        career_goals=("I build distributed backend services and want to lead platform "
                      "work at scale."),
        resume_text=("Experience\nSenior Software Engineer, 9 years. Java and Go "
                     "microservices on Kubernetes serving 2M requests/minute.\n\n"
                     "Skills\nJava, Go, Kubernetes, Terraform, PostgreSQL, Kafka, gRPC.\n\n"
                     "Education\nB.S. Computer Science."),
        skills={"java", "go", "kubernetes", "terraform", "postgresql", "kafka",
                "grpc", "docker", "aws", "microservices"},
        years_experience=9.0,
        highest_completed_education=3,
        preferred_titles=["backend engineer", "platform engineer", "software engineer"],
        preferred_location="Seattle, WA",
        min_salary=170000,
        accepted_work_settings={"Remote"},
    ),
    # AC-5.6: a third domain again - security, not software or data.
    "security_analyst": UserProfile(
        name="Information security analyst",
        career_goals="I want to work in threat detection and incident response.",
        resume_text=("Experience\nSecurity Analyst, 4 years in a SOC. Investigated "
                     "alerts in Splunk, tuned SIEM detections, led incident response.\n\n"
                     "Skills\nSplunk, SIEM, incident response, Wireshark, Nessus, Python.\n\n"
                     "Certifications\nCISSP.\n\nEducation\nB.S. Information Technology."),
        skills={"splunk", "siem", "incident response", "wireshark", "nessus",
                "python", "cissp", "firewall", "iam"},
        years_experience=4.0,
        highest_completed_education=3,
        preferred_titles=["security analyst", "soc analyst"],
        preferred_location="Washington, DC",
        min_salary=110000,
        accepted_work_settings={"Hybrid", "On-site"},
    ),
}

#: AC-5.5: the profiles whose similarity distribution defines the calibration
#: background. Uses all three so no single person's vocabulary dominates.
CALIBRATION_PROFILES = tuple(PRESETS)
