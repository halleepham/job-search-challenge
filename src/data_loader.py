"""
Data loading and preset profile management utilities.
"""

import json
import os
from typing import List, Dict, Any, Optional
from src.models import JobListing, UserProfile

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "jobs.json"
)

PRESET_PROFILES: Dict[str, UserProfile] = {
    "Full-Stack Web Developer (Mid-Level)": UserProfile(
        name="Alex Chen",
        headline="Full-Stack Developer | React & Python",
        target_fields=["Software Engineering"],
        skills=["Python", "JavaScript", "React", "Node.js", "PostgreSQL", "Docker", "Git", "REST APIs"],
        years_of_experience=3.5,
        education_level="Bachelor's Degree",
        preferred_work_type="Hybrid",
        min_salary=115000,
        summary="Energetic full-stack software engineer with 3.5 years of experience building modern React frontends and scalable Python/Node backends. Passionate about responsive UI, RESTful APIs, and database performance optimization."
    ),
    "Machine Learning & AI Specialist (Senior)": UserProfile(
        name="Dr. Maya Patel",
        headline="Senior AI / ML Research Engineer",
        target_fields=["Data Science & AI"],
        skills=["Python", "PyTorch", "Deep Learning", "NLP", "Machine Learning", "Transformers", "MLOps", "Kubernetes", "SQL"],
        years_of_experience=6.0,
        education_level="Master's Degree",
        preferred_work_type="Remote",
        min_salary=160000,
        summary="Senior Machine Learning engineer with 6 years leading NLP model training, LLM fine-tuning, and scalable distributed AI inference pipelines using PyTorch, Transformers, and MLOps tooling."
    ),
    "UI/UX Product Designer (Mid-Level)": UserProfile(
        name="Jordan Rivera",
        headline="Lead UI/UX & Design Systems Specialist",
        target_fields=["Product & Design"],
        skills=["Figma", "UI/UX Design", "Wireframing", "User Research", "Prototyping", "Design Systems", "Usability Testing"],
        years_of_experience=3.0,
        education_level="Bachelor's Degree",
        preferred_work_type="Remote",
        min_salary=100000,
        summary="Product designer focused on user-centric design systems, rapid prototyping in Figma, and conducting qualitative user research sessions to deliver intuitive SaaS interfaces."
    ),
    "Cloud & DevOps Infrastructure Engineer (Senior)": UserProfile(
        name="Marcus Vance",
        headline="Senior DevOps & SRE Engineer",
        target_fields=["DevOps & Cloud"],
        skills=["AWS", "Kubernetes", "Docker", "Terraform", "CI/CD", "Linux", "Python", "Prometheus", "Grafana"],
        years_of_experience=5.5,
        education_level="Bachelor's Degree",
        preferred_work_type="Remote",
        min_salary=145000,
        summary="DevOps and Site Reliability Engineer with over 5 years architecting highly available Kubernetes clusters on AWS, automated CI/CD pipelines, and infrastructure-as-code using Terraform."
    ),
    "Entry-Level Junior Software Developer": UserProfile(
        name="Taylor Smith",
        headline="Junior Frontend / Full-Stack Developer",
        target_fields=["Software Engineering"],
        skills=["HTML5", "CSS3", "JavaScript", "React", "Git", "Responsive Design", "Python"],
        years_of_experience=0.5,
        education_level="Bachelor's Degree",
        preferred_work_type="Any",
        min_salary=60000,
        summary="Recent Computer Science graduate with strong foundation in JavaScript, React, and HTML/CSS. Eager to contribute to web applications, write unit tests, and learn modern software engineering practices."
    ),
    "Cybersecurity Analyst (Mid-Level)": UserProfile(
        name="Samira Khan",
        headline="Cybersecurity & Incident Response Analyst",
        target_fields=["Cybersecurity"],
        skills=["Cybersecurity", "SIEM", "Network Security", "Incident Response", "Vulnerability Assessment", "Linux", "Wireshark", "Splunk"],
        years_of_experience=3.0,
        education_level="Bachelor's Degree",
        preferred_work_type="Hybrid",
        min_salary=105000,
        summary="Security analyst experienced in continuous threat monitoring, SOC incident triage, network vulnerability scanning with Wireshark, and log analysis in Splunk."
    )
}


def load_jobs(file_path: Optional[str] = None) -> List[JobListing]:
    """Loads job listings from a JSON file."""
    path = file_path or DEFAULT_DATA_PATH
    if not os.path.exists(path):
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [JobListing.from_dict(item) for item in data]


def save_jobs(jobs: List[JobListing], file_path: Optional[str] = None) -> None:
    """Saves job listings to a JSON file."""
    path = file_path or DEFAULT_DATA_PATH
    data = []
    for job in jobs:
        data.append({
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "field": job.field,
            "location": job.location,
            "work_type": job.work_type,
            "experience_level": job.experience_level,
            "min_years_exp": job.min_years_exp,
            "salary_range": job.salary_range,
            "min_salary": job.min_salary,
            "max_salary": job.max_salary,
            "education_required": job.education_required,
            "required_skills": job.required_skills,
            "preferred_skills": job.preferred_skills,
            "description": job.description,
            "responsibilities": job.responsibilities,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_available_fields(jobs: List[JobListing]) -> List[str]:
    """Returns a sorted list of unique job fields/categories."""
    fields = set(job.field for job in jobs if job.field)
    return sorted(list(fields))


def get_all_skills(jobs: List[JobListing]) -> List[str]:
    """Returns a sorted list of unique skills from all job listings."""
    skills = set()
    for job in jobs:
        skills.update(s.strip() for s in job.required_skills)
        skills.update(s.strip() for s in job.preferred_skills)
    return sorted(list(skills))

