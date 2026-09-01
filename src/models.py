"""
Data models for the Job Search and Candidate Profile Matching application.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class UserProfile:
    """Represents a candidate/user profile."""
    name: str = "Anonymous Candidate"
    headline: str = "Software Professional"
    target_fields: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    years_of_experience: float = 2.0
    education_level: str = "Bachelor's Degree"
    preferred_work_type: str = "Any"  # "Remote", "Hybrid", "On-site", "Any"
    min_salary: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "headline": self.headline,
            "target_fields": self.target_fields,
            "skills": self.skills,
            "years_of_experience": self.years_of_experience,
            "education_level": self.education_level,
            "preferred_work_type": self.preferred_work_type,
            "min_salary": self.min_salary,
            "summary": self.summary,
        }


@dataclass
class JobListing:
    """Represents a single job posting."""
    id: str
    title: str
    company: str
    field: str
    location: str
    work_type: str
    experience_level: str
    min_years_exp: int
    salary_range: str
    min_salary: int
    max_salary: int
    education_required: str
    required_skills: List[str]
    preferred_skills: List[str]
    description: str
    responsibilities: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobListing":
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            company=data.get("company", ""),
            field=data.get("field", ""),
            location=data.get("location", ""),
            work_type=data.get("work_type", "Hybrid"),
            experience_level=data.get("experience_level", "Mid-Level"),
            min_years_exp=int(data.get("min_years_exp", 0)),
            salary_range=data.get("salary_range", "Competitive"),
            min_salary=int(data.get("min_salary", 0)),
            max_salary=int(data.get("max_salary", 0)),
            education_required=data.get("education_required", "Bachelor's Degree"),
            required_skills=list(data.get("required_skills", [])),
            preferred_skills=list(data.get("preferred_skills", [])),
            description=data.get("description", ""),
            responsibilities=list(data.get("responsibilities", [])),
        )


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of matching score components."""
    skill_score: float = 0.0          # 0-100
    text_score: float = 0.0           # 0-100
    experience_score: float = 0.0     # 0-100
    preference_score: float = 0.0     # 0-100
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    matched_preferred_skills: List[str] = field(default_factory=list)
    experience_status: str = "Compatible"
    field_match: bool = False
    work_type_match: bool = False


@dataclass
class JobMatchResult:
    """Result of matching a UserProfile against a JobListing."""
    job: JobListing
    overall_score: float
    rank: int = 1
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    match_tier: str = "Moderate Match"
    match_summary: str = ""

    @staticmethod
    def calculate_tier(score: float) -> str:
        if score >= 80.0:
            return "Strong Match"
        elif score >= 65.0:
            return "Good Match"
        elif score >= 45.0:
            return "Moderate Match"
        else:
            return "Low Match"

