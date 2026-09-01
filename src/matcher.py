"""
Matching and ranking engine for comparing user profiles against job listings.
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import UserProfile, JobListing, JobMatchResult, ScoreBreakdown


# Common skill aliases for fuzzy normalization
SKILL_ALIASES: Dict[str, str] = {
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "ts": "typescript",
    "js": "javascript",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "k8s": "kubernetes",
    "py": "python",
    "python3": "python",
    "golang": "go",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "llm": "llms",
    "large language models": "llms",
    "tf": "tensorflow",
    "ui": "ui/ux design",
    "ux": "ui/ux design",
    "ui/ux": "ui/ux design",
    "css": "css3",
    "html": "html5",
    "ci-cd": "ci/cd",
    "cicd": "ci/cd",
}


def normalize_skill(skill: str) -> str:
    """Normalize skill string by lowering, trimming, and applying alias mappings."""
    s = skill.strip().lower()
    # Remove punctuation except special ones in programming (e.g. c++, c#, .js, /)
    s = re.sub(r"[^\w\+\#\.\/\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return SKILL_ALIASES.get(s, s)


def match_skill_presence(user_skills_norm: Set[str], job_skill: str) -> bool:
    """Checks if a job skill matches any user skill with exact or substring inclusion."""
    job_s_norm = normalize_skill(job_skill)
    if job_s_norm in user_skills_norm:
        return True
    
    # Substring / partial match check
    for u_skill in user_skills_norm:
        if len(u_skill) >= 3 and len(job_s_norm) >= 3:
            if u_skill in job_s_norm or job_s_norm in u_skill:
                return True
    return False


def calculate_skill_score(
    user_skills: List[str],
    required_skills: List[str],
    preferred_skills: List[str]
) -> Tuple[float, List[str], List[str], List[str]]:
    """
    Computes skill score (0-100), matched required skills, missing required skills,
    and matched preferred skills.
    """
    user_skills_norm = {normalize_skill(s) for s in user_skills if s.strip()}
    
    matched_req = []
    missing_req = []
    for req in required_skills:
        if match_skill_presence(user_skills_norm, req):
            matched_req.append(req)
        else:
            missing_req.append(req)
            
    matched_pref = []
    for pref in preferred_skills:
        if match_skill_presence(user_skills_norm, pref):
            matched_pref.append(pref)

    # Required skills score (85% of skill score)
    req_ratio = len(matched_req) / len(required_skills) if required_skills else 1.0
    
    # Preferred skills score (15% bonus)
    pref_ratio = len(matched_pref) / len(preferred_skills) if preferred_skills else 1.0
    
    score = (req_ratio * 85.0) + (pref_ratio * 15.0)
    score = min(100.0, max(0.0, score))
    
    return round(score, 1), matched_req, missing_req, matched_pref


def calculate_text_similarity(user_profile: UserProfile, job: JobListing) -> float:
    """
    Calculates TF-IDF cosine similarity between candidate profile and job listing text.
    Returns score scaled 0-100.
    """
    user_corpus = f"{user_profile.headline} {' '.join(user_profile.target_fields)} {' '.join(user_profile.skills)} {user_profile.summary}".strip()
    job_corpus = f"{job.title} {job.field} {' '.join(job.required_skills)} {' '.join(job.preferred_skills)} {job.description} {' '.join(job.responsibilities)}".strip()
    
    if not user_corpus or not job_corpus:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([user_corpus, job_corpus])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        # Rescale cosine similarity for realistic representation (typical cosine sim is 0.1 to 0.7)
        # We apply a smooth sigmoid-like multiplier so strong matches reach 85-95%
        scaled_score = min(100.0, sim * 140.0)
        return round(scaled_score, 1)
    except Exception:
        return 50.0


def calculate_experience_score(user_years: float, job_min_years: int) -> Tuple[float, str]:
    """
    Calculates experience alignment score (0-100) and status description.
    """
    if user_years >= job_min_years:
        surplus = user_years - job_min_years
        if surplus <= 2.0:
            return 100.0, "Optimal Experience Match"
        elif surplus <= 5.0:
            return 90.0, f"Exceeds Requirements (+{surplus:.1f} yrs)"
        else:
            return 85.0, f"Significantly Experienced (+{surplus:.1f} yrs)"
    else:
        gap = job_min_years - user_years
        if gap <= 1.0:
            return 75.0, f"Close Match (-{gap:.1f} yr required)"
        elif gap <= 2.5:
            return 50.0, f"Under Requirements (-{gap:.1f} yrs)"
        else:
            penalty_score = max(15.0, 50.0 - (gap * 10.0))
            return round(penalty_score, 1), f"Significant Experience Gap (-{gap:.1f} yrs)"


def calculate_preference_score(user_profile: UserProfile, job: JobListing) -> Tuple[float, bool, bool]:
    """
    Calculates preference score (0-100) based on Field and Work Type alignment.
    """
    field_match = False
    if not user_profile.target_fields:
        # Open to all fields
        field_score = 50.0
        field_match = True
    elif job.field in user_profile.target_fields:
        field_score = 60.0
        field_match = True
    else:
        field_score = 15.0
        field_match = False

    user_wt = (user_profile.preferred_work_type or "Any").lower()
    job_wt = (job.work_type or "Hybrid").lower()

    if user_wt in ["any", "all", ""]:
        wt_score = 40.0
        wt_match = True
    elif user_wt == job_wt:
        wt_score = 40.0
        wt_match = True
    elif user_wt == "remote" and job_wt == "hybrid":
        wt_score = 20.0
        wt_match = False
    elif user_wt == "remote" and job_wt == "on-site":
        wt_score = 5.0
        wt_match = False
    elif user_wt in ["hybrid", "on-site"] and job_wt in ["hybrid", "remote", "on-site"]:
        wt_score = 30.0
        wt_match = True
    else:
        wt_score = 15.0
        wt_match = False

    total_pref = field_score + wt_score
    return round(min(100.0, max(0.0, total_pref)), 1), field_match, wt_match


def evaluate_job_match(
    user_profile: UserProfile,
    job: JobListing,
    skill_weight: float = 0.40,
    text_weight: float = 0.30,
    exp_weight: float = 0.15,
    pref_weight: float = 0.15
) -> JobMatchResult:
    """
    Evaluates a single job listing against a user profile and returns a JobMatchResult.
    """
    # 1. Skill Score
    skill_score, matched_req, missing_req, matched_pref = calculate_skill_score(
        user_profile.skills, job.required_skills, job.preferred_skills
    )

    # 2. Text Similarity Score
    text_score = calculate_text_similarity(user_profile, job)

    # 3. Experience Score
    exp_score, exp_status = calculate_experience_score(
        user_profile.years_of_experience, job.min_years_exp
    )

    # 4. Preference Score
    pref_score, field_match, wt_match = calculate_preference_score(user_profile, job)

    # Composite Overall Score (0-100%)
    overall = (
        (skill_score * skill_weight) +
        (text_score * text_weight) +
        (exp_score * exp_weight) +
        (pref_score * pref_weight)
    )
    overall = round(min(100.0, max(0.0, overall)), 1)

    tier = JobMatchResult.calculate_tier(overall)

    # Construct narrative summary
    summary_parts = []
    req_total = len(job.required_skills)
    matched_cnt = len(matched_req)
    
    if matched_cnt == req_total and req_total > 0:
        summary_parts.append(f"Complete match for all {req_total} required skills.")
    elif matched_cnt > 0:
        summary_parts.append(f"Matches {matched_cnt}/{req_total} required skills ({', '.join(matched_req[:3])}).")
    else:
        summary_parts.append("Few matching required technical skills.")

    if field_match:
        summary_parts.append(f"Aligns with target field '{job.field}'.")
    
    summary_parts.append(exp_status)
    match_summary = " ".join(summary_parts)

    breakdown = ScoreBreakdown(
        skill_score=skill_score,
        text_score=text_score,
        experience_score=exp_score,
        preference_score=pref_score,
        matched_skills=matched_req,
        missing_skills=missing_req,
        matched_preferred_skills=matched_pref,
        experience_status=exp_status,
        field_match=field_match,
        work_type_match=wt_match,
    )

    return JobMatchResult(
        job=job,
        overall_score=overall,
        rank=1,
        breakdown=breakdown,
        match_tier=tier,
        match_summary=match_summary,
    )


def rank_jobs(
    user_profile: UserProfile,
    jobs: List[JobListing],
    min_score: float = 0.0,
    field_filter: Optional[str] = None,
    work_type_filter: Optional[str] = None,
    experience_filter: Optional[str] = None
) -> List[JobMatchResult]:
    """
    Evaluates, filters, and ranks all jobs for a given user profile.
    Returns a sorted list of JobMatchResult in descending score order.
    """
    results: List[JobMatchResult] = []

    for job in jobs:
        # Apply pre-filters if set
        if field_filter and field_filter != "All Fields" and job.field != field_filter:
            continue
        if work_type_filter and work_type_filter != "All Types" and job.work_type != work_type_filter:
            continue
        if experience_filter and experience_filter != "All Levels" and job.experience_level != experience_filter:
            continue

        result = evaluate_job_match(user_profile, job)
        
        if result.overall_score >= min_score:
            results.append(result)

    # Sort descending by overall score, then by skill score
    results.sort(key=lambda r: (r.overall_score, r.breakdown.skill_score), reverse=True)

    # Assign ranks
    for idx, r in enumerate(results, start=1):
        r.rank = idx

    return results


def get_field_match_summary(results: List[JobMatchResult]) -> Dict[str, Dict[str, Any]]:
    """
    Summarizes average match score and count per field.
    """
    field_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        f = r.job.field
        if f not in field_stats:
            field_stats[f] = {"count": 0, "total_score": 0.0, "scores": []}
        field_stats[f]["count"] += 1
        field_stats[f]["total_score"] += r.overall_score
        field_stats[f]["scores"].append(r.overall_score)

    summary = {}
    for f, stats in field_stats.items():
        avg = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0
        summary[f] = {
            "count": stats["count"],
            "avg_score": round(avg, 1),
            "max_score": round(max(stats["scores"]), 1) if stats["scores"] else 0.0,
        }
    return summary

