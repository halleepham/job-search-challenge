"""
REQ-9: weighted match score components (DRAFT).

Every component is a pure function over typed inputs - no app, no model, no
database - so each rule is unit-testable in isolation and every number shown to
the user is traceable to one of them.
"""

from __future__ import annotations

#: AC-9.3: minimum stated requirements for an overlap ratio to carry full weight.
#: Below this the ratio is scaled down, because a ratio computed from one
#: observation is weaker evidence than the same ratio computed from ten.
MIN_SKILLS_FOR_FULL_CREDIT = 3


def required_skill_overlap(user_skills: set[str], required_skills: set[str]) -> float:
    """
    AC-9.3: matched fraction of a posting's required skills, damped when the
    posting states very few.

    ``ratio x min(1, |required| / 3)``. A posting stating one requirement caps at
    0.33, two at 0.67, three or more is unaffected. Measured on the corpus,
    27.1% of postings state exactly one - undamped, matching that single skill
    would score a perfect 1.0 on the heaviest component and outrank a posting
    listing ten requirements of which the user matches eight.

    Returns 0.0 for a posting with no required skills; AC-2.6 removes those at
    ingestion, so this is a guard rather than a live path.
    """
    if not required_skills:
        return 0.0
    ratio = len(user_skills & required_skills) / len(required_skills)
    confidence = min(1.0, len(required_skills) / MIN_SKILLS_FOR_FULL_CREDIT)
    return ratio * confidence


def preferred_skill_overlap(user_skills: set[str], preferred_skills: set[str]) -> float | None:
    """
    AC-9.4: matched fraction of preferred skills, or **None** when the posting
    lists none - the caller then drops this component and renormalizes the
    remaining seven weights.

    Never 1.0 (which would hand free points to the 93% of postings that list no
    preferred skills - the exact defect in the Stage 2 AI code) and never 0.0
    (which would punish a posting for a gap in our extraction).
    """
    if not preferred_skills:
        return None
    return len(user_skills & preferred_skills) / len(preferred_skills)


import math
import re

import pandas as pd

from src.filters import geocode, haversine_miles, normalize_location, parse_location
from src.profiles import UserProfile
from src.seq import as_set

#: AC-9.1. The 70/30 deterministic/semantic split follows the Stage 1 design:
#: deterministic components produce evidence that can be shown to the user
#: directly, semantic similarity is a supporting signal that cannot be pointed at.
WEIGHTS = {
    "required_skills": 0.30,
    "preferred_skills": 0.08,
    "experience": 0.15,
    "education": 0.04,
    "title": 0.05,
    "location": 0.08,
    "career_goals_similarity": 0.13,
    "resume_evidence_similarity": 0.17,
}

#: AC-9.2. Retained from the Stage 2 AI design - a "kept" item for report §4.
TIERS = ((80, "Strong"), (65, "Good"), (45, "Moderate"), (0, "Low"))

_SENIORITY = re.compile(r"\b(senior|sr|junior|jr|lead|staff|principal|i{1,3}|iv)\b", re.I)
_NON_WORD = re.compile(r"[^\w\s]")


def tier_for(score: int) -> str:
    return next(label for cutoff, label in TIERS if score >= cutoff)


def experience_alignment(user_years: float, job_min_years: float | None) -> float:
    """
    AC-9.5 (D7): asymmetric, because the two directions are different problems.

    Under-qualification is a *screening barrier* - a recruiter may not consider
    you. Over-qualification is a *preference mismatch* - the role is a step down.
    So the surplus side has a 3-year plateau, decays gently, and floors at 0.60,
    while the gap side falls to 0.15.
    """
    if job_min_years is None:
        return 0.5                                   # AC-9.13 explicit neutral
    gap = job_min_years - user_years
    if gap > 0:
        if gap <= 1.0:
            return 0.75
        if gap <= 2.5:
            return 0.50
        return max(0.15, 0.50 - 0.10 * gap)
    surplus = -gap
    if surplus <= 3.0:
        return 1.0
    return max(0.60, 1.0 - 0.05 * (surplus - 3.0))


def education_alignment(
    highest_completed: int, in_progress: int | None, required: int | None
) -> float:
    """
    AC-9.6 (D6): graded, not gated, with an in-progress half-step.

    A candidate part-way through a master's counts as 3.5 - full credit against a
    bachelor's requirement, partial against a master's. Gating would eliminate
    them from every master's-required posting, which is the objection that drove
    the decision.
    """
    if required is None:
        return 0.5                                   # AC-9.13 explicit neutral
    effective = (in_progress - 0.5) if in_progress else highest_completed
    if effective >= required:
        return 1.0
    return 0.6 if (required - effective) <= 1.0 else 0.2


def _tokens(text: str) -> set[str]:
    return set(_SENIORITY.sub(" ", _NON_WORD.sub(" ", (text or "").lower())).split())


def title_match(preferred_titles: list[str], job_title: str | None) -> float:
    """
    AC-9.7: how much of a desired title the job's title covers, best match wins.

    Seniority modifiers are stripped first, so "Data Engineer" fully matches
    "Senior Data Engineer" - seniority is scored by AC-9.5, not here.
    """
    if not preferred_titles:
        return 0.5                                   # AC-9.13 explicit neutral
    job = _tokens(job_title)
    if not job:
        return 0.0
    return max(
        (len(_tokens(p) & job) / len(_tokens(p)) if _tokens(p) else 0.0)
        for p in preferred_titles
    )


def location_proximity(
    is_remote: bool, accepts_remote: bool, miles: float | None, string_match: bool
) -> float:
    """
    AC-9.8: exponential decay on the distance already computed by REQ-6.

    ~1.0 at 0 mi, 0.85 at 10, 0.56 at 35, 0.19 at 100. This is what lets a strong
    match in a neighbouring suburb surface with a small reduction rather than
    being filtered out.
    """
    if is_remote and accepts_remote:
        return 1.0
    if miles is not None:
        return math.exp(-miles / 60.0)
    return 0.8 if string_match else 0.3


def _distance_for(job: pd.Series, profile: UserProfile) -> tuple[float | None, bool]:
    """Reuses REQ-6's geocoding rather than recomputing a second notion of distance."""
    if not profile.preferred_location:
        return None, False
    j_pt = geocode(*parse_location(job.get("location_raw")))
    u_pt = geocode(*parse_location(profile.preferred_location))
    if j_pt and u_pt:
        return haversine_miles(*u_pt, *j_pt), False
    return None, normalize_location(job.get("location_raw")) == normalize_location(
        profile.preferred_location
    )


def score_job(
    job: pd.Series,
    profile: UserProfile,
    goals_similarity: float,
    evidence_similarity: float,
) -> dict:
    """
    AC-9.1 / AC-9.14: the 0-100 score plus the breakdown that justifies it.

    Both similarity arguments must already be **calibrated** (AC-9.11) - passing
    raw cosine here would reintroduce the narrow-band problem calibration exists
    to solve.

    When a job lists no preferred skills the component is dropped and the
    remaining seven weights renormalize (AC-9.4). Measured, that is 93% of
    postings - and it is why the Stage 2 AI code's `else 1.0` would have handed
    free points to almost the whole corpus.
    """
    miles, string_match = _distance_for(job, profile)
    preferred = preferred_skill_overlap(profile.skills, as_set(job.get("preferred_skills")))

    subs = {
        "required_skills": required_skill_overlap(
            profile.skills, as_set(job.get("required_skills"))),
        "preferred_skills": preferred,
        "experience": experience_alignment(profile.years_experience, job.get("min_years_exp")),
        "education": education_alignment(
            profile.highest_completed_education, profile.education_in_progress,
            job.get("education_required")),
        "title": title_match(profile.preferred_titles, job.get("title_normalized")),
        "location": location_proximity(
            bool(job.get("is_remote")), "Remote" in profile.accepted_work_settings,
            miles, string_match),
        "career_goals_similarity": goals_similarity,
        "resume_evidence_similarity": evidence_similarity,
    }

    active = {k: w for k, w in WEIGHTS.items() if subs[k] is not None}
    total_weight = sum(active.values())

    components = []
    for name, weight in WEIGHTS.items():
        sub = subs[name]
        effective = (weight / total_weight) if sub is not None else 0.0
        components.append({
            "name": name,
            "sub_score": sub,
            "weight": effective,
            "weighted": (sub or 0.0) * effective,
        })

    score = round(sum(c["weighted"] for c in components) * 100)
    return {"job_id": job.get("job_id"), "score": score, "tier": tier_for(score),
            "components": components,
            "matched_skills": sorted(profile.skills & as_set(job.get("required_skills"))),
            "missing_skills": sorted(as_set(job.get("required_skills")) - profile.skills)}
