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
