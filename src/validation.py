"""
REQ-3: profile validation (FROZEN v1.0).

Pure functions so AC-3.4's six conditions are each unit-testable without
rendering a page.
"""

from __future__ import annotations

MIN_GOALS_CHARS = 20

EDUCATION_LADDER = {
    "High School": 1, "Associate": 2, "Bachelor's": 3, "Master's": 4, "PhD": 5,
}
NOT_ENROLLED = "Not currently enrolled"

WORK_SETTINGS = ("Remote", "Hybrid", "On-site")
EMPLOYMENT_TYPES = ("Full-time", "Contract", "Part-time", "Temporary", "Internship")

#: A location is only required when the user would have to physically attend.
_PHYSICAL = {"Hybrid", "On-site"}


def validate_profile(
    resume_text: str,
    career_goals: str,
    skills: set[str] | list[str],
    work_settings: set[str] | list[str],
    employment_types: set[str] | list[str],
    preferred_location: str | None,
) -> list[str]:
    """
    AC-3.4: every blocking condition, each with a specific message.

    Returns an empty list when the profile may be submitted.
    """
    errors: list[str] = []

    if not (resume_text or "").strip():
        errors.append("Upload a résumé or paste its text — the match evidence comes from it.")

    goals = (career_goals or "").strip()
    if len(goals) < MIN_GOALS_CHARS:
        errors.append(
            f"Describe your career goals in at least {MIN_GOALS_CHARS} characters "
            f"(currently {len(goals)}). This drives the semantic match."
        )

    if not skills:
        errors.append("Add at least one skill — skills are the largest part of the match score.")

    if not work_settings:
        errors.append("Check at least one work setting.")

    if not employment_types:
        errors.append("Check at least one employment type.")

    if set(work_settings) & _PHYSICAL and not (preferred_location or "").strip():
        errors.append(
            "Set a preferred location, or uncheck Hybrid and On-site — "
            "we cannot judge distance without one."
        )

    return errors


def education_ordinal(label: str | None) -> int | None:
    """AC-3.6: ladder label to the ordinal AC-9.6 scores on."""
    if not label or label == NOT_ENROLLED:
        return None
    return EDUCATION_LADDER.get(label)
