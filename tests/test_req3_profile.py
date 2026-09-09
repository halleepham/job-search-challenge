"""AC-derived tests for REQ-3 (FROZEN v1.0) — profile validation."""

import pytest

from src.validation import (
    EDUCATION_LADDER,
    EMPLOYMENT_TYPES,
    MIN_GOALS_CHARS,
    NOT_ENROLLED,
    WORK_SETTINGS,
    education_ordinal,
    validate_profile,
)

VALID = dict(
    resume_text="Built ETL pipelines in Airflow.",
    career_goals="I want a data engineering role building pipelines.",
    skills={"python"},
    work_settings={"Remote"},
    employment_types={"Full-time"},
    preferred_location="Kansas City, MO",
)


def test_valid_profile_has_no_errors():
    assert validate_profile(**VALID) == []


# ---------------------------------------------------------------- AC-3.4
# Each blocking condition, independently.

def test_ac_3_4_missing_resume():
    errors = validate_profile(**{**VALID, "resume_text": "  "})
    assert len(errors) == 1 and "résumé" in errors[0]


def test_ac_3_4_short_career_goals():
    errors = validate_profile(**{**VALID, "career_goals": "data job"})
    assert len(errors) == 1 and str(MIN_GOALS_CHARS) in errors[0]


def test_ac_3_4_goals_boundary_is_inclusive():
    assert validate_profile(**{**VALID, "career_goals": "x" * MIN_GOALS_CHARS}) == []
    assert validate_profile(**{**VALID, "career_goals": "x" * (MIN_GOALS_CHARS - 1)})


def test_ac_3_4_no_skills():
    errors = validate_profile(**{**VALID, "skills": set()})
    assert len(errors) == 1 and "skill" in errors[0]


def test_ac_3_4_no_work_setting():
    errors = validate_profile(**{**VALID, "work_settings": set()})
    assert any("work setting" in e for e in errors)


def test_ac_3_4_no_employment_type():
    errors = validate_profile(**{**VALID, "employment_types": set()})
    assert len(errors) == 1 and "employment type" in errors[0]


def test_ac_3_4_location_required_only_when_physical():
    """A location is needed only if the user would have to attend in person."""
    for setting in ("Hybrid", "On-site"):
        errors = validate_profile(**{**VALID, "work_settings": {setting},
                                     "preferred_location": ""})
        assert len(errors) == 1 and "location" in errors[0]

    assert validate_profile(**{**VALID, "work_settings": {"Remote"},
                               "preferred_location": ""}) == []


def test_ac_3_4_errors_accumulate_independently():
    """An empty form must report every problem at once, not one at a time."""
    errors = validate_profile("", "", set(), set(), set(), "")
    assert len(errors) == 5


# ---------------------------------------------------------------- AC-3.3

def test_ac_3_3_work_settings_are_a_set_of_three():
    assert set(WORK_SETTINGS) == {"Remote", "Hybrid", "On-site"}


def test_ac_3_3_five_employment_types_after_ac_1_6a():
    """AC-1.6a added Temporary; Volunteer and Other are excluded from the corpus."""
    assert set(EMPLOYMENT_TYPES) == {"Full-time", "Contract", "Part-time",
                                     "Temporary", "Internship"}


def test_ac_3_3_multi_select_is_valid():
    assert validate_profile(**{**VALID, "work_settings": {"Remote", "Hybrid"}}) == []


# ---------------------------------------------------------------- AC-3.6

@pytest.mark.parametrize("label,expected", [
    ("High School", 1), ("Associate", 2), ("Bachelor's", 3),
    ("Master's", 4), ("PhD", 5), (NOT_ENROLLED, None), (None, None),
])
def test_ac_3_6_education_ordinal(label, expected):
    assert education_ordinal(label) == expected


def test_ac_3_6_ladder_matches_ac_9_6():
    """The UI ladder and the scoring ladder must be the same five levels."""
    assert EDUCATION_LADDER["High School"] < EDUCATION_LADDER["Bachelor's"] \
        < EDUCATION_LADDER["PhD"]
    assert len(EDUCATION_LADDER) == 5


# ---------------------------------------------------------------- AC-3.7

def test_ac_3_7_presets_exist_and_differ():
    from src.profiles import PRESETS

    assert len(PRESETS) >= 3
    locations = {p.preferred_location for p in PRESETS.values()}
    assert len(locations) >= 3, "presets must differ in geography"
    assert len({p.years_experience for p in PRESETS.values()}) >= 3, "…and in seniority"


def test_ac_3_7_every_preset_validates():
    """A preset that cannot pass validation would be a broken demo."""
    from src.profiles import PRESETS

    for key, p in PRESETS.items():
        assert validate_profile(p.resume_text, p.career_goals, p.skills,
                                p.accepted_work_settings, p.accepted_employment_types,
                                p.preferred_location) == [], f"preset {key} fails validation"
