"""
AC-derived tests for REQ-2 (FROZEN v1.0) — skill vocabulary and extraction.

AC-2.2 is the most important test in the suite: it encodes the exact defect in
the Stage 2 AI code (`agent-exercise:src/matcher.py:60-63`), where bidirectional
substring matching made "Java" match "JavaScript".
"""

import json
from pathlib import Path

import pytest

from src.skills import (
    VOCAB_PATH,
    extract_skills,
    load_vocabulary,
    match_skills,
    split_sections,
)


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


# ---------------------------------------------------------------- AC-2.1

def test_ac_2_1_vocabulary_is_a_committed_artifact():
    assert VOCAB_PATH.exists(), "gazetteer must be committed, not generated at runtime"


def test_ac_2_1_minimum_size(vocab):
    """AC-2.1: target >= 350 canonical terms."""
    assert len(vocab.canonical) >= 350


def test_ac_2_1_seeded_and_extended():
    """AC-2.1: seeded from data_jobs, hand-extended for non-data tech."""
    meta = json.loads(VOCAB_PATH.read_text())["_meta"]
    assert meta["seed_dataset"] == "lukebarousse/data_jobs"
    assert meta["seed_terms"] >= 250 and meta["hand_added"] >= 100


@pytest.mark.parametrize("skill", [
    "python", "sql", "spark", "airflow",                 # data (seed)
    "react", "kubernetes", "terraform", "docker",         # software (seed)
    "ios", "android", "figma", "dotnet",                  # hand-added
    "selenium", "owasp", "penetration testing",           # hand-added
])
def test_ac_2_1_covers_data_and_non_data_tech(vocab, skill):
    assert skill in vocab.canonical or skill in vocab.alias_to_canonical


# ---------------------------------------------------------------- AC-2.8

def test_ac_2_8_aliases_are_data_not_code():
    """AC-2.8: aliases live in the JSON artifact, editable without touching code."""
    raw = json.loads(VOCAB_PATH.read_text())
    assert any(aliases for aliases in raw["skills"].values())


@pytest.mark.parametrize("alias,canonical", [
    ("k8s", "kubernetes"), ("react.js", "react"), ("reactjs", "react"),
    ("ml", "machine learning"), ("golang", "go"), ("postgres", "postgresql"),
    ("js", "javascript"), ("nodejs", "node.js"), (".net", "dotnet"),
])
def test_ac_2_8_alias_resolves_to_canonical(alias, canonical):
    assert match_skills(f"Experience with {alias} required") == {canonical}


def test_ac_2_8_canonical_and_alias_sets_are_disjoint(vocab):
    """A surface form must not resolve two ways depending on lookup order."""
    assert not (set(vocab.canonical) & set(vocab.alias_to_canonical))


# ---------------------------------------------------------------- AC-2.2
# The defect class from the Stage 2 AI code. Every case here is a false positive
# that bidirectional substring matching produces.

@pytest.mark.parametrize("text,absent", [
    ("Strong JavaScript and TypeScript skills", "java"),
    ("Experience with MongoDB at scale", "go"),
    ("Contact HR for details", "r"),
    ("Our R&D team builds prototypes", "r"),
    ("Background in Scala development", "c"),
    ("Familiar with GoLand IDE", "go"),
])
def test_ac_2_2_no_substring_false_positives(text, absent):
    assert absent not in match_skills(text), f"{absent!r} falsely matched in {text!r}"


@pytest.mark.parametrize("text,expected", [
    ("Proficient in R and Python", {"r", "python"}),
    ("Java and Spring Boot", {"java"}),
    ("C++ and C# both required", {"c++", "c#"}),
    ("Go microservices on Kubernetes", {"go", "microservices", "kubernetes"}),
])
def test_ac_2_2_true_positives_still_match(text, expected):
    assert expected <= match_skills(text)


def test_ac_2_2_longest_term_wins():
    """'power bi' must not also yield 'bi' (business intelligence)."""
    found = match_skills("Dashboards in Power BI")
    assert "power bi" in found
    assert "business intelligence" not in found


def test_ac_2_2_case_insensitive():
    assert match_skills("PYTHON, Sql and AirFlow") == {"python", "sql", "airflow"}


# ---------------------------------------------------------------- AC-2.4

def test_ac_2_4_splits_required_and_preferred():
    text = ("About us: we build things.\n"
            "Required Qualifications:\n- Python\n- SQL\n"
            "Preferred Qualifications:\n- Spark\n- Airflow\n")
    sections = split_sections(text)
    assert "python" in sections["required"].lower()
    assert "spark" in sections["preferred"].lower()
    assert "spark" not in sections["required"].lower(), "preferred leaked into required"


@pytest.mark.parametrize("heading", [
    "Requirements:", "Qualifications:", "Must have:", "Minimum Qualifications:",
    "Basic Qualifications:",
])
def test_ac_2_4_required_headings(heading):
    assert "python" in split_sections(f"Intro\n{heading}\nPython\n")["required"].lower()


@pytest.mark.parametrize("heading", [
    "Preferred:", "Nice to have:", "Bonus:", "Desired:", "Preferred Qualifications:",
])
def test_ac_2_4_preferred_headings(heading):
    assert "spark" in split_sections(f"Intro\n{heading}\nSpark\n")["preferred"].lower()


def test_ac_2_4_section_ends_at_next_heading():
    text = "Requirements:\nPython\nPreferred:\nSpark\nBenefits:\nDental\n"
    s = split_sections(text)
    assert "dental" not in s["preferred"].lower(), "section ran past the next heading"


# ---------------------------------------------------------------- AC-2.3

def test_ac_2_3_description_is_the_primary_path():
    """AC-2.3 v1.1: skills_desc is 98% null, so description parsing must stand alone."""
    req, pref = extract_skills(
        description="Requirements:\nPython and SQL\nPreferred:\nSpark\n", skills_desc=None
    )
    assert {"python", "sql"} <= req and "spark" in pref


def test_ac_2_3_skills_desc_supplements_into_required():
    req, pref = extract_skills(
        description="Requirements:\nPython\nPreferred:\nSpark\n",
        skills_desc="Must also know Kubernetes and Terraform",
    )
    assert {"python", "kubernetes", "terraform"} <= req


def test_ac_2_3_required_wins_conflicts():
    """A skill in both buckets belongs to required only."""
    req, pref = extract_skills(
        description="Requirements:\nPython\nPreferred:\nPython and Spark\n", skills_desc=None
    )
    assert "python" in req and "python" not in pref


def test_ac_2_3_no_behaviour_depends_on_skills_desc():
    """AC-2.3: skills_desc can only add skills, never gate them."""
    d = "Requirements:\nPython and SQL\n"
    with_none, _ = extract_skills(description=d, skills_desc=None)
    with_empty, _ = extract_skills(description=d, skills_desc="")
    assert with_none == with_empty == {"python", "sql"}


# ---------------------------------------------------------------- AC-2.5

def test_ac_2_5_no_preferred_section_yields_empty():
    """AC-2.5: never a copy of required, never a guess."""
    req, pref = extract_skills(description="Requirements:\nPython and SQL\n", skills_desc=None)
    assert pref == set()
    assert req == {"python", "sql"}


def test_ac_2_5_unsectioned_description_all_required():
    """No headings at all: everything found is required, preferred stays empty."""
    req, pref = extract_skills(
        description="We need someone strong in Python, SQL and Airflow.", skills_desc=None
    )
    assert {"python", "sql", "airflow"} <= req
    assert pref == set()


def test_ac_2_5_nothing_found_is_empty_not_none():
    req, pref = extract_skills(description="We value teamwork and grit.", skills_desc=None)
    assert req == set() and pref == set()


# ---------------------------------------------------------------- AC-2.2 v1.1
# Single characters in prose are not evidence. Every case below was found in the
# real corpus, where 645 postings matched `r` and 83 had it as their only skill.

@pytest.mark.parametrize("text", [
    "the P/R organization whose primary responsibility",     # slash
    "meet mission r equirements. Provide technical input",   # word broken mid-token
    "report to your Project Man******r. Recognize when",     # masked text
    "R&D team builds prototypes",
    "Contact HR for details",
])
def test_ac_2_2_v1_1_lone_letter_is_not_a_skill(text):
    assert "r" not in match_skills(text)


@pytest.mark.parametrize("text", [
    "Programming experience with R and SAS",
    "Strong Python, R, and SQL skills",
    "Modelling in R or Python required",
])
def test_ac_2_2_v1_1_lone_letter_counts_beside_a_real_skill(text):
    assert "r" in match_skills(text)


def test_ac_2_2_v1_1_co_occurrence_must_be_nearby():
    """A skill three paragraphs away is not context for a stray letter."""
    far = "P/R organization. " + "filler " * 40 + "We also use Python."
    assert "r" not in match_skills(far)
    assert "python" in match_skills(far)


def test_ac_2_2_v1_1_multi_letter_skills_need_no_neighbour():
    """Only single characters are gated — a lone 'python' is still evidence."""
    assert match_skills("Python developer wanted") == {"python"}
