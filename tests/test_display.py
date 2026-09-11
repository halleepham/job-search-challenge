"""Display names must be readable without changing what is stored or matched."""

import pytest

from src.display import display_map, pretty, to_canonical


@pytest.mark.parametrize("canonical,shown", [
    ("python", "Python"), ("sql", "SQL"), ("aws", "AWS"),
    ("power bi", "Power BI"), ("javascript", "JavaScript"),
    ("c++", "C++"), ("c#", "C#"), ("dotnet", ".NET"), ("ios", "iOS"),
    ("machine learning", "Machine Learning"), ("rest api", "REST API"),
    ("scikit-learn", "scikit-learn"), ("dbt", "dbt"),
    ("data engineer", "Data Engineer"), ("qa engineer", "QA Engineer"),
])
def test_pretty_renders_readably(canonical, shown):
    assert pretty(canonical) == shown


def test_pretty_handles_empty():
    assert pretty(None) == "" and pretty("") == ""


def test_round_trip_preserves_canonical_form():
    """The stored value is what matching uses — display must not corrupt it."""
    canonical = {"python", "sql", "power bi", "c++"}
    mapping = display_map(canonical)
    assert to_canonical(mapping.keys(), mapping) == canonical


def test_free_text_is_normalized_not_rejected():
    """A user wanting a skill the vocabulary lacks must not be blocked."""
    assert to_canonical(["  Rust Lang "], {}) == {"rust lang"}


def test_free_text_mixes_with_known_values():
    mapping = display_map({"python"})
    assert to_canonical(["Python", "Cobol"], mapping) == {"python", "cobol"}


def test_blank_entries_dropped():
    assert to_canonical(["", "  "], {}) == set()
