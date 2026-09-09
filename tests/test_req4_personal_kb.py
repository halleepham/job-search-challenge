"""AC-derived tests for REQ-4 (FROZEN v1.0) — personal knowledge base."""

import numpy as np
import pytest

from src.personal_kb import PersonalKB, chunk_documents

RESUME = """Education
M.S. Data Science, University of Missouri-Kansas City, expected 2026.
B.S. Mathematics, 2023.

Skills
Python, SQL, Spark, Airflow, AWS, Tableau.

Experience
Data Engineering Intern, Acme Analytics, 2024-2025.
Built ETL pipelines in Airflow moving 40M rows/day into Snowflake.
Wrote dbt models and Great Expectations tests for data quality.

Projects
Kansas City housing price model using scikit-learn and XGBoost.
Deployed a Streamlit dashboard on AWS.
"""

GOALS = "I want a data engineering role building pipelines that turn raw data into dashboards."


@pytest.fixture(scope="module")
def kb():
    return PersonalKB.build(resume_text=RESUME, career_goals=GOALS)


# ---------------------------------------------------------------- AC-4.1

def test_ac_4_1_chunks_by_paragraph_block(kb):
    """AC-4.1 v1.1: blank-line split, short blocks merged. Typical resume 8-15."""
    assert 4 <= len(kb.chunks) <= 20


def test_ac_4_1_career_goals_is_its_own_chunk(kb):
    goals = [c for c in kb.chunks if c.section == "career_goals"]
    assert len(goals) == 1 and goals[0].text == GOALS


def test_ac_4_1_short_blocks_are_merged():
    """A stray one-line block must not become a chunk of its own."""
    chunks = chunk_documents("Skills\n\nPython\n\nExperience\n\nBuilt pipelines in Airflow daily.")
    assert all(len(c.text) >= 40 or c.section == "career_goals" for c in chunks)


# ---------------------------------------------------------------- AC-4.2

def test_ac_4_2_chunks_carry_metadata(kb):
    for c in kb.chunks:
        assert c.section and c.source and c.char_span


def test_ac_4_2_char_spans_are_verbatim(kb):
    """
    AC-4.2: every span must slice back to exactly the chunk text. This is what
    lets AC-11.5 display quotes that are provably source text, never paraphrase.
    """
    for c in kb.chunks:
        source = GOALS if c.source == "career_goals" else RESUME
        start, end = c.char_span
        assert source[start:end] == c.text, f"span does not round-trip for {c.section}"


# ---------------------------------------------------------------- AC-4.3

def test_ac_4_3_embedding_model_is_pinned(kb):
    from src.personal_kb import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert kb.embeddings.shape[1] == 384


def test_ac_4_3_embeddings_are_normalized(kb):
    """Unit-norm vectors make cosine similarity a plain dot product."""
    norms = np.linalg.norm(kb.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


# ---------------------------------------------------------------- AC-4.4

@pytest.mark.parametrize("k", [1, 3, 100])
def test_ac_4_4_retrieve_returns_min_k_chunks(kb, k):
    hits = kb.retrieve("data pipelines in Airflow", k)
    assert len(hits) == min(k, len(kb.chunks))


def test_ac_4_4_ordered_by_descending_similarity(kb):
    scores = [s for _, s in kb.retrieve("ETL pipelines", 5)]
    assert scores == sorted(scores, reverse=True)


def test_ac_4_4_scores_attached_and_bounded(kb):
    for _, score in kb.retrieve("python", 3):
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------- AC-4.5

def test_ac_4_5_retrieval_finds_the_relevant_section(kb):
    """
    AC-4.5: a pipelines-heavy job description must surface the experience chunk
    ahead of unrelated ones. Tested with two contrasting descriptions.
    """
    eng_top = kb.retrieve("Build and maintain ETL data pipelines with Airflow and Snowflake", 1)[0][0]
    assert "airflow" in eng_top.text.lower() or "etl" in eng_top.text.lower()

    edu_top = kb.retrieve("Requires a Master's degree in a quantitative field", 1)[0][0]
    assert "m.s." in edu_top.text.lower() or "degree" in edu_top.text.lower()


def test_ac_4_5_contrasting_queries_return_different_chunks(kb):
    a = kb.retrieve("Airflow ETL pipelines Snowflake", 1)[0][0].text
    b = kb.retrieve("Master's degree university coursework", 1)[0][0].text
    assert a != b, "retrieval is not discriminating between contrasting queries"


# ---------------------------------------------------------------- AC-4.6

def test_ac_4_6_structured_and_unstructured_are_separate(kb):
    """
    AC-4.6: structured fields live on the profile, embedded chunks are separate.
    The same text feeds both, but they never share a representation.
    """
    assert not hasattr(kb, "skills"), "structured fields must not live on the KB"
    assert all(isinstance(c.text, str) for c in kb.chunks)


def test_ac_4_6_empty_input_is_handled():
    kb = PersonalKB.build(resume_text="", career_goals="Looking for data work.")
    assert len(kb.chunks) == 1 and kb.retrieve("anything", 3)
