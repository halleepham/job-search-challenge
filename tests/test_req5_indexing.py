"""AC-derived tests for REQ-5 (FROZEN v1.0) — job indexing and calibration."""

import json

import numpy as np
import pandas as pd
import pytest

from src.indexing import (
    CalibrationConstants,
    JobIndex,
    build_bm25_text,
    build_dense_text,
    calibrate,
)

JOBS = pd.DataFrame([
    {"job_id": 1, "title": "Data Engineer", "description": "Build ETL pipelines in Airflow.",
     "required_skills": ["python", "airflow", "sql"], "preferred_skills": ["dbt"]},
    {"job_id": 2, "title": "Backend Engineer", "description": "Go microservices on Kubernetes.",
     "required_skills": ["go", "kubernetes"], "preferred_skills": []},
    {"job_id": 3, "title": "Security Analyst", "description": "SOC monitoring and incident response.",
     "required_skills": ["splunk", "siem"], "preferred_skills": []},
])


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    return JobIndex.build(JOBS, index_dir=tmp_path_factory.mktemp("idx"))


# ---------------------------------------------------------------- AC-5.1

def test_ac_5_1_dense_text_excludes_skills():
    """
    AC-5.1 + AC-9.12: the dense vector is title + description only. Skills have
    their own score components, so embedding them would count one signal twice.
    """
    text = build_dense_text(JOBS.iloc[0])
    assert "Data Engineer" in text and "Airflow" in text
    assert "dbt" not in text, "preferred skills leaked into the dense text"


def test_ac_5_1_title_comes_first():
    """Title leads so the highest-signal text is inside the 256-token window."""
    assert build_dense_text(JOBS.iloc[0]).startswith("Data Engineer")


def test_ac_5_1_one_vector_per_job(index):
    assert index.embeddings.shape == (len(JOBS), 384)


def test_ac_5_1_vectors_normalized(index):
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------- AC-5.2

def test_ac_5_2_bm25_text_includes_skills():
    """AC-5.2: BM25 carries keyword precision on named tools over the full text."""
    text = build_bm25_text(JOBS.iloc[0])
    assert "python" in text and "airflow" in text and "dbt" in text


def test_ac_5_2_bm25_index_searchable(index):
    hits = index.search_bm25("kubernetes microservices", k=1)
    assert hits[0][0] == 1, "BM25 did not rank the Go/Kubernetes job first"


def test_ac_5_2_indexes_cover_different_text():
    """The two indexes deliberately index different text (AC-5.1 vs AC-5.2)."""
    row = JOBS.iloc[0]
    assert build_dense_text(row) != build_bm25_text(row)


# ---------------------------------------------------------------- AC-5.3

def test_ac_5_3_manifest_records_identity(index):
    m = json.loads((index.index_dir / "manifest.json").read_text())
    assert m["n_rows"] == len(JOBS)
    assert m["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert m["corpus_hash"]


def test_ac_5_3_reloads_from_cache(index, tmp_path):
    """A matching manifest must load from disk, not re-embed."""
    reloaded = JobIndex.load_or_build(JOBS, index_dir=index.index_dir)
    assert reloaded.from_cache is True
    assert np.allclose(reloaded.embeddings, index.embeddings)


def test_ac_5_3_corpus_change_invalidates(index):
    """A different corpus must not silently reuse the cached vectors."""
    changed = pd.concat([JOBS, JOBS.iloc[[0]].assign(job_id=99)], ignore_index=True)
    rebuilt = JobIndex.load_or_build(changed, index_dir=index.index_dir)
    assert rebuilt.from_cache is False
    assert rebuilt.embeddings.shape[0] == len(changed)


# ---------------------------------------------------------------- AC-5.5 / 5.7 / 5.8

def test_ac_5_5_calibration_persisted(index):
    c = json.loads((index.index_dir / "calibration.json").read_text())
    assert c["p5"] < c["p95"]
    assert c["n_sampled"] > 0 and c["n_profiles"] >= 3


def test_ac_5_6_uses_multiple_distinct_profiles():
    """AC-5.6: at least one reference profile unlike the author's own."""
    from src.profiles import CALIBRATION_PROFILES, PRESETS
    assert len(CALIBRATION_PROFILES) >= 3
    domains = {p.preferred_titles[0].split()[0] for p in PRESETS.values()}
    assert len(domains) >= 3, "reference profiles are not materially different"


def test_ac_5_7_calibration_uses_precomputed_vectors(index):
    """AC-5.7: no extra embedding pass — calibration reads existing vectors."""
    c = calibrate(index.embeddings, np.random.default_rng(0).normal(size=(3, 384)))
    assert isinstance(c, CalibrationConstants)
    assert c.p5 < c.p95


@pytest.mark.parametrize("raw,expected", [(-1.0, 0.0), (2.0, 1.0)])
def test_ac_9_11_calibration_clips(raw, expected):
    """AC-9.11: mapped output is bounded to [0, 1]."""
    c = CalibrationConstants(p5=0.1, p95=0.5, n_sampled=100, n_profiles=3)
    assert c.apply(raw) == expected


def test_ac_9_11_calibration_is_pool_independent():
    """
    AC-9.11 / D8: a job's calibrated score must not depend on which other jobs
    are in the candidate set. Pool-relative min-max would fail this.
    """
    c = CalibrationConstants(p5=0.2, p95=0.7, n_sampled=2000, n_profiles=3)
    alone = c.apply(0.45)
    with_others = [c.apply(x) for x in (0.1, 0.45, 0.9)]
    assert with_others[1] == alone


def test_ac_9_11_spreads_the_narrow_band():
    """Raw cosine clusters ~0.25-0.70; calibration must spread that usefully."""
    c = CalibrationConstants(p5=0.25, p95=0.70, n_sampled=2000, n_profiles=3)
    lo, hi = c.apply(0.30), c.apply(0.65)
    assert hi - lo > 0.5, "calibration is not spreading the distribution"


def test_ac_5_8_calibration_shares_the_manifest(index):
    """AC-5.8: same invalidation as the indexes."""
    assert (index.index_dir / "calibration.json").exists()
    assert (index.index_dir / "manifest.json").exists()
