"""
Pre-class readiness check: verifies every external dependency is in place so no
class time is spent downloading packages, models, or datasets.

Checks packages, credentials, datasets, and model caches. Everything listed is
REQUIRED - the system has no optional runtime dependency and needs no API key
(D11: no LLM in the application).

    python -m src.preflight
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "

PACKAGES = [
    ("duckdb", "REQ-1 ingestion, REQ-12 analytics"),
    ("pandas", "REQ-1, query-time path"),
    ("pyarrow", "parquet"),
    ("kagglehub", "dataset download"),
    ("datasets", "data_jobs (REQ-2 seed, REQ-13 scale test)"),
    ("sentence_transformers", "REQ-4/5 embeddings, REQ-8 reranker"),
    ("faiss", "REQ-5 vector index"),
    ("rank_bm25", "REQ-7 keyword retrieval"),
    ("geonamescache", "REQ-6 geocoding, REQ-3 location validation"),
    ("pypdf", "REQ-3 resume parsing"),
    ("streamlit", "REQ-3/11/12 UI"),
    ("plotly", "REQ-11/12 charts"),
    ("pytest", "AC-derived tests"),
]

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "AC-4.3 embeddings"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "AC-8.1 reranker (stretch)"),
]

results: list[tuple[str, str, str]] = []


def check(status: str, name: str, detail: str) -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name:44s} {detail}")


def hf_cached(repo: str) -> bool:
    """True if the model is in the local HF cache (no network call)."""
    cache = Path.home() / ".cache/huggingface/hub"
    slug = "models--" + repo.replace("/", "--")
    d = cache / slug
    return d.is_dir() and any(d.glob("snapshots/*/*"))


def main() -> None:
    print(f"\nPython {sys.version.split()[0]}  ({sys.executable})\n")

    print("--- packages ---")
    for mod, why in PACKAGES:
        try:
            importlib.import_module(mod)
            check(OK, mod, why)
        except ImportError:
            check(FAIL, mod, f"MISSING - {why}")

    print("\n--- credentials ---")
    tok = Path.home() / ".kaggle/access_token"
    legacy = Path.home() / ".kaggle/kaggle.json"
    if tok.exists() or legacy.exists():
        check(OK, "kaggle credentials", str(tok if tok.exists() else legacy))
    else:
        check(FAIL, "kaggle credentials", "no ~/.kaggle/access_token")

    print("\n--- datasets ---")
    postings = Path("data/raw/postings.csv")
    if postings.exists():
        n = sum(1 for _ in postings.open(encoding="utf-8", errors="ignore")) - 1
        csvs = len(list(Path("data/raw").rglob("*.csv")))
        check(OK, "LinkedIn job postings", f"{csvs} CSVs, ~{n:,} raw lines in postings.csv")
    else:
        check(FAIL, "LinkedIn job postings", "run: python -m src.download_data")

    dj = list((Path.home() / ".cache/huggingface/datasets").glob("*data_jobs*")) if \
        (Path.home() / ".cache/huggingface/datasets").is_dir() else []
    if dj:
        check(OK, "data_jobs (HF)", "cached")
    else:
        check(FAIL, "data_jobs (HF)", "run: python -m src.download_data_jobs")

    print("\n--- models (cached locally, no network needed at runtime) ---")
    for repo, why in MODELS:
        if hf_cached(repo):
            check(OK, repo, why)
        else:
            check(FAIL, repo, f"NOT CACHED - {why}")

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print("\n" + "=" * 78)
    if fails:
        print(f"NOT READY - {len(fails)} blocking item(s):")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print(f"READY for offline work. {len(warns)} optional item(s) outstanding.")
    if warns:
        for _, name, detail in warns:
            print(f"  - {name}: {detail}")


if __name__ == "__main__":
    main()
