"""
REQ-1: Job data ingestion and corpus construction (FROZEN v1.0).

Phase 1a implements AC-1.1 (tech scoping), AC-1.2 (recall-over-precision), and
AC-1.3 (deduplication). Field normalization (AC-1.4-1.7), coverage stats (AC-1.8),
and the Parquet write (AC-1.9) land in phases 1b and 1c.

AC-1.10 boundary: SQL owns the set-oriented work (semi-join, filter, dedupe);
per-row logic lives in the pure Python functions below and is unit-tested with no
database connection. DuckDB calls them as registered scalar UDFs, so there is one
implementation of each rule.

    python -m src.ingest
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from duckdb.typing import BOOLEAN, VARCHAR

# --- AC-1.1: named constants, not inlined -----------------------------------

#: LinkedIn job-function codes retained as "tech". Measured 2026-09-08: these
#: five cover 33,502 distinct postings (27% of the corpus). Note these are coarse
#: job-function categories, NOT tool-level skills (D4) - useless for matching,
#: exactly right for domain scoping (D3).
TECH_CODES = frozenset({"IT", "ENG", "ANLS", "QA", "SCI"})

#: Title whitelist for the OR branch of AC-1.1, for tech roles miscoded outside
#: TECH_CODES. Tightened 2026-09-08 after measuring the first version: bare
#: `analyst`, `administrator`, `technical`, `data`, and `it` contributed 6,157 rows
#: of which only 18% were software/data - pulling in financial analysts (131),
#: board-certified behavior analysts (67), office administrators (76), and data
#: entry clerks (31). Those five now appear only in compounds. AC-1.2's
#: recall-over-precision stance still holds for the TECH_CODES branch, where
#: AC-2.6 (drop postings with zero gazetteer skills) is the real precision filter.
TECH_TITLE_PATTERN = (
    # Unambiguous role words
    r"software|developer|programmer|devops|\bsre\b|site reliability|full[ -]?stack"
    r"|front[ -]?end|back[ -]?end|machine learning|\bml engineer\b|data scien"
    r"|analytics engineer|business intelligence|cyber|penetration test"
    # Compounds only - the bare forms of these are noise (measured 2026-09-08)
    r"|data (engineer|analyst|architect|warehouse)|database administrator|\bdba\b"
    r"|systems? (analyst|administrator|engineer)|network engineer"
    r"|\bcloud\b|security engineer|security analyst|information technology"
    r"|\bqa\b|quality assurance (engineer|analyst)|test engineer|\betl\b"
    r"|platform engineer|infrastructure engineer|solutions architect|web develop"
)
_TECH_TITLE_RE = re.compile(TECH_TITLE_PATTERN)

#: Seniority modifiers stripped before title comparison, so that
#: "Senior Data Engineer" and "Data Engineer" share a normalized form.
_SENIORITY_RE = re.compile(
    r"\b(senior|sr|junior|jr|lead|staff|principal|entry[ -]level|experienced)\b"
)
_TRAILING_LEVEL_RE = re.compile(r"\s+(i{1,3}|iv|vi?)$")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


# --- pure per-row functions (AC-1.10) ---------------------------------------

def normalize_key(value: str | None) -> str:
    """
    AC-1.3: lowercase, strip punctuation, collapse whitespace.

    Used for the company and location components of the dedupe key so that
    "ACME CORP" and "Acme Corp", or "Kansas City,  MO" and "kansas city, mo",
    compare equal *before* comparison rather than after.
    """
    if not value:
        return ""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", value.lower())).strip()


def normalize_title(title: str | None) -> str:
    """
    Normalized title: lowercased, punctuation stripped, seniority modifiers and
    trailing level numerals removed, whitespace collapsed.

    Serves double duty - the title component of AC-1.3's dedupe key, and the
    input to AC-1.1's title whitelist.
    """
    if not title:
        return ""
    t = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", title.lower())).strip()
    t = _SENIORITY_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    t = _TRAILING_LEVEL_RE.sub("", t)
    return _WS_RE.sub(" ", t).strip()


def is_tech_title(title_normalized: str | None) -> bool:
    """AC-1.1: the OR branch - retain a posting whose title looks technical."""
    if not title_normalized:
        return False
    return _TECH_TITLE_RE.search(title_normalized) is not None


# --- set-oriented work (SQL) ------------------------------------------------

@dataclass
class ScopeResult:
    """Phase 1a output plus the funnel counts AC-1.8 will persist."""

    frame: pd.DataFrame
    n_raw: int
    n_after_scoping: int
    n_after_dedupe: int

    @property
    def n_dropped_non_tech(self) -> int:
        return self.n_raw - self.n_after_scoping

    @property
    def n_dropped_duplicates(self) -> int:
        return self.n_after_scoping - self.n_after_dedupe


def _connect() -> duckdb.DuckDBPyConnection:
    """In-process connection with the per-row functions registered as UDFs."""
    con = duckdb.connect()
    con.create_function("norm_title", normalize_title, [VARCHAR], VARCHAR)
    con.create_function("norm_key", normalize_key, [VARCHAR], VARCHAR)
    con.create_function("is_tech_title", is_tech_title, [VARCHAR], BOOLEAN)
    return con


def scope_and_dedupe(postings_path: Path | str, job_skills_path: Path | str) -> ScopeResult:
    """
    AC-1.1 / AC-1.2 / AC-1.3: scope the corpus to tech roles and deduplicate.

    Note the two different title normalizations, which is deliberate (AC-1.3 v1.1):
    `norm_key(title)` preserves seniority and forms the dedupe key, so that
    "Senior Data Engineer" and "Data Engineer" at one company stay distinct;
    `norm_title(title)` strips seniority and drives AC-1.1's whitelist.

    The tech filter is a **semi-join** (`job_id IN (SELECT ...)`), not an inner
    join: job_skills.csv holds ~1.69 rows per job, so an inner join would emit a
    posting once per code it carries.
    """
    con = _connect()
    con.execute(
        "CREATE TABLE postings AS "
        "SELECT * FROM read_csv_auto(?, sample_size=100000, types={'job_id':'BIGINT'})",
        [str(postings_path)],
    )
    con.execute(
        "CREATE TABLE job_skills AS "
        "SELECT * FROM read_csv_auto(?, types={'job_id':'BIGINT'})",
        [str(job_skills_path)],
    )

    n_raw = con.execute("SELECT count(*) FROM postings").fetchone()[0]

    con.execute(
        f"""
        CREATE TABLE scoped AS
        SELECT p.*, norm_title(p.title) AS title_normalized
        FROM postings p
        WHERE p.job_id IN (
                  SELECT job_id FROM job_skills
                  WHERE skill_abr IN ({','.join('?' * len(TECH_CODES))})
              )
           OR is_tech_title(norm_title(p.title))
        """,
        sorted(TECH_CODES),
    )
    n_after_scoping = con.execute("SELECT count(*) FROM scoped").fetchone()[0]

    frame = con.execute(
        """
        SELECT * FROM scoped
        QUALIFY row_number() OVER (
            PARTITION BY norm_key(title),
                         norm_key(company_name),
                         norm_key(location)
            ORDER BY job_id
        ) = 1
        ORDER BY job_id
        """
    ).df()
    con.close()

    return ScopeResult(
        frame=frame,
        n_raw=n_raw,
        n_after_scoping=n_after_scoping,
        n_after_dedupe=len(frame),
    )


def main() -> None:
    result = scope_and_dedupe("data/raw/postings.csv", "data/raw/jobs/job_skills.csv")
    pct = 100 * result.n_after_dedupe / result.n_raw
    print("\nREQ-1 phase 1a funnel (AC-1.1, AC-1.2, AC-1.3)")
    print("-" * 52)
    print(f"  raw postings              {result.n_raw:>8,}")
    print(f"  - non-tech dropped        {result.n_dropped_non_tech:>8,}")
    print(f"  = after tech scoping      {result.n_after_scoping:>8,}")
    print(f"  - duplicates dropped      {result.n_dropped_duplicates:>8,}")
    print(f"  = corpus                  {result.n_after_dedupe:>8,}   ({pct:.1f}% of raw)")
    print("\nNote: final corpus size is set by AC-2.6 (gazetteer coverage), not by this stage.")


if __name__ == "__main__":
    main()
