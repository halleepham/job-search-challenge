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

import json
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from duckdb.typing import BOOLEAN, VARCHAR

from src.skills import extract_skills, load_vocabulary, split_sections

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


# --- AC-1.4 / AC-1.5 / AC-1.6 / AC-1.6a / AC-1.7 constants ------------------

#: Annualization multipliers. AC-1.4 defines rules for these three only; every
#: other or missing pay_period yields no salary. Measured 2026-09-08: WEEKLY (177)
#: and BIWEEKLY (9) exist in the data but every one of those rows also carries
#: `normalized_salary`, so the primary path covers them and the derivation
#: fallback never needs them.
SALARY_MULTIPLIERS = {"HOURLY": 2080, "MONTHLY": 12, "YEARLY": 1}

#: AC-1.6a: retained employment types. `Volunteer` (562) and `Other` (487) are
#: excluded from the corpus - neither is a role a user of this system searches for.
EMPLOYMENT_TYPES = frozenset(
    {"Full-time", "Contract", "Part-time", "Temporary", "Internship"}
)

#: AC-1.6: fallback mapping when the description states no numeric requirement.
SENIORITY_YEARS = {
    "Internship": 0.0, "Entry level": 0.0, "Associate": 2.0,
    "Mid-Senior level": 5.0, "Director": 8.0, "Executive": 10.0,
}

#: AC-1.7 / AC-9.6: ordinal education ladder.
EDUCATION_LEVELS = {
    "high school": 1, "associate": 2, "bachelor": 3, "master": 4, "phd": 5,
}

#: AC-1.7: explicit degree phrasings only. Bare `bs`/`ms`/`ba`/`ma` are never
#: matched - "MS Office" and "MS Word" would otherwise read as a Master's degree.
_EDUCATION_PATTERNS = [
    (1, r"high[- ]school|\bged\b|h\.s\.\s+diploma"),
    (2, r"associate(?:'s|s)?\s+(?:degree|of)|\ba\.?a\.?s?\.?\s+degree"),
    (3, r"bachelor|undergraduate\s+degree|\bb\.?s\.?\s*/\s*b\.?a\.?|"
        r"\b(?:bs|ba)\s+degree|\bb\.s\.|\bb\.a\."),
    (4, r"master(?:'s|s)?\s+(?:degree|of|in)|\bmba\b|\bm\.s\.|\bm\.a\.|"
        r"graduate\s+degree"),
    (5, r"\bph\.?\s?d\.?\b|doctorate|doctoral"),
]
_EDUCATION_RES = [(lvl, re.compile(pat, re.I)) for lvl, pat in _EDUCATION_PATTERNS]

_YEARS_RE = re.compile(r"(\d+)\s*(?:[-–—]\s*\d+)?\s*\+?\s*years?", re.I)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.I)

#: AC-1.6: an experience requirement above this is implausible and is treated as
#: no match ("100 years of combined experience" is a company boast, not a rule).
MAX_PLAUSIBLE_YEARS = 30


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


def _is_num(value) -> bool:
    """
    True only for a real number. Guards the NaN trap: pandas yields NaN rather
    than None for a missing numeric cell, and ``NaN is not None`` is True, so a
    plain None check silently admits NaN and propagates it through arithmetic.
    """
    return value is not None and isinstance(value, (int, float)) and value == value


def normalize_salary(
    normalized_salary: float | None,
    salary_min: float | None,
    salary_max: float | None,
    pay_period: str | None,
    currency: str | None,
) -> tuple[float | None, float | None, bool, str]:
    """
    AC-1.4: resolve salary to an annual USD range.

    Returns ``(salary_min, salary_max, salary_listed, source)`` where source is
    one of ``normalized_salary`` | ``derived`` | ``non_usd`` | ``none``, so
    coverage stats can report how many rows took each path.

    Precedence: the dataset's own `normalized_salary` (an annualized midpoint -
    verified: (17+20)/2 x 2080 = 38,480) is authoritative. Deriving from
    min/max x pay_period is the fallback. Measured 2026-09-08: every row with
    min/max also has normalized_salary, so on this corpus the fallback never
    fires - it is retained for correctness on other sources.
    """
    if currency is not None and str(currency).upper() not in ("USD", "NAN"):
        return None, None, False, "non_usd"

    mult = SALARY_MULTIPLIERS.get((pay_period or "").upper())

    has_range = mult and _is_num(salary_min) and _is_num(salary_max)

    if _is_num(normalized_salary):
        if has_range:
            return salary_min * mult, salary_max * mult, True, "normalized_salary"
        return normalized_salary, normalized_salary, True, "normalized_salary"

    if has_range:
        return salary_min * mult, salary_max * mult, True, "derived"

    return None, None, False, "none"


def derive_work_setting(
    remote_allowed: float | None, title: str | None, description: str | None
) -> tuple[str, bool]:
    """
    AC-1.5: ``(work_setting, work_setting_inferred)``.

    `remote_allowed` is a **sparse flag**, not a nullable boolean - measured
    2026-09-08 it takes exactly two values, 1.0 (15,246 rows) and null (108,603).
    Null therefore means "not flagged remote", never "unknown": routing it to
    Unknown would place 87.7% of the corpus there and, under AC-6.6's
    pass-and-tag policy, silently turn the work-setting filter into a no-op.
    """
    if remote_allowed == 1:
        return "Remote", False
    haystack = f"{title or ''} {description or ''}"
    if _HYBRID_RE.search(haystack):
        return "Hybrid", True
    return "On-site", True


def normalize_employment_type(formatted_work_type: str | None) -> str | None:
    """AC-1.6a: retained type, or None for rows excluded from the corpus."""
    if not formatted_work_type:
        return None
    value = str(formatted_work_type).strip()
    return value if value in EMPLOYMENT_TYPES else None


def parse_min_years(
    description: str | None, experience_level: str | None
) -> tuple[float | None, str]:
    """
    AC-1.6: ``(min_years_exp, min_years_exp_source)``.

    A numeric requirement in the description wins over the seniority label. The
    first stated figure is taken - it is the requirement line, whereas later
    figures are usually "X years preferred". For a range, the **lower** bound is
    taken (AC-1.6 v1.3): the field is a minimum, so "4-7 years" means 4. Values
    above ``MAX_PLAUSIBLE_YEARS`` are treated as no match.
    """
    for raw in _YEARS_RE.findall(description or ""):
        years = float(raw)
        if 0 < years <= MAX_PLAUSIBLE_YEARS:
            return years, "description_regex"

    if experience_level in SENIORITY_YEARS:
        return SENIORITY_YEARS[experience_level], "seniority_label"

    return None, "none"


def parse_education(description: str | None) -> int | None:
    """
    AC-1.7: lowest degree ordinal stated in the description, or None.

    Lowest wins because "Bachelor's required, Master's preferred" requires a
    bachelor's. Only explicit degree phrasings match - bare `bs`/`ms`/`ba`/`ma`
    are excluded so that "MS Office" does not read as a Master's degree.
    """
    if not description:
        return None
    found = [lvl for lvl, rx in _EDUCATION_RES if rx.search(description)]
    return min(found) if found else None


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


def normalize_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    AC-1.4 through AC-1.7: apply the per-row rules over the scoped corpus.

    Rows whose employment type is excluded by AC-1.6a (Volunteer, Other) are
    dropped here and counted. Returns the normalized frame plus the per-path
    counts AC-1.8 will persist to coverage_stats.json.
    """
    out = df.copy()

    out["employment_type"] = out["formatted_work_type"].map(normalize_employment_type)
    n_before = len(out)
    out = out[out["employment_type"].notna()].copy()
    n_excluded_employment = n_before - len(out)

    salary = out.apply(
        lambda r: normalize_salary(
            r.get("normalized_salary"), r.get("min_salary"), r.get("max_salary"),
            r.get("pay_period"), r.get("currency"),
        ),
        axis=1, result_type="expand",
    )
    out[["salary_min", "salary_max", "salary_listed", "salary_source"]] = salary

    setting = out.apply(
        lambda r: derive_work_setting(r.get("remote_allowed"), r.get("title"),
                                      r.get("description")),
        axis=1, result_type="expand",
    )
    out[["work_setting", "work_setting_inferred"]] = setting

    years = out.apply(
        lambda r: parse_min_years(r.get("description"), r.get("formatted_experience_level")),
        axis=1, result_type="expand",
    )
    out[["min_years_exp", "min_years_exp_source"]] = years

    out["education_required"] = out["description"].map(parse_education)

    stats = {
        "n_excluded_employment_type": n_excluded_employment,
        "salary_source": out["salary_source"].value_counts().to_dict(),
        "min_years_exp_source": out["min_years_exp_source"].value_counts().to_dict(),
        "work_setting": out["work_setting"].value_counts().to_dict(),
        "employment_type": out["employment_type"].value_counts().to_dict(),
        "education_required_pct": round(100 * out["education_required"].notna().mean(), 1),
        "salary_listed_pct": round(100 * out["salary_listed"].mean(), 1),
    }
    return out, stats


#: REQ-1 normalized schema — the columns phase 1 owns. `required_skills` and
#: `preferred_skills` are added by REQ-2; `location_city/state/lat/lon` by REQ-6's
#: geocoding. They are absent here rather than written as empty placeholders.
CORPUS_SCHEMA = [
    "job_id", "title", "title_normalized", "company", "location_raw", "is_remote",
    "work_setting", "work_setting_inferred", "employment_type", "min_years_exp",
    "min_years_exp_source", "education_required", "salary_min", "salary_max",
    "salary_listed", "salary_source", "required_skills", "preferred_skills",
    "n_required_skills", "description", "posted_date", "expiry_date", "posting_url",
]


def build_corpus(
    postings_path: Path | str,
    job_skills_path: Path | str,
    out_dir: Path | str = "data/processed",
) -> tuple[pd.DataFrame, dict]:
    """
    AC-1.8 / AC-1.9: run REQ-1 end to end and persist the corpus and its stats.

    Writes ``jobs_tech.parquet`` and ``coverage_stats.json`` under *out_dir*.
    Deterministic: dedupe breaks ties on ``job_id`` and the frame is ordered by
    ``job_id``, so repeated runs produce identical output (AC-1.9).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scoped = scope_and_dedupe(postings_path, job_skills_path)
    raw = scoped.frame
    skills_desc_pct = (
        round(100 * raw["skills_desc"].notna().mean(), 1)
        if "skills_desc" in raw.columns else 0.0
    )

    df, field_stats = normalize_fields(raw)

    # REQ-2: extract skills, then AC-2.6 drops postings with none.
    vocab = load_vocabulary()
    sections = df["description"].map(split_sections)
    extracted = df.apply(
        lambda r: extract_skills(r.get("description"), r.get("skills_desc"), vocab),
        axis=1, result_type="expand",
    )
    df["required_skills"] = extracted[0].map(sorted)
    df["preferred_skills"] = extracted[1].map(sorted)
    df["n_required_skills"] = df["required_skills"].str.len()

    skill_stats = {
        "pct_with_required_section": round(
            100 * sections.map(lambda x: bool(x["required"])).mean(), 1),
        "pct_with_preferred_section": round(
            100 * sections.map(lambda x: bool(x["preferred"])).mean(), 1),
        "pct_with_preferred_skills": round(
            100 * df.loc[df.n_required_skills > 0, "preferred_skills"].str.len().gt(0).mean(), 1),
        "median_required_skills": int(
            df.loc[df.n_required_skills > 0, "n_required_skills"].median()),
        "pct_with_one_required_skill": round(
            100 * df.loc[df.n_required_skills > 0, "n_required_skills"].eq(1).mean(), 1),
    }

    n_before_skills = len(df)
    df = df[df["n_required_skills"] > 0].copy()      # AC-2.6
    n_zero_skill_dropped = n_before_skills - len(df)

    df = df.rename(columns={"company_name": "company", "location": "location_raw"})
    df["is_remote"] = df["work_setting"].eq("Remote")
    df["posted_date"] = pd.to_datetime(
        df.get("listed_time"), unit="ms", errors="coerce"
    ).dt.date
    # AC-1.11: the source posting and when it closed. Every posting in this
    # historical corpus has expired, and the UI has to be able to say so.
    # `df.get` yields None for an absent column, and pd.to_datetime(None) is not
    # a Series - so guard rather than assume every source carries these.
    expiry = df["expiry"] if "expiry" in df.columns else pd.Series(pd.NaT, index=df.index)
    df["expiry_date"] = pd.to_datetime(expiry, unit="ms", errors="coerce").dt.date
    df["posting_url"] = (df["job_posting_url"] if "job_posting_url" in df.columns
                         else pd.Series(None, index=df.index, dtype="object"))
    for col in CORPUS_SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[CORPUS_SCHEMA].sort_values("job_id").reset_index(drop=True)

    n_final = len(df)
    stats = {
        "funnel": {
            "raw_load": scoped.n_raw,
            "after_tech_scoping": scoped.n_after_scoping,
            "after_dedupe": scoped.n_after_dedupe,
            "after_employment_filter": n_before_skills,
            "after_zero_skill_drop": n_final,
        },
        "dropped": {
            "non_tech": scoped.n_dropped_non_tech,
            "duplicates": scoped.n_dropped_duplicates,
            "excluded_employment_type": field_stats["n_excluded_employment_type"],
            "zero_skill": n_zero_skill_dropped,
        },
        "coverage_pct": {
            "salary_min": round(100 * df["salary_min"].notna().mean(), 1),
            "education_required": round(100 * df["education_required"].notna().mean(), 1),
            "min_years_exp": round(100 * df["min_years_exp"].notna().mean(), 1),
            # AC-1.5: Unknown is expected to be empty on the LinkedIn corpus.
            "work_setting_known": round(100 * df["work_setting"].ne("Unknown").mean(), 1),
            "skills_desc": skills_desc_pct,
        },
        "skills": skill_stats,
        # Every distribution below is computed from the FINAL corpus, after all
        # drops, so it describes the same population as coverage_pct above.
        "salary_source": df["salary_source"].value_counts().to_dict(),
        "min_years_exp_source": df["min_years_exp_source"].value_counts().to_dict(),
        "work_setting": df["work_setting"].value_counts().to_dict(),
        "employment_type": df["employment_type"].value_counts().to_dict(),
    }

    df.to_parquet(out_dir / "jobs_tech.parquet", index=False)
    (out_dir / "coverage_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return df, stats


def main() -> None:
    df, stats = build_corpus("data/raw/postings.csv", "data/raw/jobs/job_skills.csv")
    f, d = stats["funnel"], stats["dropped"]

    class _R:
        n_raw = f["raw_load"]; n_after_scoping = f["after_tech_scoping"]
        n_after_dedupe = f["after_dedupe"]
        n_dropped_non_tech = d["non_tech"]; n_dropped_duplicates = d["duplicates"]
    result = _R()
    pct = 100 * result.n_after_dedupe / result.n_raw
    print("\nREQ-1 phase 1a funnel (AC-1.1, AC-1.2, AC-1.3)")
    print("-" * 52)
    print(f"  raw postings              {result.n_raw:>8,}")
    print(f"  - non-tech dropped        {result.n_dropped_non_tech:>8,}")
    print(f"  = after tech scoping      {result.n_after_scoping:>8,}")
    print(f"  - duplicates dropped      {result.n_dropped_duplicates:>8,}")
    print(f"  = corpus                  {result.n_after_dedupe:>8,}   ({pct:.1f}% of raw)")
    print(f"  - non-job types dropped   {d['excluded_employment_type']:>8,}   (AC-1.6a)")
    print(f"  - zero-skill dropped      {d['zero_skill']:>8,}   (AC-2.6)")
    print(f"  = CORPUS                  {len(df):>8,}")
    k = stats["skills"]
    print("\nREQ-2 skill extraction (AC-2.6, AC-2.7)")
    print("-" * 52)
    print(f"  parsed required section   {k['pct_with_required_section']:>7}%")
    print(f"  parsed preferred section  {k['pct_with_preferred_section']:>7}%")
    print(f"  has preferred skills      {k['pct_with_preferred_skills']:>7}%")
    print(f"  median required skills    {k['median_required_skills']:>7}")
    print(f"  exactly 1 required skill  {k['pct_with_one_required_skill']:>7}%")

    print("\nREQ-1 field coverage over the FINAL corpus (AC-1.4 - AC-1.7)")
    print("-" * 52)
    print(f"  salary listed             {stats['coverage_pct']['salary_min']:>7}%")
    for k, v in stats["salary_source"].items():
        print(f"      via {k:<20}{v:>8,}")
    print(f"  education parsed          {stats['coverage_pct']['education_required']:>7}%")
    print("  experience source")
    for k, v in stats["min_years_exp_source"].items():
        print(f"      {k:<24}{v:>8,}")
    print("  work setting")
    for k, v in stats["work_setting"].items():
        print(f"      {k:<24}{v:>8,}")
    print("  employment type")
    for k, v in stats["employment_type"].items():
        print(f"      {k:<24}{v:>8,}")
    print("\n  wrote data/processed/jobs_tech.parquet  and  coverage_stats.json")



if __name__ == "__main__":
    main()
