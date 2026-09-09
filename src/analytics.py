"""
REQ-12: dataset analytics (FROZEN v1.0).

Descriptive analysis *of the corpus*, distinct from the result-level charts on
the results page. Computed as DuckDB aggregations (D9) and written to small CSVs
that are committed, so the report can quote exact numbers without a pipeline run.

Every output is labelled with what the corpus actually is (D3): LinkedIn's IT,
engineering, analytics, QA and science job functions - roughly a quarter of which
is strictly software/data work. Calling it "tech jobs" would overstate it.
"""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

from src.filters import parse_location
from src.seq import as_list

OUT_DIR = Path("data/processed/analytics")

CORPUS_LABEL = (
    "LinkedIn IT, engineering, analytics, QA and science job functions "
    "(~25% strictly software/data)"
)

#: Title families for the salary breakdown. First match wins, so order matters:
#: "data engineer" must be tested before the broader "engineer".
TITLE_FAMILIES = [
    ("Data Engineer", r"data engineer|etl|data pipeline"),
    ("Data Scientist / ML", r"data scientist|machine learning|\bml\b|scientist"),
    ("Data Analyst / BI", r"data analyst|business intelligence|analytics"),
    ("Software Engineer", r"software|developer|programmer|full[ -]?stack|backend|frontend"),
    ("DevOps / Cloud / SRE", r"devops|\bsre\b|site reliability|cloud|infrastructure|platform"),
    ("Security", r"security|cyber|threat|soc\b"),
    ("QA / Test", r"\bqa\b|quality assurance|test engineer"),
    ("Other Engineering", r"engineer"),
    ("Other", r"."),
]


def title_family(title: str | None) -> str:
    """First-match-wins family label. Order in TITLE_FAMILIES is significant."""
    t = (title or "").lower()
    for name, pattern in TITLE_FAMILIES:
        if pd.Series([t]).str.contains(pattern, regex=True, na=False).iloc[0]:
            return name
    return "Other"


def state_of(location_raw: str | None) -> str | None:
    """State code for the geographic distribution, via REQ-6's parser."""
    return parse_location(location_raw)[1]


def compute(jobs: pd.DataFrame, out_dir: Path | str = OUT_DIR) -> dict[str, pd.DataFrame]:
    """
    AC-12.1: the seven corpus aggregations, written to `out_dir` as CSVs.

    Set-oriented work in SQL (D9); the two per-row derivations that SQL cannot
    express cleanly - state parsing and title-family labelling - are the same
    pure Python functions used elsewhere.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = jobs.copy()
    df["state"] = df["location_raw"].map(state_of)
    df["title_family"] = df["title"].map(title_family)
    df["n_skills"] = df["required_skills"].map(lambda v: len(as_list(v)))

    con = duckdb.connect()
    con.register("jobs", df.drop(columns=["required_skills", "preferred_skills"],
                                 errors="ignore"))

    tables = {
        "top_titles": """
            SELECT title, count(*) AS n FROM jobs
            GROUP BY title ORDER BY n DESC LIMIT 20""",
        "geographic_distribution": """
            SELECT state, count(*) AS n FROM jobs WHERE state IS NOT NULL
            GROUP BY state ORDER BY n DESC""",
        "salary_by_title_family": """
            SELECT title_family,
                   count(*) AS n_jobs,
                   sum(CASE WHEN salary_listed THEN 1 ELSE 0 END) AS n_with_salary,
                   round(median(salary_max)) AS median_salary_max,
                   round(median(salary_min)) AS median_salary_min
            FROM jobs GROUP BY title_family ORDER BY n_jobs DESC""",
        "work_setting": """
            SELECT work_setting, count(*) AS n FROM jobs
            GROUP BY work_setting ORDER BY n DESC""",
        "experience_distribution": """
            SELECT min_years_exp AS years_required, count(*) AS n FROM jobs
            WHERE min_years_exp IS NOT NULL
            GROUP BY min_years_exp ORDER BY years_required""",
        "top_companies": """
            SELECT company, count(*) AS n FROM jobs WHERE company IS NOT NULL
            GROUP BY company ORDER BY n DESC LIMIT 20""",
    }
    results = {name: con.execute(sql).df() for name, sql in tables.items()}
    con.close()

    # Skill frequency and co-occurrence come from the list column, so they are
    # computed in pandas rather than SQL.
    counts: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for skills in df["required_skills"]:
        items = sorted(set(as_list(skills)))
        counts.update(items)
        pairs.update(itertools.combinations(items[:12], 2))   # cap: guards O(n^2) blowup

    results["top_skills"] = pd.DataFrame(
        counts.most_common(25), columns=["skill", "n"])
    results["skill_cooccurrence"] = pd.DataFrame(
        [(a, b, n) for (a, b), n in pairs.most_common(40)],
        columns=["skill_a", "skill_b", "n"])

    for name, table in results.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "_corpus_label.txt").write_text(CORPUS_LABEL + f"\nn_jobs={len(df)}\n")
    return results
