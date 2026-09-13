"""
REQ-11 v1.1: presentation helpers for the explainable-match card.

Pure functions over a scored result, so the panels the UI renders are testable
without Streamlit and cannot drift from what the scorer actually produced.
"""

from __future__ import annotations

import pandas as pd

from src.scoring import WEIGHTS
from src.seq import as_list

COMPONENT_LABELS = {
    "required_skills": "Skills", "preferred_skills": "Preferred skills",
    "experience": "Experience", "education": "Education", "title": "Job title",
    "location": "Location", "career_goals_similarity": "Career goals",
    "resume_evidence_similarity": "Résumé evidence",
}

EVIDENCE_SECTION_LABELS = {
    "education": "Education", "skills": "Skills", "experience": "Experience",
    "projects": "Project", "certifications": "Certification",
    "career_goals": "Career goals", "other": "Résumé",
}


def component_bars(result: dict) -> pd.DataFrame:
    """
    AC-11.10: earned points out of maximum, per component.

    `max_points` uses the *effective* weight after AC-9.4's renormalization, so a
    dropped component shows 0/0 rather than appearing to have been failed.
    """
    exact = [c["weighted"] * 100 for c in result["components"]]

    # Largest-remainder allocation so the displayed points sum *exactly* to the
    # displayed score. Rounding each component independently and adding them does
    # not: eight values rounded to 0.1 drifted to 78.5 against a card reading 79,
    # and a breakdown that does not add up is not a breakdown.
    tenths = [int(v * 10) for v in exact]
    residual = round(result["score"] * 10) - sum(tenths)

    # Rows with the largest fractional remainder absorb a surplus; rows with the
    # smallest absorb a shortfall. Two things make a naive pass wrong: a negative
    # residual (the exact total rounds *down*, which is common), and rows already
    # at zero - dropped components have both the smallest remainder and nothing
    # to give, so they would silently swallow a decrement. Cycle until the
    # residual is actually consumed.
    order = sorted(range(len(exact)), key=lambda i: -((exact[i] * 10) - tenths[i]))
    if residual < 0:
        order = order[::-1]
    step = 1 if residual > 0 else -1
    remaining, guard = abs(residual), 0
    while remaining and guard < 1000:
        for i in order:
            if not remaining:
                break
            if step < 0 and tenths[i] == 0:
                continue                     # nothing left to take from this row
            tenths[i] += step
            remaining -= 1
        guard += 1

    rows = []
    for c, points in zip(result["components"], tenths):
        rows.append({
            "Component": COMPONENT_LABELS[c["name"]],
            "Earned": round(points / 10, 1),
            "Max": round(c["weight"] * 100, 1),
            "Fraction": (c["weighted"] / c["weight"]) if c["weight"] else 0.0,
            "Dropped": c["sub_score"] is None,
        })
    return pd.DataFrame(rows)


def _strip_leading_heading(text: str, label: str) -> str:
    """
    Drop the chunk's own section heading when the Type column already says it.

    Paragraph chunking (AC-4.1) keeps the heading line inside the chunk, so a
    Skills block renders as "Skills | Skills Python, SQL…" without this. The
    heading is only removed for display; `char_span` still points at the full
    verbatim chunk.
    """
    body = text.strip()
    first, sep, rest = body.partition("\n")
    if sep and first.strip().lower().rstrip(":") == label.lower():
        body = rest.strip()
    return " ".join(body.split())[:220]


def evidence_table(result: dict, profile) -> pd.DataFrame:
    """
    AC-11.9: typed evidence with its source.

    Résumé rows are verbatim spans (AC-4.2); job-requirement and profile rows
    state where the fact came from, so nothing on this table is generated text.
    """
    job = result["job"]
    rows = []

    for ev in result.get("evidence", []):
        label = EVIDENCE_SECTION_LABELS.get(ev["section"], ev["section"].title())
        rows.append({
            "Type": label,
            "Evidence": _strip_leading_heading(ev["text"], label),
            "Source": "Résumé",
        })

    if result["matched_skills"]:
        rows.append({"Type": "Skills matched",
                     "Evidence": ", ".join(result["matched_skills"]),
                     "Source": "Profile ∩ job description"})

    required = as_list(job.get("required_skills"))
    if required:
        rows.append({"Type": "Job requirement",
                     "Evidence": ", ".join(required[:12]),
                     "Source": "Job description"})

    rows.append({"Type": "Location",
                 "Evidence": f"{job['location_raw']} · {job['work_setting']}"
                             f"{' (inferred)' if job.get('work_setting_inferred') else ''}",
                 "Source": "Job posting"})

    if profile.preferred_location:
        rows.append({"Type": "Candidate location",
                     "Evidence": f"{profile.preferred_location} · accepts "
                                 f"{', '.join(sorted(profile.accepted_work_settings))}",
                     "Source": "Profile"})
    return pd.DataFrame(rows)


def gaps_and_unknowns(result: dict, profile) -> pd.DataFrame:
    """
    AC-11.11: two different things in one table.

    *Gaps* are qualifications the user lacks. *Unknowns* are facts the posting
    never stated, where the score fell back to a neutral default. Both change
    what a reader should do next, and the second is invisible without this table —
    a score built partly on neutral defaults must disclose them or it implies a
    confidence the data does not support.
    """
    job = result["job"]
    rows = []

    missing = result["missing_skills"]
    if missing:
        share = len(missing) / max(1, len(as_list(job.get("required_skills"))))
        rows.append({
            "Issue": "Missing skills", "Kind": "Gap",
            "Details": ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""),
            # High means *more* than half the requirements are unmet; exactly
            # half is a real gap but not a disqualifying one.
            "Impact": "High" if share > 0.5 else "Medium",
        })

    if not job.get("salary_listed"):
        rows.append({"Issue": "Salary not listed", "Kind": "Unknown",
                     "Details": "The posting states no salary; it passed your filter "
                                "because you chose to include unlisted salaries.",
                     "Impact": "Medium"})
    elif profile.min_salary and job.get("salary_max", 0) < profile.min_salary * 1.1:
        rows.append({"Issue": "Salary near your floor", "Kind": "Gap",
                     "Details": f"Top of range ${job['salary_max']:,.0f} is close to your "
                                f"${profile.min_salary:,.0f} minimum.",
                     "Impact": "Low"})

    if job.get("work_setting_inferred"):
        rows.append({"Issue": "Work setting inferred", "Kind": "Unknown",
                     "Details": f"“{job['work_setting']}” was derived from the posting text, "
                                "not stated in a structured field.",
                     "Impact": "Low"})

    if job.get("education_required") is None:
        rows.append({"Issue": "Education requirement not stated", "Kind": "Unknown",
                     "Details": "No degree requirement was parsed, so this component "
                                "scored neutral (0.5) rather than for or against you.",
                     "Impact": "Low"})

    source = job.get("min_years_exp_source")
    if source == "none":
        rows.append({"Issue": "Experience requirement not stated", "Kind": "Unknown",
                     "Details": "No numeric requirement and no seniority label; this "
                                "component scored neutral (0.5).",
                     "Impact": "Medium"})
    elif source == "seniority_label":
        rows.append({"Issue": "Experience inferred", "Kind": "Unknown",
                     "Details": f"Estimated {job['min_years_exp']:.0f} years from the posting's "
                                "seniority label, not from a stated figure.",
                     "Impact": "Low"})

    n_required = len(as_list(job.get("required_skills")))
    if n_required < 3:
        rows.append({"Issue": "Few requirements listed", "Kind": "Unknown",
                     "Details": f"Only {n_required} requirement(s) were extractable, so the "
                                "skills score is discounted for thin evidence.",
                     "Impact": "Medium"})

    return pd.DataFrame(rows, columns=["Issue", "Kind", "Details", "Impact"])


def verdict(result: dict, gaps: pd.DataFrame) -> tuple[str, str]:
    """AC-11.13: a headline and the one thing to check before applying."""
    headline = {
        "Strong": "Strong candidate", "Good": "Good candidate",
        "Moderate": "Possible candidate", "Low": "Weak candidate",
    }[result["tier"]]

    unknowns = gaps[gaps["Kind"] == "Unknown"]["Issue"].tolist() if len(gaps) else []
    blocking = gaps[(gaps["Kind"] == "Gap") & (gaps["Impact"] == "High")]["Issue"].tolist() \
        if len(gaps) else []

    if blocking:
        advice = f"Close the {blocking[0].lower()} before applying."
    elif unknowns:
        advice = "Verify " + " and ".join(u.lower() for u in unknowns[:2]) + " before applying."
    else:
        advice = "Nothing material is unstated in this posting."
    return headline, advice
