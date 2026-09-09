"""
REQ-13: method evaluation (FROZEN v1.0).

Evaluation of *method quality*, distinct from the AC-derived tests, which verify
*spec compliance*. Logic lives here rather than in the notebook so it is testable
and so the notebook stays readable.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

#: Anchored to the repo root, not the working directory - the notebook runs from
#: notebooks/ while scripts run from the root, and relative paths silently wrote
#: to notebooks/notebooks/.
_ROOT = Path(__file__).resolve().parents[1]
FIGURES = _ROOT / "notebooks" / "figures"
RESULTS = _ROOT / "notebooks" / "results"
_RAW = _ROOT / "data" / "raw"

#: AC-13.1: how relevance is judged when no human labels are supplied.
#: A documented proxy, never presented as human judgment - `pooled_candidates()`
#: writes the pool to CSV so real labels can replace it.
RELEVANCE_PROXY_NOTE = (
    "Automatic proxy: a posting counts as relevant if its title family matches the "
    "profile's target domain. Weaker than human judgment and stated as such; "
    "pooled_candidates() exports the pool for manual labelling."
)

PROFILE_DOMAINS = {
    "data_science_student": {"Data Engineer", "Data Scientist / ML", "Data Analyst / BI"},
    "senior_backend_engineer": {"Software Engineer", "DevOps / Cloud / SRE"},
    "security_analyst": {"Security"},
}


def _domain_relevant(title: str, profile_key: str) -> bool:
    from src.analytics import title_family

    return title_family(title) in PROFILE_DOMAINS.get(profile_key, set())


def pooled_candidates(jobs, index, profile, profile_key: str, k: int = 10) -> pd.DataFrame:
    """
    AC-13.1: the pooled judgment set — the union of each configuration's top-k.

    Pooling rather than pre-labelling a fixed set is both less manual work and
    methodologically sounder: it cannot under-credit a configuration for
    surfacing a relevant job nobody thought to label in advance.
    """
    from src.retrieval import retrieve

    pool: dict[int, dict] = {}
    for mode in ("bm25", "dense", "hybrid"):
        for rank, (position, _) in enumerate(retrieve(index, profile, mode=mode, k=k), 1):
            row = pool.setdefault(position, {
                "position": position, "job_id": jobs.iloc[position]["job_id"],
                "title": jobs.iloc[position]["title"], "bm25": None, "dense": None, "hybrid": None})
            row[mode] = rank

    frame = pd.DataFrame(pool.values())
    frame["relevant_proxy"] = frame["title"].map(lambda t: _domain_relevant(t, profile_key))
    frame["relevant_manual"] = ""          # for human labelling
    return frame.sort_values("position").reset_index(drop=True)


def precision_at_k(pool: pd.DataFrame, k: int = 10, label: str = "relevant_proxy") -> dict:
    """AC-13.1: precision@k per retrieval mode, scored against the shared pool."""
    out = {}
    for mode in ("bm25", "dense", "hybrid"):
        ranked = pool[pool[mode].notna()].nsmallest(k, mode)
        out[mode] = round(ranked[label].mean(), 3) if len(ranked) else 0.0
    return out


def recall_of_top_scored(jobs, index, profile, kb, ks=(10, 25, 50, 100, 200),
                        n_relevant: int = 20) -> pd.DataFrame:
    """
    AC-13.1 (v1.1): does retrieval surface the jobs that scoring would rank highest?

    Score **every** filter survivor, take the true top-`n_relevant` by final match
    score, then report each retrieval mode's recall@k against that set.

    Not circular: retrieval ranks by BM25 and embedding similarity, scoring ranks
    by eight weighted components including skills, experience, education and
    location. The question is whether the cheap approximate stage keeps what the
    expensive exact stage wants - which is the only thing the retrieval stage is
    for.

    Replaces a title-family relevance proxy that returned precision@10 = 1.00 for
    both BM25 and hybrid on every profile, and so could not separate them.
    """
    from src.filters import apply_filters
    from src.pipeline import search
    from src.retrieval import retrieve

    kept, _ = apply_filters(jobs, profile)
    if kept.empty:
        return pd.DataFrame()

    # Exhaustive scoring of the survivors defines the ground truth.
    full = search(jobs, index, profile, kb, top_n=n_relevant, retrieve_k=len(kept))
    truth = {r["job_id"] for r in full.results}

    position_of = {jid: i for i, jid in enumerate(jobs["job_id"])}
    candidates = [position_of[j] for j in kept["job_id"]]

    rows = []
    for mode in ("bm25", "dense", "hybrid"):
        ranked = retrieve(index, profile, candidate_positions=candidates,
                          mode=mode, k=max(ks))
        ids = [jobs.iloc[p]["job_id"] for p, _ in ranked]
        for k in ks:
            found = len(truth & set(ids[:k]))
            rows.append({"mode": mode, "k": k,
                         "recall": round(found / len(truth), 3) if truth else 0.0,
                         "found": found, "of": len(truth)})
    return pd.DataFrame(rows)


def sensitivity_all_profiles(jobs, index, presets, kbs) -> pd.DataFrame:
    """
    AC-13.6 (v1.1): component sensitivity across every reference profile.

    A single profile's result is not a property of the weight. The data-science
    profile's top results are mostly remote, so location is constant *for it* -
    a profile that rejects remote would see the same component discriminate.
    """
    frames = []
    for key, profile in presets.items():
        frame = component_sensitivity(jobs, index, profile, kbs[key])
        frame.insert(0, "profile", profile.name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["jobs_replaced"] = 5 - combined["top5_overlap"]
    summary = (combined.groupby("component")
               .agg(weight=("weight", "first"),
                    mean_jobs_replaced=("jobs_replaced", "mean"),
                    profiles_affected=("order_changed", "sum"),
                    n_profiles=("order_changed", "size"))
               .reset_index().sort_values("mean_jobs_replaced", ascending=False))
    return combined, summary


def latency_profile(jobs, index, profile, kb, runs: int = 20) -> pd.DataFrame:
    """AC-13.3: per-stage median and p95 over repeated searches."""
    from src.pipeline import search

    samples: list[dict] = []
    for _ in range(runs):
        samples.append(search(jobs, index, profile, kb).timings_ms)
    frame = pd.DataFrame(samples)
    return pd.DataFrame({
        "stage": frame.columns,
        "median_ms": frame.median().round(1).values,
        "p95_ms": frame.quantile(0.95).round(1).values,
    })


def scalability(postings_path: str | Path | None = None,
                sizes: tuple[int, ...] = (10_000, 30_000, 60_000, 123_849)) -> pd.DataFrame:
    """
    AC-13.4: ingestion wall-clock and peak memory over increasing corpus size.

    Times the DuckDB scoping and dedupe stage, which is the part whose cost grows
    with the corpus. Each size is read from the same raw file so the comparison is
    like-for-like.
    """
    import tracemalloc

    import duckdb

    postings_path = str(postings_path or (_RAW / "postings.csv"))
    skills_path = str(_RAW / "jobs" / "job_skills.csv")
    rows = []
    for n in sizes:
        con = duckdb.connect()
        tracemalloc.start()
        t = time.perf_counter()
        con.execute(
            "CREATE TABLE p AS SELECT * FROM read_csv_auto(?, sample_size=100000, "
            "types={'job_id':'BIGINT'}) LIMIT ?", [postings_path, n])
        con.execute("CREATE TABLE js AS SELECT * FROM read_csv_auto(?, "
                    "types={'job_id':'BIGINT'})", [skills_path])
        con.execute("""SELECT count(*) FROM (
                         SELECT DISTINCT title, company_name, location FROM p
                         WHERE job_id IN (SELECT job_id FROM js
                                          WHERE skill_abr IN ('IT','ENG','ANLS','QA','SCI')))""")
        elapsed = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        con.close()
        rows.append({"rows": n, "seconds": round(elapsed, 2), "peak_mb": round(peak, 1),
                     "rows_per_sec": int(n / elapsed)})
    return pd.DataFrame(rows)


def calibration_evidence(index, profile, n: int = 4000) -> pd.DataFrame:
    """
    AC-13.5: raw cosine vs calibrated score over the same sample.

    The figure this produces is the direct answer to the Stage 2 AI code's
    `sim * 140.0` rescaling: same goal, measured constants instead of a chosen one.

    Reports both populations, because they show different halves of the point:
    over the *whole corpus* most jobs compress toward 0 (they are irrelevant, and
    should be), while over the *retrieved candidates* - the population the score
    is applied to - the values spread across the range. Showing only the corpus
    would make a working calibration look broken; showing only the candidates
    would hide what it does to everything else.
    """
    from src.personal_kb import embed

    vec = embed([profile.career_goals])[0]
    scores = index.embeddings @ vec
    calibration = index.calibration_for_scores(scores)

    rng = np.random.default_rng(0)
    corpus = rng.choice(scores, size=min(n, len(scores)), replace=False)
    retrieved = np.sort(scores)[-200:]

    frames = [
        pd.DataFrame({"population": "whole corpus", "raw_cosine": corpus}),
        pd.DataFrame({"population": "retrieved candidates", "raw_cosine": retrieved}),
    ]
    out = pd.concat(frames, ignore_index=True)
    out["calibrated"] = [calibration.apply(float(v)) for v in out["raw_cosine"]]
    return out


def component_sensitivity(jobs, index, profile, kb) -> pd.DataFrame:
    """
    AC-13.6: zero each component in turn and report how the top-5 changes.

    A component that never changes the top 5 is decoration regardless of its
    weight — this is what decides Q6 (education's 4%).
    """
    from src.pipeline import search
    from src.scoring import WEIGHTS

    baseline = [r["job_id"] for r in search(jobs, index, profile, kb).results]
    rows = []
    for name in WEIGHTS:
        original = WEIGHTS[name]
        WEIGHTS[name] = 0.0
        try:
            changed = [r["job_id"] for r in search(jobs, index, profile, kb).results]
        finally:
            WEIGHTS[name] = original
        rows.append({
            "component": name, "weight": original,
            "top5_overlap": len(set(baseline) & set(changed)),
            "order_changed": changed != baseline,
        })
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def semantic_correlation(jobs, index, profile, kb, k: int = 200) -> float:
    """
    AC-13.6 / Q2: correlation between the two semantic components.

    High correlation would mean 30% of the weight is one signal counted twice —
    which is exactly what AC-9.10's career-goals exclusion was introduced to fix.
    """
    from src.pipeline import search

    res = search(jobs, index, profile, kb, top_n=k)
    goals, evidence = [], []
    for r in res.results:
        by_name = {c["name"]: c["sub_score"] for c in r["components"]}
        g, e = by_name["career_goals_similarity"], by_name["resume_evidence_similarity"]
        if g is not None and e is not None:
            goals.append(g)
            evidence.append(e)
    if len(set(goals)) < 2 or len(set(evidence)) < 2:
        return float("nan")
    return float(np.corrcoef(goals, evidence)[0, 1])
