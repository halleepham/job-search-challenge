"""
End-to-end search: profile in, ranked jobs with evidence out.

Composes the frozen requirements in the order D2 settled:

    FILTER (REQ-6) -> RETRIEVE (REQ-7) -> SCORE (REQ-9) -> RANK -> EVIDENCE (REQ-4)

Returns per-stage timings for AC-13.3 and the filter funnel for AC-11.2, so the
UI and the evaluation notebook read the same numbers the pipeline actually
produced rather than recomputing them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.filters import apply_filters
from src.indexing import JobIndex
from src.personal_kb import PersonalKB, embed
from src.profiles import UserProfile
from src.retrieval import retrieve
from src.scoring import score_job

#: Candidates that receive full scoring. Retrieval is cheap and approximate;
#: scoring is exact and explainable, so the shortlist only needs to be generous.
RETRIEVE_K = 200
TOP_N = 5


@dataclass
class SearchResult:
    results: list[dict]
    funnel: dict
    timings_ms: dict[str, float] = field(default_factory=dict)


def search(
    jobs: pd.DataFrame,
    index: JobIndex,
    profile: UserProfile,
    kb: PersonalKB | None = None,
    mode: str = "hybrid",
    top_n: int = TOP_N,
    retrieve_k: int = RETRIEVE_K,
) -> SearchResult:
    """
    Run one search. *jobs* and *index* must describe the same corpus in the same
    order - positional indexes are the join key between them.
    """
    timings: dict[str, float] = {}
    position_of = {job_id: i for i, job_id in enumerate(jobs["job_id"])}

    t = time.perf_counter()
    kept, funnel = apply_filters(jobs, profile)
    timings["filter"] = (time.perf_counter() - t) * 1000
    if kept.empty:
        return SearchResult([], funnel, timings)

    t = time.perf_counter()
    candidates = [position_of[j] for j in kept["job_id"]]
    hits = retrieve(index, profile, candidate_positions=candidates, mode=mode, k=retrieve_k)
    timings["retrieve"] = (time.perf_counter() - t) * 1000

    # Semantic sub-scores, calibrated against the fixed background (AC-9.11).
    t = time.perf_counter()
    positions = np.array([p for p, _ in hits], dtype=int)
    job_vecs = index.embeddings[positions]

    goals_vec = embed([profile.career_goals])[0] if profile.career_goals else None
    goals_raw = job_vecs @ goals_vec if goals_vec is not None else np.zeros(len(positions))

    if kb is not None and len(kb.chunks):
        evidence_raw = (job_vecs @ kb.embeddings.T).max(axis=1)
    else:
        evidence_raw = np.zeros(len(positions))

    scored = []
    for offset, position in enumerate(positions):
        scored.append(score_job(
            jobs.iloc[position], profile,
            goals_similarity=index.calibration.apply(float(goals_raw[offset])),
            evidence_similarity=index.calibration.apply(float(evidence_raw[offset])),
        ))
    timings["score"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    scored.sort(key=lambda r: (-r["score"], r["job_id"]))
    top = scored[:top_n]
    timings["rank"] = (time.perf_counter() - t) * 1000

    # AC-11.5: attach the resume chunk supporting each result, sliced verbatim
    # from the source by character span - never generated.
    t = time.perf_counter()
    for result in top:
        row = jobs.iloc[position_of[result["job_id"]]]
        result["job"] = row
        result["evidence"] = []
        if kb is not None and len(kb.chunks):
            job_text = f"{row['title']}. {row['description'] or ''}"
            result["evidence"] = [
                {"text": chunk.text, "section": chunk.section,
                 "char_span": chunk.char_span, "similarity": round(sim, 3)}
                for chunk, sim in kb.retrieve(job_text, k=2)
            ]
    timings["evidence"] = (time.perf_counter() - t) * 1000

    funnel["retrieved"] = len(hits)
    funnel["returned"] = len(top)
    timings["total"] = sum(timings.values())
    return SearchResult(top, funnel, timings)
