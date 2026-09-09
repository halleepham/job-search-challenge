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

#: Candidates that receive full scoring when retrieval is used.
RETRIEVE_K = 200

#: D13 - below this many filter survivors, skip retrieval and score all of them.
#: Measured: scoring every survivor costs ~220 ms more than retrieving 200 and
#: produced an identical top-5 for all three reference profiles, while retrieval
#: at k=200 recovered only 45-95% of the true top-20 by score (AC-13.1). Exact is
#: available for the price of a fifth of a second, so approximation is not worth
#: its recall loss at this corpus size. Retrieval remains for scalability - the
#: same argument as D9 choosing DuckDB over Spark: use the approximate machinery
#: when the data demands it, and measure rather than assume that it does.
SCORE_ALL_THRESHOLD = 2500
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
    if len(candidates) <= SCORE_ALL_THRESHOLD and mode == "hybrid":
        # D13: exact - every eligible job is scored, no candidate is discarded.
        hits = [(p, 0.0) for p in candidates]
        funnel["retrieval"] = "skipped (scored every survivor)"
    else:
        hits = retrieve(index, profile, candidate_positions=candidates, mode=mode, k=retrieve_k)
        funnel["retrieval"] = f"{mode} top-{retrieve_k}"
    timings["retrieve"] = (time.perf_counter() - t) * 1000

    # Semantic sub-scores, calibrated against the fixed background (AC-9.11).
    t = time.perf_counter()
    positions = np.array([p for p, _ in hits], dtype=int)
    job_vecs = index.embeddings[positions]

    # AC-5.5 v1.1: each semantic component is calibrated against its OWN
    # similarity distribution over the whole corpus, for THIS profile. Sharing
    # one set of anchors between two different query texts collapses one of them.
    goals_cal = evidence_cal = None
    goals_raw = np.zeros(len(positions))
    evidence_raw = None

    if profile.career_goals:
        goals_vec = embed([profile.career_goals])[0]
        goals_cal = index.calibration_for_scores(index.embeddings @ goals_vec)
        goals_raw = job_vecs @ goals_vec

    # AC-9.10: evidence excludes the career-goals chunk, which AC-9.9 already scores.
    if kb is not None and len(kb.chunks):
        mask = kb.evidence_mask
        if mask.any():
            chunk_vecs = kb.embeddings[mask]
            evidence_cal = index.calibration_for_scores(
                (index.embeddings @ chunk_vecs.T).max(axis=1))
            evidence_raw = (job_vecs @ chunk_vecs.T).max(axis=1)

    scored = []
    for offset, position in enumerate(positions):
        scored.append(score_job(
            jobs.iloc[position], profile,
            goals_similarity=(goals_cal.apply(float(goals_raw[offset]))
                              if goals_cal else None),
            evidence_similarity=(evidence_cal.apply(float(evidence_raw[offset]))
                                 if evidence_cal is not None else None),
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
        position = position_of[result["job_id"]]
        row = jobs.iloc[position]
        result["job"] = row
        result["evidence"] = []
        if kb is not None and len(kb.chunks):
            # AC-9.10: the job's stored index vector, not a re-embedded copy of
            # its text - so shown evidence and scored evidence are one measurement.
            result["evidence"] = [
                {"text": chunk.text, "section": chunk.section,
                 "char_span": chunk.char_span, "similarity": round(sim, 3)}
                for chunk, sim in kb.retrieve_by_vector(
                    index.embeddings[position], k=2, evidence_only=True)
            ]
    timings["evidence"] = (time.perf_counter() - t) * 1000

    funnel["retrieved"] = len(hits)
    funnel["returned"] = len(top)
    timings["total"] = sum(timings.values())
    return SearchResult(top, funnel, timings)
