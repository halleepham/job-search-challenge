"""
REQ-7: hybrid retrieval (FROZEN v1.0).

BM25 and dense retrieval run over the **filter survivors** (D2) and are fused
with Reciprocal Rank Fusion.

Two design points the draft plan got right and this keeps:

* Three *separate* queries - career goals, skills, preferred titles - rather than
  one concatenated blob. A 200-word blob produces a fuzzier dense vector than
  three focused ones, so concatenation discards the very signal being retrieved on.
* Fusion by **rank**, never by summing raw scores. BM25 scores and cosine
  similarities live on incompatible scales, and normalizing them across queries
  is fragile; RRF only needs the ordering.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.indexing import JobIndex
from src.personal_kb import embed
from src.profiles import UserProfile

#: AC-7.2: the standard RRF constant. Damps the influence of any single
#: retriever's top position, so agreement across retrievers outweighs one
#: retriever's confidence.
RRF_K = 60

MODES = ("bm25", "dense", "hybrid")


def build_queries(profile: UserProfile) -> dict[str, str]:
    """
    AC-7.1: the three query facets, empty ones omitted.

    Kept separate so each is fused as its own ranked list - concatenating them
    would dilute the dense embedding.
    """
    queries = {
        "career_goals": (profile.career_goals or "").strip(),
        "skills": " ".join(sorted(profile.skills)),
        "titles": " ".join(profile.preferred_titles or []),
    }
    return {k: v for k, v in queries.items() if v.strip()}


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """
    AC-7.2: ``score(doc) = sum over lists of 1 / (k + rank)``, rank from 1.

    Scores in the input lists are ignored entirely - only position matters. That
    is the point: it makes fusion immune to the incompatible scales of BM25 and
    cosine.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked, start=1):
            fused[doc] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def _dense_ranked(index: JobIndex, query: str, positions: np.ndarray, k: int):
    scores = index.embeddings[positions] @ embed([query])[0]
    order = np.argsort(-scores)[: min(k, len(scores))]
    return [(int(positions[i]), float(scores[i])) for i in order]


def _bm25_ranked(index: JobIndex, query: str, positions: np.ndarray, k: int):
    all_scores = np.asarray(index.bm25.get_scores(query.lower().split()))
    scores = all_scores[positions]
    order = np.argsort(-scores)[: min(k, len(scores))]
    return [(int(positions[i]), float(scores[i])) for i in order]


def retrieve(
    index: JobIndex,
    profile: UserProfile,
    candidate_positions: list[int] | None = None,
    mode: str = "hybrid",
    k: int = 200,
) -> list[tuple[int, float]]:
    """
    Top-*k* candidate positions for *profile*, restricted to
    *candidate_positions* (the filter survivors).

    ``mode`` is one of ``bm25`` | ``dense`` | ``hybrid``. Each is independently
    callable (AC-7.3) so REQ-13 can compare them without reconstructing the
    pipeline. AC-7.4: a subset smaller than *k* returns everything it has rather
    than raising.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    positions = (np.arange(len(index.embeddings)) if candidate_positions is None
                 else np.asarray(candidate_positions, dtype=int))
    if positions.size == 0:
        return []

    queries = build_queries(profile)
    if not queries:
        return [(int(p), 0.0) for p in positions[:k]]

    ranked_lists = []
    for query in queries.values():
        if mode in ("dense", "hybrid"):
            ranked_lists.append(_dense_ranked(index, query, positions, k))
        if mode in ("bm25", "hybrid"):
            ranked_lists.append(_bm25_ranked(index, query, positions, k))

    return reciprocal_rank_fusion(ranked_lists)[:k]
