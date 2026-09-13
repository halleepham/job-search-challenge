"""
REQ-8 v2.0: optional cross-encoder reranking.

**Off by default.** A cross-encoder reads the (profile, job) pair as one input and
attends across both, unlike the bi-encoder whose vectors are computed separately -
more accurate per pair, but one forward pass per pair, so it only runs over a
shortlist.

It stays off because at k=400 hybrid retrieval already recovers 100% of the true
top-20 by score, leaving nothing for a reranker to recover, and because a
cross-encoder score is undecomposable: it may prune, but it may never enter a
score built to be explainable (D10, AC-8.2).
"""

from __future__ import annotations

from functools import lru_cache

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def rerank_candidates(index, jobs, profile, hits, top_k: int = 50):
    """
    AC-8.1: re-order *hits* by joint (profile, job) relevance, keep the top_k.

    AC-8.2: the returned scores are deliberately zeroed. Nothing downstream may
    read a cross-encoder score - it decides *which* jobs are considered, never
    *why* one scored what it did.
    """
    if not hits:
        return hits
    query = f"{profile.career_goals} {' '.join(sorted(profile.skills))}"
    pairs = [(query, f"{jobs.iloc[p]['title']}. {jobs.iloc[p]['description'] or ''}"[:2000])
             for p, _ in hits]
    scores = _model().predict(pairs, show_progress_bar=False)
    ordered = sorted(zip(hits, scores), key=lambda x: -x[1])[:top_k]
    return [(position, 0.0) for (position, _), _ in ordered]
