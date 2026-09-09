"""
REQ-5: job corpus indexing and calibration (FROZEN v1.0).

Two indexes over deliberately different text (AC-5.1, AC-5.2):

  dense  = title + description        - meaning, truncated to the model's 256 tokens
  BM25   = title + skills + description - keyword precision, untruncated

That split is the point of hybrid retrieval: BM25 carries the named tools the
dense window may not reach, and the dense vector carries paraphrase BM25 cannot
see. It is also why the 256-token limit is tolerable - text beyond it is still
fully searchable by the other retriever.

Everything is built once and cached; re-embedding on every app start would make
the application unusable.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.personal_kb import EMBEDDING_MODEL, embed
from src.seq import as_list

INDEX_DIR = Path("data/index")

#: AC-5.5 (v1.1): how many top candidates per reference profile define the
#: calibration background. Matches the pipeline's retrieval depth, so the
#: background is the population the score is actually applied to.
CALIBRATION_TOP_K = 200


def build_dense_text(row) -> str:
    """
    AC-5.1: ``title + description``. Skills are deliberately excluded - they have
    their own score components, and embedding them would let one signal feed two
    components (AC-9.12). Title first so the highest-signal text is always inside
    the 256-token window.
    """
    return f"{row['title']}. {row['description'] or ''}".strip()


def build_bm25_text(row) -> str:
    """AC-5.2: ``title + skills + description``, untruncated."""
    skills = as_list(row.get("required_skills")) + as_list(row.get("preferred_skills"))
    return f"{row['title']} {' '.join(skills)} {row['description'] or ''}".lower()


@dataclass(frozen=True)
class CalibrationConstants:
    """
    AC-9.11 / D8: maps raw cosine onto [0, 1] against a **fixed background**
    distribution measured at index-build time.

    Deliberately not pool-relative min-max: that would make every job's score
    depend on which other jobs happened to be retrieved, so adding one unrelated
    job could reorder the top 5 and no acceptance criterion could assert a score.
    """

    p5: float
    p95: float
    n_sampled: int
    n_profiles: int

    def apply(self, raw_cosine: float) -> float:
        spread = self.p95 - self.p5
        if spread <= 0:
            return 0.5
        return float(np.clip((raw_cosine - self.p5) / spread, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {"p5": self.p5, "p95": self.p95,
                "n_sampled": self.n_sampled, "n_profiles": self.n_profiles}


def calibrate(job_embeddings: np.ndarray, profile_embeddings: np.ndarray,
              top_k: int = CALIBRATION_TOP_K) -> CalibrationConstants:
    """
    AC-5.5 (v1.1) / AC-5.7: percentiles of the similarity distribution over the
    candidates that actually reach scoring.

    For each reference profile, take its ``top_k`` most similar jobs and pool
    them. Calibrating on *random* corpus jobs instead put the p95 anchor below
    the p5 of the retrieved set, so ~95% of everything scored clipped to 1.0 and
    the semantic components stopped discriminating (AC-9.11a).

    The constants are still computed **once at build time** from a fixed
    reference set, so a job's calibrated score never depends on which other jobs
    a live search retrieved - D8's pool-independence is untouched.

    Reads already-computed vectors: a few matrix multiplies, well under a second.
    """
    sims = profile_embeddings @ job_embeddings.T          # (profiles, jobs)
    k = min(top_k, sims.shape[1])
    top = np.sort(sims, axis=1)[:, -k:].ravel()
    return CalibrationConstants(
        p5=float(np.percentile(top, 5)), p95=float(np.percentile(top, 95)),
        n_sampled=int(top.size), n_profiles=len(profile_embeddings),
    )


def _corpus_hash(jobs: pd.DataFrame) -> str:
    ids = ",".join(map(str, jobs["job_id"].tolist()))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


@dataclass
class JobIndex:
    jobs: pd.DataFrame
    embeddings: np.ndarray
    bm25: object
    calibration: CalibrationConstants
    index_dir: Path
    from_cache: bool = False

    # -------------------------------------------------------------- build

    @classmethod
    def build(cls, jobs: pd.DataFrame, index_dir: Path | str = INDEX_DIR) -> JobIndex:
        from rank_bm25 import BM25Okapi

        from src.profiles import PRESETS

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        embeddings = embed([build_dense_text(r) for _, r in jobs.iterrows()])
        bm25 = BM25Okapi([build_bm25_text(r).split() for _, r in jobs.iterrows()])

        profile_vecs = embed([p.career_goals + " " + p.resume_text for p in PRESETS.values()])
        calibration = calibrate(embeddings, profile_vecs)

        np.save(index_dir / "embeddings.npy", embeddings)
        (index_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
        (index_dir / "calibration.json").write_text(json.dumps(calibration.to_dict(), indent=2))
        (index_dir / "manifest.json").write_text(json.dumps({
            "n_rows": len(jobs),
            "corpus_hash": _corpus_hash(jobs),
            "embedding_model": EMBEDDING_MODEL,
        }, indent=2))
        return cls(jobs, embeddings, bm25, calibration, index_dir, from_cache=False)

    @classmethod
    def load_or_build(cls, jobs: pd.DataFrame, index_dir: Path | str = INDEX_DIR) -> JobIndex:
        """
        AC-5.3: load from disk when the manifest matches the corpus and model;
        rebuild only when it does not. AC-5.8: calibration shares this check, so
        changing the corpus or the model invalidates it too.
        """
        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            if (m.get("n_rows") == len(jobs)
                    and m.get("corpus_hash") == _corpus_hash(jobs)
                    and m.get("embedding_model") == EMBEDDING_MODEL):
                from rank_bm25 import BM25Okapi  # noqa: F401  (unpickling needs the class)

                return cls(
                    jobs,
                    np.load(index_dir / "embeddings.npy"),
                    pickle.loads((index_dir / "bm25.pkl").read_bytes()),
                    CalibrationConstants(**json.loads((index_dir / "calibration.json").read_text())),
                    index_dir,
                    from_cache=True,
                )
        return cls.build(jobs, index_dir)

    # ------------------------------------------------------------- search

    def calibration_for_scores(self, corpus_scores: np.ndarray,
                               top_k: int = CALIBRATION_TOP_K) -> CalibrationConstants:
        """
        AC-5.5 (v1.1): anchors for one semantic component, from the top-K of that
        component's *own* similarity distribution over the whole corpus.

        Calibrated **per component**, not once per profile: the career-goals
        query and the resume-chunk query are different texts with different
        similarity scales, so anchors measured on one and applied to the other
        pushed a whole top-5 to 0.0. Calibrated **per profile**, not pooled
        across reference profiles, because resume length and vocabulary shift the
        scale the same way.

        D8's pool-independence is intact either way: the anchors depend only on
        the profile and the corpus, never on which candidates survived filtering,
        so adding a job to a live search cannot change a score. Cost is one
        matrix multiply against precomputed vectors.
        """
        if corpus_scores.size == 0:
            return self.calibration
        k = min(top_k, corpus_scores.size)
        top = np.sort(corpus_scores)[-k:]
        return CalibrationConstants(
            p5=float(np.percentile(top, 5)), p95=float(np.percentile(top, 95)),
            n_sampled=int(k), n_profiles=1,
        )

    def search_dense(self, query: str, k: int = 200) -> list[tuple[int, float]]:
        """Cosine similarity over unit-normalized vectors, as positional indexes."""
        scores = self.embeddings @ embed([query])[0]
        order = np.argsort(-scores)[: min(k, len(scores))]
        return [(int(i), float(scores[i])) for i in order]

    def search_bm25(self, query: str, k: int = 200) -> list[tuple[int, float]]:
        scores = np.asarray(self.bm25.get_scores(query.lower().split()))
        order = np.argsort(-scores)[: min(k, len(scores))]
        return [(int(i), float(scores[i])) for i in order]
