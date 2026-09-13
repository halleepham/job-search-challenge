"""
REQ-4: personal knowledge base (FROZEN v1.0).

Chunks the user's career documents, embeds them, and retrieves the chunks that
best support a given job. This is what lets the UI show *"you match Airflow -
here is the line from your resume that shows it"* without generating text: every
displayed quote is sliced from the source document by character span (AC-4.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: AC-4.3: one pinned model for the personal KB, the job corpus, and the query.
#: Mixing embedding spaces is a correctness error, not a tuning choice.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: AC-4.1: blocks shorter than this are merged into their neighbour rather than
#: becoming chunks of their own - a bare "Skills" heading is not a chunk.
MIN_CHUNK_CHARS = 40

_BLANK_LINE = re.compile(r"\n\s*\n")
#: Best-effort section label from the block's opening words. Labelling only -
#: chunk boundaries come from blank lines (AC-4.1 v1.1), never from headings.
_SECTION_HINTS = [
    ("education", r"educat|degree|university|college|\bb\.?s\.?\b|\bm\.?s\.?\b"),
    ("skills", r"^skills|technical skills|proficien"),
    ("experience", r"experien|intern|employment|work history"),
    ("projects", r"project|portfolio"),
    ("certifications", r"certificat|credential|licens"),
]


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    source: str
    char_span: tuple[int, int]


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Unit-normalized embeddings, so cosine similarity is a dot product."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)


def _label(text: str) -> str:
    head = text[:80].lower()
    for name, pattern in _SECTION_HINTS:
        if re.search(pattern, head, re.I):
            return name
    return "other"


def chunk_documents(
    resume_text: str, career_goals: str = "", extra_documents: str = ""
) -> list[Chunk]:
    """
    AC-4.1 (v1.1): split on blank lines, merging blocks under
    ``MIN_CHUNK_CHARS`` into the previous one; the career-goals statement is
    always its own chunk.

    Paragraph splitting rather than heading detection: resumes are already
    visually blocked, so the boundaries land in nearly the same place, while
    heading detection across arbitrary resume formats is brittle.
    """
    chunks: list[Chunk] = []

    for source, body in (("resume", resume_text), ("extra_documents", extra_documents)):
        if not body or not body.strip():
            continue
        cursor = 0
        pending: list[tuple[str, int, int]] = []
        for block in _BLANK_LINE.split(body):
            start = body.index(block, cursor)
            end = start + len(block)
            cursor = end
            stripped = block.strip()
            if not stripped:
                continue
            pending.append((block, start, end))
            joined = "".join(b for b, _, _ in pending).strip()
            if len(joined) >= MIN_CHUNK_CHARS:
                s, e = pending[0][1], pending[-1][2]
                text = body[s:e]
                chunks.append(Chunk(text, _label(text), source, (s, e)))
                pending = []
        if pending:  # trailing short block: attach rather than discard
            s, e = pending[0][1], pending[-1][2]
            text = body[s:e]
            if chunks and chunks[-1].source == source:
                prev = chunks.pop()
                span = (prev.char_span[0], e)
                merged = body[span[0]:span[1]]
                chunks.append(Chunk(merged, prev.section, source, span))
            else:
                chunks.append(Chunk(text, _label(text), source, (s, e)))

    if career_goals and career_goals.strip():
        chunks.append(
            Chunk(career_goals, "career_goals", "career_goals", (0, len(career_goals)))
        )
    return chunks


@dataclass
class PersonalKB:
    """Embedded career documents plus per-job evidence retrieval."""

    chunks: list[Chunk]
    embeddings: np.ndarray

    @classmethod
    def build(
        cls, resume_text: str, career_goals: str = "", extra_documents: str = ""
    ) -> PersonalKB:
        chunks = chunk_documents(resume_text, career_goals, extra_documents)
        return cls(chunks, embed([c.text for c in chunks]))

    @property
    def evidence_mask(self) -> np.ndarray:
        """
        AC-9.10: chunks eligible as *evidence* - everything except the
        career-goals statement, which has its own score component (AC-9.9).

        Included, it won the max on most results and the two semantic components
        returned identical values, making 30% of the weight one signal counted
        twice (AC-9.12).
        """
        return np.array([c.section != "career_goals" for c in self.chunks])

    def retrieve_by_vector(self, job_vector: np.ndarray, k: int = 3,
                           evidence_only: bool = False) -> list[tuple[Chunk, float]]:
        """
        Evidence for a job whose vector is already known (AC-9.10).

        Preferred over :meth:`retrieve` in the pipeline: the job's index vector
        already exists, so re-embedding its text is both wasted work and a second
        measurement that could in principle disagree with the one that produced
        the score. Passing the stored vector makes the displayed evidence and the
        scored evidence provably the same computation.
        """
        if not self.chunks:
            return []
        mask = self.evidence_mask if evidence_only else np.ones(len(self.chunks), bool)
        if not mask.any():
            return []
        idx = np.flatnonzero(mask)
        scores = self.embeddings[idx] @ job_vector
        order = np.argsort(-scores)[: min(k, len(idx))]
        return [(self.chunks[idx[i]], float(scores[i])) for i in order]

    def retrieve(self, job_text: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """
        AC-4.4: the ``min(k, n_chunks)`` most similar chunks, descending, each
        with its cosine similarity attached.

        Exact search over a NumPy array - the KB is tens of chunks, so no index
        is warranted on this side.
        """
        if not self.chunks:
            return []
        scores = self.embeddings @ embed([job_text])[0]
        order = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [(self.chunks[i], float(scores[i])) for i in order]
