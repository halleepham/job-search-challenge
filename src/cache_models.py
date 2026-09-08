"""
Pre-download the sentence-transformer models so first run is offline-safe.

~180MB total. Doing this ahead of time keeps class/work sessions from stalling on
a model download. Cached under ~/.cache/huggingface/hub.

    python -m src.cache_models
"""

from __future__ import annotations

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # AC-4.3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"      # AC-8.1 (stretch)


def main() -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Caching {EMBEDDING_MODEL} ...")
    m = SentenceTransformer(EMBEDDING_MODEL)
    dim = m.get_sentence_embedding_dimension()
    print(f"  ok - {dim} dimensions, max_seq_length={m.max_seq_length}")

    print(f"Caching {RERANKER_MODEL} ...")
    ce = CrossEncoder(RERANKER_MODEL)
    score = ce.predict([("data engineer python airflow", "Data Engineer building ETL in Airflow")])
    print(f"  ok - smoke-test score {float(score[0]):.3f}")

    print("\nBoth models cached. Next: python -m src.preflight")


if __name__ == "__main__":
    main()
