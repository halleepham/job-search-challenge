"""
Download the `data_jobs` corpus into the local Hugging Face cache.

Two uses (AC-2.1, AC-13.4):
  - seeds the REQ-2 skill gazetteer from its pre-parsed `job_skills` column
  - the ~786k-row scale test that carries the Volume claim

Public dataset, no token required. Cached under ~/.cache/huggingface/.

    python -m src.download_data_jobs
"""

from __future__ import annotations

from datasets import load_dataset

DATASET = "lukebarousse/data_jobs"


def main() -> None:
    print(f"Downloading {DATASET} (~786k rows)...")
    ds = load_dataset(DATASET, split="train")
    print(f"\nrows    : {len(ds):,}")
    print(f"columns : {len(ds.column_names)}")
    print(f"          {', '.join(ds.column_names)}")

    if "job_skills" in ds.column_names:
        sample = [s for s in ds["job_skills"][:200] if s]
        print(f"\njob_skills present - AC-2.1 gazetteer seed available")
        print(f"  example: {sample[0] if sample else '(none in first 200)'}")
    else:
        print("\nWARNING: no `job_skills` column - AC-2.1 needs amending")

    print("\nCached. Next: python -m src.preflight")


if __name__ == "__main__":
    main()
