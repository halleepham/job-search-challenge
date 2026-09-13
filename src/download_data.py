"""
Download the raw LinkedIn job postings dataset into data/raw/ (AC-14.3).

Uses kagglehub rather than the `kaggle` CLI: as of kaggle==1.7.4.5 the CLI only
accepts legacy username+key credentials (kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY),
while Kaggle's website now issues a single API token. kagglehub reads that token
from ~/.kaggle/access_token or $KAGGLE_API_TOKEN.

    python -m src.download_data
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "arshkon/linkedin-job-postings"
RAW = Path("data/raw")


def main() -> None:
    print(f"Downloading {DATASET} (several hundred MB, this takes a while)...")
    cached = Path(kagglehub.dataset_download(DATASET))
    print(f"Cached at: {cached}")

    RAW.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in cached.rglob("*.csv"):
        dst = RAW / src.relative_to(cached)          # preserve jobs/ companies/ mappings/
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        copied += 1
        print(f"  {dst.relative_to(RAW)}  ({src.stat().st_size / 1e6:.1f} MB)")

    print(f"\n{copied} CSV files in {RAW}/")
    print("Next: python -m src.audit_schema")


if __name__ == "__main__":
    main()
