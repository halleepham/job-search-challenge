# Job Matcher

**CS 5542 Big Data Analytics and Applications — Challenge 1**
Hallee Pham · Human-AI Co-Design

An explainable job-matching application. A job seeker describes themselves once; the system searches
**18,461 real LinkedIn job postings** and returns ranked matches where **every number can be traced
to a stated rule** and every quoted line comes verbatim from the user's own résumé.

---

## What it does

```
Profile  →  FILTER  →  RETRIEVE  →  SCORE  →  RANK  →  matches with evidence
            eliminate  BM25+dense  8 weighted  top 10
            ~90-99%    fused (RRF) components
```

**Filters eliminate; scores rank.** A job paying below your floor disappears entirely — it cannot
compensate with a strong skill match. A job thirty miles away simply scores slightly lower.

Every result shows the job beside your profile, a points breakdown that **sums exactly to the score**,
the résumé lines supporting each matched skill, and — unusually — the **unknowns**: facts the posting
never stated, where the score fell back to a neutral default.

---

## Quick start

**macOS / Linux:**

```bash
git clone <repo-url> && cd job-search-challenge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
git clone <repo-url>; cd job-search-challenge
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks activation with an execution-policy error, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first — this only affects the current
> terminal session. Using Command Prompt instead of PowerShell? Activate with
> `.venv\Scripts\activate.bat`.

### Credentials

Set this up **before** the steps below — the data-download step depends on it. Only one credential is
needed, and it is free. Create a Kaggle API token (kaggle.com → Settings → API → Create New Token),
then:

**macOS / Linux:**

```bash
mkdir -p ~/.kaggle && echo "<your-token>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
"<your-token>" | Out-File -FilePath "$env:USERPROFILE\.kaggle\access_token" -Encoding ascii -NoNewline
```

> `chmod` has no Windows equivalent and isn't needed there — NTFS permissions on your user profile
> folder already keep the file private to your account.

**No other keys.** No LLM API, no paid services, nothing to configure at runtime.

### Build and run

```bash
# 1. get the data — REQUIRED. The app checks for the corpus file at startup
#    and will not run without it.
python -m src.download_data          # LinkedIn postings  → data/raw/

# 2. build the corpus (~40 s) — REQUIRED for the same reason.
python -m src.ingest                 # → data/processed/jobs_tech.parquet

# 3. run
streamlit run app.py
```

Two more scripts exist but are **not required** to run the app:

- `python -m src.download_data_jobs` — only needed to regenerate the skill vocabulary from scratch.
  The vocabulary it produces (`data/vocabulary/skills_vocabulary.json`) is already committed, so this
  step is skipped by default.
- `python -m src.cache_models` — pre-downloads the sentence-embedding model (~180 MB) so the first
  `ingest` run doesn't pause to fetch it. Skipping this just means that download happens automatically
  the first time it's needed, during step 2 above, instead of ahead of time.

`python -m src.preflight` verifies packages, credentials, datasets and models in one command and
names anything missing.

> **Note on the Kaggle client.** `src/download_data.py` uses **`kagglehub`**, not the `kaggle` CLI. As
> of `kaggle==1.7.4.5` — the newest published version — the CLI accepts only legacy `username`+`key`
> credentials, while Kaggle's site now issues a single API token. Verified by grepping the installed
> package: zero references to `access_token`.

**No other keys.** No LLM API, no paid services, nothing to configure at runtime.

---

## Dataset

| | |
|---|---|
| **Source** | [LinkedIn Job Postings 2023-2024](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) — 123,849 postings, 31 columns, 11 CSVs (~517 MB uncompressed) |
| **Secondary** | [`lukebarousse/data_jobs`](https://huggingface.co/datasets/lukebarousse/data_jobs) — 785,741 rows; seeds the skill vocabulary and provides the scalability test |
| **Storage** | Parquet (columnar, compressed, dtype-preserving) |
| **Raw data** | Not committed. Regenerated from the commands above |

### ⚠️ These postings have expired

The corpus runs **December 2023 – April 2024**, expiring April–October 2024. **100% of it is closed
today.** The application links to each original posting and states the date it closed rather than
implying an application is still possible. This is a record of a real job market, not a live one.

### Ingestion funnel

```
123,849  raw postings
 -90,252  not IT / engineering / analytics / QA / science
  -1,815  duplicates (title + company + location, normalised)
    -123  Volunteer and Other employment types
 -13,198  no extractable skill
= 18,461  corpus
```

Field coverage, measured rather than assumed (`data/processed/coverage_stats.json`):

| Field | Coverage | Consequence |
|---|---:|---|
| Experience requirement | 89.5% | Regex first, seniority label as fallback |
| Education requirement | 55.2% | Graded, never gated — absent scores a neutral 0.5 |
| Salary | 29.9% | Unlisted passes only if the user opts in, and is flagged |
| `skills_desc` | **1.5%** | Forced skill extraction from description prose |

---

## Pipeline

### 1 · Ingestion — DuckDB
Joins postings to job-function codes, scopes to technical roles, deduplicates, and normalises salary,
work setting, experience and education. **DuckDB rather than Spark**: the corpus is ~124k rows, which
fits single-machine, so Spark's JVM startup and shuffle overhead would be cost without benefit. The
crossover is documented rather than asserted — see Results.

### 2 · Skill extraction
A **372-term vocabulary** (252 seeded from `data_jobs`, 121 hand-added for mobile, .NET, design,
security and testing) matched by **word-boundary regex — never substring**. "Java" must not match
"JavaScript"; "Go" must not match "MongoDB". Single-letter skills like `R` require another recognised
skill within 40 characters, because `"P/R organization"` and `"mission r equirements"` are not
evidence of anything.

### 3 · Hard filters
Location (geocoded offline via `geonamescache`, distance-decayed), salary, work setting, employment
type, and at least one shared non-generic skill — generic office tools cannot be the sole reason a
job is considered.

### 4 · Hybrid retrieval
**BM25** over the full posting text and **dense embeddings** (`all-MiniLM-L6-v2`) over title plus
description, fused by **Reciprocal Rank Fusion**. The two indexes deliberately cover different text:
BM25 carries keyword precision on named tools, the dense vector carries paraphrase. Raw scores are
never summed — they live on incompatible scales, so fusion uses ranks only.

### 5 · Scoring

| Component | Weight | |
|---|---:|---|
| Required skills | 30% | Damped when a posting states few requirements |
| Résumé evidence ~ description | 17% | Excludes the career-goals chunk |
| Experience | 15% | Asymmetric — under-qualification penalised harder than over |
| Career goals ~ description | 13% | |
| Preferred skills | 8% | Dropped and renormalised when absent (93% of postings) |
| Location | 8% | `exp(−miles/60)` decay |
| Job title | 5% | Seniority stripped first |
| Education | 4% | In-progress degrees count as a half-step |

Semantic similarities are **calibrated against a fixed background distribution** measured per profile,
not rescaled by a chosen constant.

---

## Analytics

The **Insights** page reports the corpus itself: top skills and their co-occurrence, titles,
geography, salary by role family, work setting and experience requirements.

Top requested skills are `excel` (3,455), `python` (3,140), `sql` (2,997), `agile` (2,697),
`word` (2,289), `aws` (2,092); the most common skill pairs are `excel+word` and `excel+powerpoint`.
That reflects what a corpus of IT, engineering, analytics, QA and science job functions actually
contains — roughly a quarter is strictly software/data work, and the interface labels it that way
rather than calling itself a "tech jobs" dataset.

Geographic distribution is led by CA (1,858), TX (1,771), VA (753), NY (728) and IL (675). Work
setting splits 11,443 on-site · 4,207 remote · 2,811 hybrid.

---

## Results

Full evaluation: **`notebooks/evaluation.ipynb`** (6 figures, 10 tables under `notebooks/results/`).

**Retrieval** — recall of the true top-20 by final score, at k=200:

| | BM25 | Dense | Hybrid |
|---|---:|---:|---:|
| Data science profile | 0.55 | 0.95 | **0.95** |
| Backend profile | 0.75 | 0.90 | **0.95** |
| Security profile | 0.95 | 1.00 | **1.00** |

Hybrid matches or beats both single retrievers on every profile, and reaches **100% recall at k=400**
while BM25 alone needs k=1400. That measurement sets the production `k`.

**Latency** — 556 ms median, 576 ms p95 (filter 62 · retrieve 365 · score 128 · evidence 0.6).

**Scalability** — ingestion throughput *rises* with corpus size (7.1k → 37.6k rows/sec) as fixed
startup cost amortises. On the 785,741-row `data_jobs` corpus a `GROUP BY` with a median aggregate
returns in **0.02 s**, about 40M rows/sec. This is the evidence behind choosing DuckDB.

**Component sensitivity** — zeroing each component and counting how many top-5 results change, across
all three profiles: required skills 3.33 · experience 1.33 · education 0.67 · career goals 0.67 ·
title 0.67 · résumé evidence 0.67 · location 0.33 · preferred skills 0.00.

---

## Repository

### Branches — the three design stages

| Branch | Stage |
|---|---|
| `human-baseline` | Stage 1 — the original paper design, before any AI involvement |
| `agent-exercise` | Stage 2 — code an AI agent produced from the problem statement alone, never merged |
| `main` / `human-ai-codesign` | Stage 3 — this system |

The Stage 2 branch is kept unmerged **on purpose**: it is the comparison baseline. Any claim in the
report about what the AI got wrong can be checked directly —

```bash
git show agent-exercise:src/matcher.py       # the AI's matching engine
git diff main agent-exercise -- src/         # what changed between stages
```

### Layout

```
app.py                  router and top navigation
views/                  profile · results · my jobs · insights
src/                    ingest · skills · filters · retrieval · scoring · explain · evaluation
tests/                  537 tests, one file per requirement area
notebooks/              evaluation.ipynb, figures/, results/
docs/project-plan.md    the specification: numbered requirements and acceptance criteria
data/                   raw (gitignored) · processed · vocabulary · index (gitignored)
```

The project is **specification-driven**: every behaviour is a numbered requirement (`REQ-n`) with
testable acceptance criteria (`AC-n.m`), tests cite the AC they verify, and commits reference the
requirement they implement. When implementation contradicted a frozen requirement, the specification
was amended first with the reason recorded — `docs/project-plan.md` ends with a changelog of every
such change.

```bash
pytest tests/ -q          # 537 tests
python -m src.preflight   # environment check
```

---

## Limitations

- **Historical data.** Every posting expired in 2024. The interface discloses this per result.
- **Domain-scoped.** IT, engineering, analytics, QA and science job functions; ~25% is strictly
  software/data. Skill extraction is vocabulary-bound and would not transfer to other fields.
- **Batch, not streaming.** No live job-feed ingestion.
- **Single machine.** DuckDB and pandas, justified by measurement at this scale.
- **Education has no field of study.** A Bachelor's in Mathematics scores identically against
  "Bachelor's in Radiologic Technology" — the pipeline records a level, not a subject.
- **Preferred skills (8%) rarely fires** — only 6.6% of postings list any.
- **Decisions are session-scoped.** Saved jobs and profiles do not persist across restarts.
