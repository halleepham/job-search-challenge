# Project Spec: Job Search & Matching Application

**Course:** CS 5542 Big Data Analytics and Applications — Challenge 1, Stage 3 (Human-AI Co-Design)
**Author:** Hallee Pham
**Branch:** `human-ai-codesign`

**Status:** FROZEN v1.0 — every active requirement is frozen. REQ-1 is at v1.3 and REQ-5 at v1.1; REQ-8 and REQ-10 are REMOVED.
<!-- Individual REQ sections are frozen one at a time. Update this line to FROZEN v1.0 only when every REQ below reads FROZEN. -->

**Last updated:** 2026-09-08

---

## 0. Purpose and scope of this document

This is the authoritative specification for the Stage 3 system. It supersedes
`docs/context/job-matching-application-plan.md`, which was an unvetted draft used as input.
Where this spec differs from that draft, the difference is deliberate and recorded in §0.3.

Code and tests reference the `REQ-n` / `AC-n.m` identifiers below directly. Commits reference
the REQ they implement. No code is written against a section still marked DRAFT.

### 0.1 What the system does

A job seeker enters a profile — career documents plus structured preferences — and receives the
top 5 matching job postings from a corpus of real tech job listings, each with a 0-100 match score,
a decomposed breakdown of how that score was produced, the matched and missing skills, and a
grounded natural-language explanation of why the job sits at its rank.

### 0.2 Traceability to the assignment

Every REQ carries a **Traces to** line pointing at a section of
`docs/context/challenge-1-report-instructions.md`. A requirement that traces to nothing does not
belong in this spec.

### 0.3 Design decisions and rationale

These were settled in review before drafting. They are recorded here because several of them
reverse the draft plan or the Stage 1 human design, and the report (Section 4) must cite them.

| # | Decision | Rationale | Rejected alternative |
|---|---|---|---|
| D1 | **LinkedIn Job Postings is the primary dataset**; `data_jobs` is the scale-test corpus and the seed for the skill vocabulary | LinkedIn requires real cleaning and skill extraction, which is directly graded (report §5). `data_jobs` ships pre-parsed skills, which would hand that work away | Using `data_jobs` as primary |
| D2 | **Hard filters run BEFORE retrieval** | The draft ran filters after retrieving top-200. For a narrow profile (one metro or remote, salary floor, full-time) roughly 1-3% of a national corpus survives, leaving ~2-6 candidates — too few to rerank or rank. Filters are vectorized boolean masks over precomputed columns: milliseconds over 30k rows, not an expensive stage | Retrieve-then-filter (draft §3) |
| D3 | **Corpus is scoped to LinkedIn's IT, engineering, analytics, QA, and science job functions** — described that way, not as "tech roles" | Scoping keeps the corpus relevant without pretending to general-purpose coverage. **Measured 2026-09-08: only ~25% of the 31,782-row corpus is strictly software/data**, because `ENG` covers every engineering discipline and `ANLS` every kind of analyst. We keep them rather than filter further: a mechanical-engineering posting scores near zero on required-skill overlap (30%), title match (5%), and both semantic components (30%), so it cannot reach a top-5. The match score is the mechanism that makes them irrelevant, not the corpus definition. The obligation this creates is descriptive — the README, REQ-12 charts, and the report must say what the corpus actually contains | Filtering harder on job-function codes; a core-vs-generic skill distinction to force non-software rows out (rejected — the ranking already handles it, and it would add a vocabulary-tagging step for no user-visible gain) |
| D4 | **Skills are extracted from `skills_desc` and `description`, not from `job_skills.csv`** | `job_skills.csv` contains coarse job-function categories (`IT`, `ENG`, `ANLS`), not tool-level skills. It is unusable for matching — but it is exactly right for domain scoping (D3) | Trusting the dataset's skills column |
| D5 | **Work setting and employment type are multi-select set-membership filters** | Equality forces a false choice on a user who would accept remote *or* hybrid. Set membership is the correct primitive | Exact equality (Stage 1 human design) |
| D6 | **Education is graded, not gated, with an `in_progress` level** | A gate eliminates candidates who are mid-degree — the decisive reason, and the one that motivated the decision. `in_progress` models "finishing a master's" as 3.5 on the ladder. **Correction (2026-09-08):** this was also justified as "`education_required` is absent from most rows". Measurement refuted that — AC-1.7 parses a requirement from **58.5%** of the corpus. The mid-degree argument stands on its own; the coverage argument is withdrawn | Hard eligibility gate (Stage 1 human design) |
| D7 | **Experience is graded with an asymmetric curve** | Under-qualification is a screening barrier; over-qualification is a preference mismatch. The penalty is therefore gentler on the surplus side and floored well above zero | Symmetric penalty; no over-qualification handling (draft §8.1) |
| D8 | **Semantic scores are calibrated against a fixed background distribution** | Pool-relative min-max makes every score depend on every other candidate, so adding one job reorders the top 5 and no acceptance criterion can assert a score value | Pool-relative min-max (draft §8.2); a fixed rescaling multiplier (Stage 2 AI code) |
| D9 | **DuckDB (SQL) for offline ingestion and analytics; pandas for the query-time path** | The corpus is ~124k rows / ~800MB — it fits single-machine, so Spark's JVM startup and shuffle overhead exceed its benefit. DuckDB is columnar, vectorized, out-of-core, and pip-installable with no JVM. REQ-1's join/filter/dedupe and REQ-12's aggregations are naturally SQL. The crossover point where distributed execution would win is documented instead of assumed | PySpark (real setup cost, no gain at this scale); pandas alone (786k-row scale test gets memory-tight, manual chunking) |
| D10 | **No cross-encoder. Removed entirely (REQ-8 → Non-Goals)** | Accepted from the AI at Stage 2, removed at Stage 3 once filter-first (D2) made it redundant. The assignment never asks for reranking — report §6 and §9 name BM25, embeddings and hybrid retrieval only. In this architecture it would trim 200 candidates to 50 before scoring, but scoring 200 costs milliseconds, and jobs that survive retrieval without merit simply score low and never reach the top 5. It saves no meaningful compute and duplicates work the scorer already does. It also carries an opacity cost either way: as a score component it would put an unjustifiable term inside the breakdown, and as a pruner it can silently drop a good job with no signal to the user — pruning relocates that opacity rather than removing it | Cross-encoder as a weighted score component (unexplainable term in the score); cross-encoder as a pruner (retained the opacity, earned too little to justify it) |
| D11 | **No LLM in the application. Evidence is retrieved, not generated** | Every number and every quote the user sees is produced by deterministic code: scores from REQ-9, supporting résumé text from REQ-4's semantic retrieval with verifiable character spans. The co-design prep's "Practice with Agentic AI" section was a prompt-engineering exercise — practice at directing an AI — not a component of the product, and building it in would have added an API key, a cost, and a reproducibility barrier (a grader cannot run it) for something the assignment never required. Report §6 asks about an LLM's role only *if* one is used | A hosted LLM explanation layer (removed, see REQ-10); LLM-computed scores (drift between runs, untestable) |
| D13 | **Retrieval always runs; `k` is chosen by measured recall** (v2). `k = max(400, 25% of survivors)` | **Revised 2026-09-11 after measuring `k` properly.** v1 skipped retrieval below 2,500 survivors because top-200 recovered only **45-95%** of the true top-20 by score, so the approximate stage was discarding jobs the exact stage wanted. Scoring every survivor instead costs ~220 ms more (877 ms vs 656 ms for the largest profile) and produced an **identical top-5** for all three reference profiles. Exactness is available for a fifth of a second, so the approximation is not worth its recall loss at this corpus size. But extending the sweep showed that was a property of **k=200, not of retrieval**: hybrid reaches **100% recall at k=400**, before either single retriever (BM25 alone needs k=1400). The fix is to size `k` by measurement, not to bypass the stage. Retrieval now always runs at `k = max(400, 25% of survivors)` — zero measured recall loss, and the pipeline the report describes is the one that runs. Skipping it would also have left the §9 comparison evaluating a stage the application bypassed | Skipping retrieval below a threshold (v1 — solved a `k`-sizing problem by deleting the stage); a fixed k=200 |
| D12 | **Industry is dropped from the score** | Scope, not data quality. Stage 1 gave it only 0.10 of the manual score, and the deadline does not allow a component that small to earn its implementation and test cost. **Correction (2026-09-08):** this decision was originally justified as "industry labels are inconsistent across sources" — the schema audit refuted that. `job_industries.csv` is 0% null on both columns and `mappings/industries.csv` carries 422 clean industry names. The data is good; the reason for dropping it is priority. Recorded as an explicit rejection, not an omission, and a strong candidate for report §12 future work | Keeping the Stage 1 industry component |

### 0.4 Architecture

```
  RAW                          OFFLINE (DuckDB, run once)                     PERSISTED
  ─────                        ───────────────────────────                    ─────────
  postings.csv (124k) ─┐
  job_skills.csv       ├─ join ─→ scope to tech (D3,D4) ─→ dedupe ─→ normalize
  companies.csv       ─┘            ~30k rows              salary / work setting /
                                                           experience / education
                                                                  │
                                    skill extraction ←────────────┤
                                    (gazetteer, REQ-2)            │
                                                                  ▼
  data_jobs (786k) ────→ skill vocabulary seed          jobs_tech.parquet
         │                                              coverage_stats.json
         └──────────────→ scale test (REQ-13)           analytics/*.csv

                                 INDEX BUILD (run once, cached)
                                 ──────────────────────────────
                       jobs_tech.parquet ─→ dense embeddings (.npy + FAISS)
                                          ─→ BM25 index
                                          ─→ calibration.json (p5/p95, D8)

  ONLINE (pandas + Streamlit, per query)
  ──────────────────────────────────────
  Profile page (REQ-3) ─→ personal KB chunk+embed (REQ-4)
         │
         ▼
  (1) HARD FILTERS   location · salary · work setting · employment type · skill floor   [D2]
         │                                                              ~30k → ~N
         ▼
  (2) RETRIEVAL      survivors ≤ 2,500 → skipped, all are scored exactly   [D13]
                     otherwise → BM25 + dense over survivors, RRF → top 200
         │
         ▼
  (3) SCORE          8-component weighted score, calibrated semantics  [D8]
         │
         ▼
  (4) RANK → top 5 ─→ (5) EVIDENCE  retrieve supporting résumé chunks per job  [D11]
                              │
                              ▼
                     Results page (REQ-11) · Analytics page (REQ-12)
```

### 0.5 Repository layout and storage

```
data/
├── raw/                     # gitignored — download instructions in README
├── processed/
│   ├── jobs_tech.parquet    # gitignored — serving corpus, regenerated by REQ-1
│   ├── coverage_stats.json  # committed — field coverage measurements
│   └── analytics/*.csv      # committed — small aggregation outputs for the report
├── vocabulary/
│   └── skills_vocabulary.json   # committed — curated gazetteer (REQ-2)
└── index/                   # gitignored — FAISS index, BM25 pickle, calibration.json
```

Parquet is the processed format: columnar, compressed, preserves dtypes, and is the natural
DuckDB output. Coverage stats and analytics outputs are committed because the report cites their
numbers and a grader must be able to read them without running the pipeline.

---

## REQ-1: Job data ingestion and corpus construction

**Status:** FROZEN v1.3 (2026-09-08) — **IMPLEMENTED**, all ACs pass (119 tests)
**Traces to:** report §5 (Big Data Collection, Storage, and Processing), §1 (big data goal)
**Tests:** `tests/test_req1_ingestion.py`

**Description:**
A DuckDB batch job that reads the raw LinkedIn dataset, scopes it to tech roles, deduplicates,
normalizes every field the downstream system depends on, and writes a Parquet serving corpus plus
a field-coverage report. Runs offline, once, and is fully reproducible from raw inputs.

**Inputs:**
`data/raw/postings.csv`, `data/raw/job_skills.csv`, `data/raw/job_industries.csv` (LinkedIn Job
Postings, `kaggle.com/datasets/arshkon/linkedin-job-postings`, ~124k rows).

**Outputs / Behavior:**
`data/processed/jobs_tech.parquet` conforming to the normalized schema below, and
`data/processed/coverage_stats.json` recording per-field non-null rates and row counts at every
pipeline stage.

Normalized job schema:

```python
{
  "job_id": str,
  "title": str,
  "title_normalized": str,            # lowercased, seniority modifiers stripped
  "company": str,
  "location_raw": str,
  "location_city": str | None,
  "location_state": str | None,
  "lat": float | None,
  "lon": float | None,
  "is_remote": bool,
  "work_setting": str,                # Remote | Hybrid | On-site | Unknown
  "work_setting_inferred": bool,
  "employment_type": str,             # Full-time | Part-time | Contract | Internship | Unknown
  "min_years_exp": float | None,
  "min_years_exp_source": str,        # description_regex | seniority_label | none
  "education_required": int | None,   # ordinal 1-5, see REQ-9
  "salary_min": float | None,         # annual USD
  "salary_max": float | None,
  "salary_listed": bool,
  "required_skills": list[str],       # REQ-2
  "preferred_skills": list[str],      # REQ-2
  "description": str,
  "posted_date": date | None,
  "expiry_date": date | None,
  "posting_url": str,
}
```

**Acceptance Criteria:**

- **AC-1.1** — The tech scoping stage joins `postings.csv` to `job_skills.csv` on `job_id` and
  retains a posting if *either* its `skill_abr` set intersects
  `TECH_CODES = {"IT", "ENG", "ANLS", "QA", "SCI"}` *or* its `title_normalized` matches the tech
  title whitelist regex. `TECH_CODES` and the regex are declared as named constants in one module,
  not inlined. `job_skills.csv` holds 213,768 rows for 126,807 distinct `job_id`s (1.69 codes per
  job) and contains ids absent from `postings.csv`, so the join **must be a semi-join or be
  deduplicated to one row per posting** — a naive inner join fans rows out. The code set alone
  yields 33,502 postings (27% of the corpus); measured 2026-09-08.
- **AC-1.2** — Given a fixture of postings spanning tech and non-tech job functions, scoping retains
  exactly the tech ones. **The two branches of AC-1.1's OR have deliberately different postures, and
  both are asserted, not incidental:**
  - The **`TECH_CODES` branch is recall-oriented.** `ENG` covers every engineering discipline, so a
    Mechanical Engineer survives scoping. This is accepted because AC-2.6 (drop postings with zero
    gazetteer skills) is the real precision filter.
  - The **title branch is precision-oriented**: compounds only (`data engineer`, `database
    administrator`, `systems analyst`), never bare `engineer`, `analyst`, `administrator`,
    `technical`, `data`, or `it`. Its purpose is to recover genuinely technical roles miscoded
    outside `TECH_CODES` — e.g. a "Software Engineering Manager" coded `MGMT` — not to widen the net.
    A "Sales Engineer" coded `SALE` is therefore **dropped**.

  Amended v1.2 (2026-09-08). The original wording made the title branch recall-oriented too and named
  Sales Engineer as a retention example. Measurement refuted it: that branch contributed 6,157 rows of
  which only 18% were software/data, pulling in financial analysts (131), board-certified behavior
  analysts (67), office administrators (76), and data entry clerks (31). It was adding noise, not
  recovering miscoded tech roles. Measured: only ~55% of the `TECH_CODES` set has a software/data title
  — `ENG` covers all engineering disciplines, so service technicians and construction project
  managers survive scoping. This is acceptable because **AC-2.6 is the real precision filter**:
  those postings match no gazetteer skill, extract zero skills, and are dropped. A consequence
  worth stating plainly — the final corpus size is determined by gazetteer coverage, not by
  `TECH_CODES`, and is not knowable until REQ-2 runs. Current estimate 18-25k.
- **AC-1.3** — Deduplication is applied on `(title_key, company_key, location_key)`, where each key
  is the raw field lowercased, punctuation-stripped, and whitespace-collapsed **before** comparison.
  The dropped row count is recorded in `coverage_stats.json`.
  **`title_key` deliberately preserves seniority modifiers**, unlike `title_normalized`, which strips
  them for AC-1.1's whitelist and AC-9.7's role matching. Amended v1.1 (2026-09-08): the original
  wording keyed dedupe on `title_normalized`, which merged "Senior Data Engineer" and "Data Engineer"
  at the same company and location into one row. Those are two distinct openings, and a user
  filtering by experience would silently lose one of them.
- **AC-1.4** — Salary uses the dataset's existing `normalized_salary` column as the authoritative
  annual-USD value when it is non-null. When it is null, fall back to deriving from
  `min_salary`/`max_salary` + `pay_period`: `HOURLY` x 2080, `MONTHLY` x 12, `YEARLY` passes through.
  Any other or missing `pay_period` yields `salary_min = salary_max = None` and
  `salary_listed = False`. `coverage_stats.json` reports how many rows came from each of the three
  paths, so the report can state how much salary normalization the dataset had already done versus
  how much this pipeline added. There is exactly one rule per input case and no case falls through
  unhandled.
- **AC-1.5** — `work_setting` is derived: `remote_allowed == 1` → `Remote`; else a word-boundary
  match for "hybrid" in title or description → `Hybrid`; else → `On-site`. Every row where the
  value was derived rather than read directly has `work_setting_inferred = True`.
  **`remote_allowed` is a sparse flag, not a nullable boolean** — measured 2026-09-08, it takes
  exactly two values: `1.0` (15,246 rows) and null (108,603). Null therefore means "not flagged
  remote", not "unknown", and must not route to `Unknown`. Doing so would place 87.7% of the
  corpus in `Unknown`, which under AC-6.6's pass-and-tag policy makes the work-setting filter a
  no-op. `Unknown` remains in the schema for the `data_jobs` corpus, which does not carry this
  flag; for the LinkedIn corpus it is expected to be empty, and that is asserted.
- **AC-1.6a** — `employment_type` comes from `formatted_work_type` (0% null; **not** `work_type`,
  which is the same field unformatted). Measured distribution: Full-time 98,814 · Contract 12,117 ·
  Part-time 9,696 · Temporary 1,190 · Internship 983 · Volunteer 562 · Other 487. The first five
  are retained as the enum; `Volunteer` and `Other` are **excluded from the corpus** at ingestion
  and their dropped count recorded, since neither is a job a user of this system is searching for.
- **AC-1.6** — `min_years_exp` is set from a `(\d+)\s*(?:[-–—]\s*\d+)?\s*\+?\s*years?` regex over
  the description when one matches, taking **the lower bound of any range**
  (`min_years_exp_source = "description_regex"`); otherwise from
  `formatted_experience_level` via the map {Internship: 0, Entry level: 0, Associate: 2,
  Mid-Senior level: 5, Director: 8, Executive: 10} (`source = "seniority_label"`); otherwise
  `None` (`source = "none"`). Regex takes precedence over the label when both are available.
  Amended v1.3 (2026-09-08): the original regex `(\d+)\+?\s*years?` took the **upper** bound of a
  range — "4-7 years related business experience" parsed as 7, because "4-" is not followed by
  "years". Since the field is a *minimum*, that overstated every ranged requirement, and under
  AC-9.5 an inflated `job_min_years` widens `gap` and pushes down candidates who in fact qualify.
  Found by sampling parsed output against source text, not by a failing test.
- **AC-1.7** — `education_required` is parsed from the description to an ordinal 1-5 by degree
  keyword ("high school", "associate", "bachelor|BS|BA", "master|MS|MA|MBA", "PhD|doctorate").
  When several appear, the **lowest** is taken, since a posting saying "Bachelor's required,
  Master's preferred" requires a bachelor's. No match yields `None`.
- **AC-1.8** — `coverage_stats.json` reports the non-null rate for `salary_min`,
  `education_required`, `min_years_exp`, `work_setting` (excluding `Unknown`), and
  `skills_desc`, plus row counts after each of: raw load, tech scoping, dedupe, zero-skill drop.
  These numbers are cited directly in the report and must exist before REQ-9 weights are frozen.
- **AC-1.9** — The job is idempotent: running it twice over the same raw inputs produces byte-identical
  Parquet row counts and an identical `coverage_stats.json`.
- **AC-1.11** — `posting_url` is carried through from `job_posting_url` (100% present) and
  `expiry_date` from `expiry`, so a result can link to its source posting and disclose whether it has
  closed. Added 2026-09-11: both were dropped from the original schema, which left the "Applied"
  action with nowhere to go and the UI unable to say that **100% of this corpus expired between
  April and October 2024** — the postings ran Dec 2023 to Apr 2024. A job-search tool that lets a
  user click "Applied" on a posting that closed two years ago is lying to them.
- **AC-1.10** — Every transformation in the job is a pure function over a row or DataFrame with no
  reliance on engine-specific types in its signature. SQL handles the set-oriented work (join,
  filter, dedupe, aggregate); Python pure functions handle per-row parsing (salary, work setting,
  experience, education, skills), and those are unit-tested directly with no database connection.
  This boundary is what keeps the engine choice reversible.

**Notes / Open Questions:**
DuckDB runs in-process with no server, no JVM, and no configuration. It reads the raw CSVs
directly, so no separate load step is needed. Because AC-1.10 confines per-row logic to pure Python
functions, swapping the set-oriented layer for pandas or PySpark later is a swap, not a rewrite —
which is what makes the optional engine comparison in AC-13.4 cheap.

---

## REQ-2: Skill vocabulary and extraction

**Status:** FROZEN v1.1 (2026-09-11) — IMPLEMENTED
**Traces to:** report §5 (skill extraction, feature construction), §6 (skill similarity)
**Tests:** `tests/test_req2_skills.py`

**Description:**
A curated skill gazetteer plus a section-aware extractor that produces required and preferred
skill lists for each posting. This is the highest-leverage preprocessing step in the system:
it feeds the two largest score components and one hard filter.

**Inputs:**
`skills_desc` and `description` columns from REQ-1; `data/vocabulary/skills_vocabulary.json`.

**Outputs / Behavior:**
`required_skills: list[str]` and `preferred_skills: list[str]` per posting, both normalized to
canonical gazetteer terms.

**Acceptance Criteria:**

- **AC-2.1** — The gazetteer is a committed JSON artifact mapping canonical skill names to alias
  lists. It is seeded from the distinct values of the `data_jobs` `job_skills` column (a parsed
  list column) and hand-extended with broad-tech terms not present in a data-roles vocabulary
  (Java, Kubernetes, Terraform, React, Go, CI/CD, and similar). Target size ≥ 350 canonical terms.
- **AC-2.2** — Skill matching is **word-boundary regex over normalized text, never substring**.
  Test explicitly: "Java" must not match a description containing only "JavaScript"; "Go" must not
  match "MongoDB"; "R" must match "R and Python" but not "R&D" or "HR". This is the single most
  important test in the suite — it is the exact defect in the Stage 2 AI code
  (`agent-exercise:src/matcher.py`, `match_skill_presence`).
  **Boundary set extended and single-letter terms gated (v1.1, 2026-09-11).** Measured: 645 postings
  had `r` extracted, 83 as their *only* skill. The matches were `"P/R organization"` (a slash is not
  a word character), `"mission r equirements"` (a word broken mid-token in the source), and
  `"Project Man******r"` (masked text). `/` and `*` are now boundary characters, **and a
  single-letter skill counts only when another recognised skill appears within 40 characters** — so
  `"experience with R and SAS"` counts and `"P/R organization"` does not.
- **AC-2.3** — **`description` section-parsing is the primary extraction path** (AC-2.4);
  `skills_desc` is a supplement applied only where present, and its gazetteer matches go to the
  **required** bucket. The two results are unioned, with `required` winning any conflict.
  This reverses the original drafting of this AC, which made `skills_desc` primary: it is
  **98.0% null**, present on only ~2,500 of 123,849 postings (measured 2026-09-08). A design that
  leaned on it would have extracted skills for 2% of the corpus. Because `skills_desc` covers so
  little, no behavior may depend on its presence — it can only add skills, never gate them.
- **AC-2.4** — Description section splitting uses heading regexes:
  `required|requirements|qualifications|must have|minimum qualifications|basic qualifications`
  opens a required section; `preferred|nice to have|bonus|a plus|desired|preferred qualifications`
  opens a preferred section. A section runs until the next recognized heading or end of text.
- **AC-2.5** — When no preferred section is detected, `preferred_skills` is an **empty list** — not
  a copy of required, and not a guess. The scorer's drop-and-renormalize rule (AC-9.4) then applies.
  Preferred skills are never fabricated to fill the field.
- **AC-2.6** — A posting yielding zero extracted skills is **dropped from the serving corpus** at
  ingestion, and the dropped count is recorded in `coverage_stats.json`. Rationale: skill overlap
  is 38% of the score, and a posting that cannot be scored on it cannot be ranked honestly.
- **AC-2.7** — Extraction is measured, not assumed: `coverage_stats.json` reports the fraction of
  retained postings that have (a) a parsed required section, (b) a parsed preferred section,
  (c) a non-empty `skills_desc`, and the median count of required skills per posting. The
  `skills_desc` figure is a reportable finding in its own right, not just a diagnostic: a
  dedicated skills field that is 98% empty is a concrete Veracity example for report §5, and it
  is why extraction from unstructured description text was necessary rather than optional.
- **AC-2.8** — Alias normalization is applied before matching and is bidirectional-safe:
  `k8s`→`kubernetes`, `react.js`/`reactjs`→`react`, `ml`→`machine learning`. Aliases are data in the
  JSON artifact, not code.

**Notes / Open Questions:**
AC-2.7's measurement of preferred-section coverage directly determines whether the 8% preferred-skill
weight in REQ-9 is defensible. If fewer than ~20% of postings yield a preferred section, that weight
is buying almost nothing and REQ-9 should be revised through the spec-change protocol before freezing.

---

## REQ-3: Profile input page

**Status:** FROZEN v1.2 (2026-09-11) — IMPLEMENTED
**Traces to:** report §1 (user inputs), §8 (final application)
**Tests:** `tests/test_req3_profile.py` (validation logic only; widget rendering is not unit-tested)

**Description:**
A dedicated Streamlit page — the application's first page — where the user builds their profile.
Nothing else shares this page. It is the sole entry point for every input the pipeline consumes.

**Inputs:** User interaction.
**Outputs / Behavior:** A validated `UserProfile` object held in `st.session_state`.

**Page structure and exact controls.** The app is multi-page via Streamlit's `pages/` convention:
`app.py` → `pages/1_Profile.py`, `pages/2_Results.py`, `pages/3_Analytics.py`.

**Section A — Career documents** *(feeds the RAG knowledge base, REQ-4)*

| Field | Widget | Required | Details |
|---|---|---|---|
| Résumé | `st.file_uploader` | Yes (or paste) | Accepts `.pdf`, `.txt`, `.md`. Single file |
| Résumé (paste alternative) | `st.text_area` | — | Shown behind an "or paste instead" `st.toggle` |
| Additional documents | `st.text_area` | No | Projects, coursework, certifications — free text |
| Career goals statement | `st.text_area` | **Yes** | Free text, min 20 characters. Drives the semantic score |

**Section B — Qualifications** *(structured; extracted from documents, always manually overridable)*

| Field | Widget | Required | Details |
|---|---|---|---|
| Extract from résumé | `st.button` | — | Populates the three fields below; user may edit any result |
| Skills | `st.multiselect` (accepts new options) | Yes | Options seeded from the REQ-2 gazetteer; free entry allowed |
| Years of experience | `st.number_input` | Yes | Float, 0.0-50.0, step 0.5 |
| Highest completed education | `st.selectbox` | Yes | High School / Associate / Bachelor's / Master's / PhD |
| Currently pursuing | `st.selectbox` | No | Same ladder plus "Not currently enrolled" (default) |

**Section C — Job preferences** *(drives hard filters, REQ-6)*

| Field | Widget | Required | Details |
|---|---|---|---|
| Preferred job titles | `st.multiselect` (accepts new options) | Yes | Seeded with common tech titles; free entry allowed |
| Preferred location | `st.selectbox` with search | Conditional | Options come from the `geonamescache` city table — validated, not free text. Required unless work setting is Remote-only |
| Work setting | Three `st.checkbox` | Yes | Remote / Hybrid / On-site. **All checked by default.** At least one must be checked |
| Employment type | Five `st.checkbox` | Yes | Full-time / Part-time / Contract / Internship / Temporary (per AC-1.6a). **All checked by default.** At least one must be checked |
| Minimum salary | `st.number_input` | No | Integer, step 5000, default 0 (meaning no minimum) |
| Include jobs with no listed salary | `st.checkbox` | — | **Checked by default.** Exposes the null-salary policy as a user choice rather than a hidden rule |
| Maximum distance | `st.slider` | No | 10-250 miles, default 100. Disabled unless a location is set and Hybrid or On-site is checked. UI-layer control only — not a `UserProfile` data-model field |

**Section D — Presets and actions**

| Field | Widget | Details |
|---|---|---|
| Load a preset profile | `st.selectbox` | "Custom" (default) plus ≥3 built-in profiles. Populates every field above |
| Find matches | `st.button` (primary) | Validates, builds `UserProfile`, navigates to Results |

**Acceptance Criteria:**

- **AC-3.1** — The profile page is a standalone page. No results, scores, or dataset analytics render
  on it.
- **AC-3.2** — Every field in the tables above exists with the specified widget type, default, and
  required/optional status. **Presentation rules (v1.1, 2026-09-11):** values stored in normalized
  lowercase (skills, job titles) are **displayed capitalised** and mapped back on selection —
  normalization is a matching concern and must not leak into the form. Job-title options are derived
  from the corpus (the 300 most common normalized titles), **not a hand-written list**, and both
  skills and titles accept free text alongside the options, so a value the vocabulary happens to lack
  is never a dead end. Education is split into *Highest completed* and *Currently studying for*,
  grouped under an Education heading, with years of experience under its own heading rather than
  beside them.
- **AC-3.11** — **The profile survives navigation.** Returning to the form after a search shows it
  filled in as submitted, with a **Clear form** control to start over. Re-entering a résumé and twelve
  fields to change one salary figure is not an acceptable cost for adjusting a search.
  **Keyed widgets are not sufficient (v1.1, 2026-09-11).** Streamlit discards the `session_state`
  entry for any widget not rendered on the current run, so navigating to Matches and back emptied the
  form despite every control being keyed. The submitted profile is a plain object and does survive,
  so the form is restored from it when the widget keys are missing.
- **AC-3.12** — **A saved profile can be updated in place.** When one is loaded, the form says which
  profile is being edited and offers **Update “name”** as the primary action, with save-as-a-copy
  available but requiring a new name. Editing a saved profile and being offered only "save as new"
  makes the obvious action the one the interface does not have.
- **AC-3.10** — **Résumé skills pre-fill the picker.** On résumé entry the REQ-2 extractor runs over
  the user's own text and adds what it finds to the selection — visible and removable, never applied
  silently. The form previously asked the user to recall skills from memory while their résumé sat in
  the next field and the extractor that reads job postings could have read it equally well; a user who
  did not think to list "machine learning" was scored as not having it.
- **AC-3.9** — **Validation is silent until submit.** Nothing is reported while the user is still
  filling the form; messages appear only after Search is pressed, naming the fields that block it.
  Warning about a field the user has not reached yet is noise, not help.
- **AC-3.3** — Work setting and employment type are **sets**, not single values. Selecting Remote and
  Hybrid together is valid and produces `accepted_work_settings = {"Remote", "Hybrid"}`.
- **AC-3.4** — Validation blocks submission and shows a specific message when: no résumé is supplied
  by either route; career goals is under 20 characters; skills is empty; work setting has zero boxes
  checked; employment type has zero boxes checked; or location is empty while Hybrid or On-site is
  checked. Each condition is independently unit-tested against the validation function.
- **AC-3.5** — Location input is validated against the `geonamescache` city set. A value not in the
  set cannot be submitted; the control offers matching cities as the user types.
- **AC-3.6** — "Currently pursuing" defaults to "Not currently enrolled" and, when set, produces
  `education_in_progress` on the profile, consumed by AC-9.6.
- **AC-3.7** — At least three preset profiles ship, and at least one differs materially from the
  author's own (different domain, seniority, and geography). Presets are what supply the
  "top-5 for different profiles" evaluation in REQ-13.
- **AC-3.8** — Résumé extraction is optional. Every field it populates remains editable, and the app
  is fully usable with extraction never invoked.

**Notes / Open Questions:**
The maximum-distance slider is intentionally a UI filter rather than a profile field: the user enters
a city and the system derives closeness. The slider only adjusts the cutoff applied to a distance the
system computed itself.

---

## REQ-4: Personal knowledge base (RAG index)

**Status:** FROZEN v1.0 (2026-09-09)
**Traces to:** report §6 (analytics and matching), §4 (co-design enhancements — RAG)
**Tests:** `tests/test_req4_personal_kb.py`

**Description:**
Chunk, embed, and index the user's career documents so that per-job evidence can be retrieved and
quoted. This is what lets explanations cite real sentences instead of templated summaries.

**Inputs:** Résumé text, additional documents, career goals statement (REQ-3 Section A).
**Outputs / Behavior:** A list of embedded chunks with metadata, held in session, plus a
`retrieve(job_text, k)` function returning the top-k chunks by cosine similarity.

**Acceptance Criteria:**

- **AC-4.1** — Chunking splits on blank lines (paragraph blocks), merging any block under 40
  characters into its neighbour, plus one dedicated chunk for the career-goals statement. A typical
  résumé yields 8-15 chunks. **Simplified 2026-09-08** from heading-driven semantic-section
  detection: résumés are already visually blocked, so paragraph splitting lands on nearly the same
  boundaries, while heading detection across arbitrary résumé formats is brittle and would have been
  the largest source of edge cases in REQ-4 for no measurable retrieval gain.
- **AC-4.2** — Each chunk carries metadata `{section, source, char_span}`. `char_span` indexes back
  into the original document so every displayed quote is provably verbatim source text (AC-11.5),
  never paraphrase.
- **AC-4.3** — The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, pinned by exact name
  in one constants module and used for the personal KB, the job corpus, and the retrieval query.
  Mixing embedding spaces is a correctness error, not a tuning choice.
- **AC-4.4** — `retrieve(job_text, k)` returns exactly `min(k, n_chunks)` chunks ordered by
  descending cosine similarity, each with its similarity score attached.
- **AC-4.5** — Given a job description heavy in one skill area, retrieval returns the résumé chunk
  covering that area ahead of unrelated chunks. Tested with a fixture résumé and two contrasting
  job descriptions.
- **AC-4.6** — Structured fields (skills, years, education) extracted from documents are stored on
  the profile separately from the embedded chunks. The same text feeds both, but the structured and
  unstructured paths never share a representation.

**Notes / Open Questions:**
The personal KB is tiny (tens of chunks), so exact search over a NumPy array is sufficient — no FAISS
index is needed on this side.

---

## REQ-5: Job corpus indexing and calibration

**Status:** FROZEN v1.2 (2026-09-11) — IMPLEMENTED
**Traces to:** report §5 (storage, indexing), §6 (embeddings, BM25)
**Tests:** `tests/test_req5_indexing.py`

**Description:**
Build and persist the dense vector index, the BM25 index, and the semantic calibration constants.
All three are computed once and cached; recomputing on app start would make the application unusable.

**Inputs:** `data/processed/jobs_tech.parquet`.
**Outputs / Behavior:** `data/index/{embeddings.npy, faiss.index, bm25.pkl, calibration.json, manifest.json}`.

**Acceptance Criteria:**

- **AC-5.1** — One dense vector per job, embedded from **`title + description`** — deliberately
  **not** the skills list, which has its own score components (AC-9.12). Truncated to the model's
  **256-token maximum** (measured on `all-MiniLM-L6-v2`, 2026-09-08; an earlier draft of this AC
  said 512, which was wrong). `title` is placed first so the highest-signal text is always inside
  the window. Job descriptions are not chunked — one job is one retrieval unit.
- **AC-5.2** — The BM25 index (`rank_bm25`) is built over **`title + skills + description`** — the
  full text, untruncated — and persisted. The two indexes deliberately cover different text: BM25
  carries keyword precision on named tools over the whole posting, the dense vector carries meaning
  over the opening. This is the division of labor that makes hybrid retrieval worth having, and it
  is also what keeps the ~66-token content beyond the dense window still searchable.
- **AC-5.3** — A `manifest.json` records the dataset row count, a content hash of the source Parquet,
  and the embedding model name. On app start, indexes are loaded from disk if the manifest matches
  the current corpus and model, and rebuilt only if it does not.
- **AC-5.4** — Loading cached indexes for the corpus (18,990 rows as built) completes in under
  10 seconds.
- **AC-5.5** — Calibration constants (D8) are computed at index-build time, against the population
  the score is actually applied to. For each of the ≥3 reference profiles, take its **top-`RETRIEVE_K`
  most similar jobs** — the candidates that survive to scoring — pool those similarities across
  profiles, and persist the 5th and 95th percentiles to `calibration.json`.
  **Anchors are per-profile, computed at query time** from that profile's similarity distribution
  over the **filter survivors** — the population that is actually scored. Anchoring on the corpus-wide
  top-`RETRIEVE_K` instead (v1.1) put the p5 floor above most of what gets scored: a job ranking first
  by *total* score is frequently not top-200 by *goals* similarity, so both semantic components
  returned **0.00 for four of the top five results**, making 30% of the weight structurally
  unavailable — a perfect skills match scored 67/100 for that reason alone (v1.2, 2026-09-11) (~5 ms — one matrix multiply against precomputed vectors). Pooling anchors across
  reference profiles was tried and failed: résumés differ in length and vocabulary, so their
  similarity scales differ, and pooled anchors that discriminated for one profile mapped another's
  entire top-5 to 0.0. Pool-independence (D8) still holds — the anchors depend only on the profile
  and the corpus, never on which candidates survived filtering, so adding a job to a search cannot
  change any score. What is given up is cross-*profile* comparability, which was never a use case:
  nobody compares their 77 to another user's 77.
  Amended v1.1 (2026-09-09): the original sampled 2,000 *random* jobs. Measured, that put the p95
  anchor at 0.500 while the top-200 retrieved candidates had a p5 of **0.501** — so ~95% of everything
  scored clipped to 1.0 and the two semantic components (30% of the weight) became a constant across
  the ranking. That is precisely the narrow-band failure AC-9.11 exists to prevent, relocated rather
  than fixed. Anchoring on retrieved candidates keeps the constants **fixed at build time**, so D8's
  pool-independence is unaffected: adding a job to a live search cannot change them.
- **AC-5.6** — At least one reference profile is materially unlike the author's own, so the background
  distribution is not calibrated to a single person.
- **AC-5.7** — Calibration uses only precomputed vectors — no additional embedding pass beyond the
  reference profiles themselves. Its build-time cost is a few matrix multiplications against the
  corpus (well under a second) and its query-time cost is one subtraction and one division per job.
- **AC-5.8** — `calibration.json` is invalidated by the same manifest check as the indexes: changing
  the corpus or the embedding model forces recomputation.

---

## REQ-6: Hard filters

**Status:** FROZEN v1.1 (2026-09-11) — IMPLEMENTED
**Traces to:** report §6 (matching method), §4 (co-design — corrected AI's missing filter layer)
**Tests:** `tests/test_req6_filters.py`

**Description:**
Non-negotiable constraints applied as vectorized boolean masks over the full serving corpus,
**before** retrieval (D2). A job failing any filter is eliminated, not penalized.

**Inputs:** `UserProfile`, the jobs DataFrame.
**Outputs / Behavior:** A filtered DataFrame plus a per-filter survivor count (the funnel).

**Acceptance Criteria:**

- **AC-6.1** — Filters run before retrieval, over the whole corpus, as pandas boolean masks.
  Filtering the 18,990-row corpus completes in under 200 ms.
- **AC-6.2** — **Location.** Decision order: (a) if the user's accepted settings are Remote-only, a job
  passes only if `is_remote`; (b) a job with `is_remote` true passes regardless of distance; (c)
  otherwise, if both the user's city and the job's city geocode, pass when haversine distance ≤ the
  distance cutoff; (d) if either fails to geocode, fall back to normalized city/state string equality;
  (e) if that also fails, the job is dropped. All five branches are independently tested.
- **AC-6.3** — Location normalization lowercases, trims, expands state names to codes (Missouri → MO),
  and strips "metro", "area", and "greater" qualifiers before comparison.
- **AC-6.4** — The geocoding resolution rate over the corpus is measured and recorded in
  `coverage_stats.json`. The location filter is the most consequential one; its coverage is reported,
  not assumed.
- **AC-6.5** — **Salary.** A job passes when `salary_max >= user.min_salary`. A job with
  `salary_listed == False` passes if and only if the user checked "include jobs with no listed salary",
  and is tagged `salary_unverified` for display. When `min_salary == 0` the filter is a no-op.
- **AC-6.6** — **Work setting.** A job passes when `job.work_setting ∈ user.accepted_work_settings`.
  A job with `work_setting == "Unknown"` passes and is tagged `work_setting_unverified`, consistent
  with the salary policy.
- **AC-6.7** — **Employment type.** A job passes when `job.employment_type ∈ user.accepted_employment_types`.
  `Unknown` passes and is tagged, same policy.
- **AC-6.8** — **Skill floor.** A job passes when at least one of its `required_skills` is in the
  user's normalized skill set **and is not a generic office tool** (`excel`, `word`, `powerpoint`,
  `outlook`, `sharepoint`, `windows`). Because zero-skill postings were dropped at ingestion
  (AC-2.6), this filter never encounters an empty required list.
  **Generic-tool exclusion added v1.1 (2026-09-11).** `excel` is the most common extracted skill in
  the corpus (3,498 postings) and the top co-occurring pairs are `excel+word` and `excel+powerpoint`.
  Sharing Microsoft Word with a posting is not evidence of fit, and it was the route by which
  unrelated roles reached the results — the radiology posting that prompted this has
  `required_skills=['word']`. Generic tools still **score** normally under AC-9.3; they simply cannot
  be the sole reason a job is considered. A core/generic split was declined on 2026-09-08 and
  re-opened when measurement showed it producing exactly the reported symptom.
- **AC-6.9** — Filters compose with AND, and the function returns a funnel: the survivor count after
  each individual filter. This is surfaced in the UI (AC-11.2) and cited in the report.
- **AC-6.10** — When the funnel empties, the system reports which filter eliminated the most candidates
  and suggests relaxing it. It never silently returns zero results.

**Notes / Open Questions:**
Location and work setting are deliberately separate filters composed with AND: location governs
geography, work setting governs arrangement, and a remote job is exempt from the distance check.
They must not double-gate.

---

## REQ-7: Hybrid retrieval

**Status:** FROZEN v1.0 (2026-09-09)
**Traces to:** report §6 (BM25, embeddings, hybrid retrieval), §9 (retrieval comparison)
**Tests:** `tests/test_req7_retrieval.py`

**Description:**
BM25 and dense retrieval run over the filter survivors and are fused with Reciprocal Rank Fusion into
a single candidate list.

**Inputs:** Filtered job set, `UserProfile`.
**Outputs / Behavior:** Top-N candidate job IDs (N = 200, or all survivors if fewer).

**Acceptance Criteria:**

- **AC-7.1** — Three retrieval queries are issued separately — career-goals statement, skills list,
  preferred titles — and all result lists are fused. They are not concatenated into one string, which
  would dilute the dense query vector.
- **AC-7.2** — Fusion is RRF: `score(doc) = Σ_retrievers 1 / (60 + rank_in_retriever)`. Raw BM25 scores
  and cosine similarities are never summed — they live on incompatible scales.
- **AC-7.3** — Each retriever is independently callable (`bm25_only`, `dense_only`, `hybrid`) so REQ-13
  can compare them without reconstructing the pipeline.
- **AC-7.4** — When the survivor set is smaller than N, all survivors are returned and no error is raised.
- **AC-7.5** — Retrieval over the 18,990-row corpus returns in under 2 seconds on CPU.

---

## REQ-8: Cross-encoder reranking — **OPTIONAL, EVALUATED**

**Status:** FROZEN v2.0 (2026-09-11) — implemented, **off by default**, compared in REQ-13

**Reinstated as an optional, measured stage (v2.0, 2026-09-11).** Removed on 2026-09-08 as
unjustifiable; brought back because it is a named methodology the evaluation should *compare* rather
than assert away. Implemented, **disabled by default**, with cost and benefit reported in AC-13.1's
four-way comparison.

A cross-encoder reads the (profile, job) pair as a **single** input and attends across both, unlike a
bi-encoder whose vectors are computed separately. More accurate per pair, but one forward pass per
pair — so it can only run over a shortlist.

**Why it stays off by default.** At `k=400` hybrid retrieval already recovers **100%** of the true
top-20 by score, so there is nothing left for a reranker to recover, and it would add seconds to a
sub-second search. The explainability objection stands too: a cross-encoder score is undecomposable,
so it cannot enter a score built to be decomposable (D10).

**Acceptance criteria**
- **AC-8.1** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores the retrieved candidates and the top-K
  are retained. Enabled by explicit flag; never on by default.
- **AC-8.2** — Cross-encoder scores never appear in the match score, the breakdown, or the UI.
- **AC-8.3** — AC-13.1 compares four configurations — BM25 · dense · hybrid · hybrid+rerank —
  reporting recall **and added latency**, so cost is visible beside benefit.
- **AC-8.4** — If reranking does not improve recall over hybrid alone, that is the finding and the
  stage stays off.

**Why.** It was accepted at Stage 2 from the AI's recommendation *"add a reranker + feedback →
re-rank loop"* (co-design prep §1). The feedback half was already cut; this is the other half.

Three reasons it does not earn its place:

1. **The assignment does not ask for it.** Report §6 (methods) and §9 (evaluation) name BM25,
   embeddings, and hybrid retrieval. Reranking appears nowhere in the instructions.
2. **Filter-first made it redundant (D2).** Candidates are already restricted to jobs the user is
   eligible for, and the weighted score ranks them. Trimming 200 → 50 saves nothing measurable —
   scoring 200 candidates is eight arithmetic components plus one cosine max over ~10 résumé chunks
   — and a job that survives retrieval without merit scores low and never reaches the top 5.
3. **It carries an opacity cost in either role.** As a score component it puts a term in the
   breakdown that cannot be justified to the user. As a pruner it can silently drop a job the user
   would have wanted, with no signal and no recourse. Pruning relocates the opacity rather than
   removing it — the argument for it was only ever that retrieval is already opaque while the score
   need not be.

**What is kept:** AC-13.1's retrieval comparison still runs BM25 vs. dense vs. hybrid RRF, which is
exactly what report §9 names. The `ms-marco` model is already cached, so this could be revisited if
the evaluation shows hybrid retrieval underperforming — but it would have to earn its way back in
on measured evidence.

**Report value:** an AI recommendation accepted at Stage 2 and reversed at Stage 3 with a measured
reason is stronger evidence of critical evaluation than shipping it would have been (report §4).

---

## REQ-9: Weighted match score

**Status:** FROZEN v1.0 (2026-09-09)
**Traces to:** report §6 (combined score), §4 (corrections to AI scoring), §2 (Stage 1 lineage)
**Tests:** `tests/test_req9_scoring.py`

**Description:**
Eight components, each normalized to [0,1], combined by fixed weights into a 0-100 score. Every
component is a pure function over typed inputs, unit-testable without the app or any model.

**Weights:**

| Block | Component | Weight |
|---|---|---:|
| **Deterministic (70%)** | Required-skill overlap | 30% |
| | Preferred-skill overlap | 8% |
| | Experience alignment | 15% |
| | Education alignment | 4% |
| | Role/title match | 5% |
| | Location proximity | 8% |
| **Semantic (30%)** | Career goals ~ job description | 13% |
| | Résumé evidence ~ job description | 17% |
| | **Total** | **100%** |

The 70/30 split follows Stage 1: deterministic components produce evidence that can be shown to the
user directly, while semantic similarity is a supporting signal that cannot be pointed at. The 13/17
semantic split is carried over from the Stage 2 co-design prep unchanged, so the lineage is traceable.

**Acceptance Criteria:**

- **AC-9.1** — Score = `round(Σ (sub_score × weight) × 100)`, an integer in [0, 100].
- **AC-9.2** — Tier thresholds: Strong ≥ 80, Good 65-79, Moderate 45-64, Low < 45. (Retained from the
  Stage 2 AI design — a "kept" item for report §4.)
- **AC-9.3** — **Required-skill overlap** = `|matched_required| / |required|`, computed by exact set
  membership on normalized canonical terms (AC-2.2). Substring matching is prohibited.
  **Evidence damping (added 2026-09-09):** the ratio is multiplied by
  `min(1, |required| / 3)`, so a posting stating one requirement caps at 0.33 and one stating two
  caps at 0.67; three or more is unaffected.
  Measured on the built corpus, **27.1% of postings yield exactly one required skill** and 43.1%
  yield two or fewer. Undamped, a posting listing one skill you happen to have scores a perfect 1.0
  on the heaviest component — beating a posting listing ten of which you match eight (0.80). The
  thin posting wins on far weaker evidence, which inverts the ranking the component exists to
  produce. Damping is a confidence discount on the *evidence*, not a penalty on the job:
  **a ratio computed from one observation should not carry the same weight as one computed from
  ten.** Explained to the user as: "this posting states only one requirement, so we are less
  confident in the match."
  Rejected alternative: raising AC-2.6's floor from >=1 to >=3 required skills, which would delete a
  further 56.3% of the corpus. A posting with one extracted skill may simply use vocabulary the
  gazetteer lacks — discounting confidence is honest, deleting the job is not.
- **AC-9.4** — **Preferred-skill overlap** = `|matched_preferred| / |preferred|`. When `preferred` is
  empty, the component is **dropped and the remaining seven weights are renormalized to sum to 1.0**
  for that job. It is never defaulted to 1.0 (inflates) or 0.0 (punishes a data gap).
- **AC-9.5** — **Experience alignment**, asymmetric (D7), with `gap = job_min_years - user_years`:

  ```
  gap >  0   →  0.75                        if gap <= 1.0
                0.50                        if gap <= 2.5
                max(0.15, 0.50 - 0.10*gap)  otherwise
  -3 <= gap <= 0  →  1.0                                       # plateau
  gap < -3   →  max(0.60, 1.0 - 0.05 * (-gap - 3))             # over-qualified
  job_min_years is None  →  0.5                                # neutral
  ```

  Each branch is independently tested, including the 0.60 floor and the neutral case.
- **AC-9.6** — **Education alignment** (D6), on the ladder High School 1 < Associate 2 < Bachelor's 3
  < Master's 4 < PhD 5:

  ```
  effective = (in_progress_level - 0.5) if in_progress else highest_completed

  effective >= required            →  1.0
  (required - effective) <= 1.0    →  0.6
  otherwise                        →  0.2
  education_required is None       →  0.5   # neutral
  ```

  Tested explicitly: a user pursuing a Master's (effective 3.5) scores 1.0 against a Bachelor's
  requirement and 0.6 against a Master's requirement.
  **The 4% weight is provisional (Q6).** It was set on the assumption that most postings state no
  requirement, leaving this component pinned at its neutral 0.5. Measured 2026-09-08, AC-1.7 parses
  a requirement from **58.5%** of the corpus, so it discriminates on most rows rather than few.
- **AC-9.7** — **Role/title match** = token containment between the user's preferred titles and
  `title_normalized`, after stripping seniority modifiers ("senior", "sr", "lead", "staff", "principal",
  "junior", "jr", "I", "II", "III").
- **AC-9.8** — **Location proximity**, reusing the distance computed in REQ-6:

  ```
  remote job, user accepts remote        →  1.0
  distance resolved                      →  exp(-miles / 60)
  unresolved, string match               →  0.8
  unresolved, no string match            →  0.3
  ```

- **AC-9.9** — **Career goals ~ description** = calibrated cosine between the career-goals embedding
  and the job embedding.
- **AC-9.10** — **Résumé evidence ~ description** = calibrated maximum cosine over the retrieved
  personal chunks for that job, computed against the job's **stored index vector** rather than a
  freshly embedded copy of its text, so the evidence displayed to the user (AC-11.5) and the
  evidence that produced this sub-score are provably the same measurement and cannot diverge —
  **excluding the career-goals chunk**, which has its own component
  (AC-9.9). Without that exclusion the two semantic components returned identical values on most
  results — the goals chunk simply won the max — so 30% of the weight was one signal counted twice,
  exactly what AC-9.12 forbids. A profile with no résumé content beyond its goals statement yields
  no evidence and the component is dropped and renormalized, as with preferred skills (AC-9.4). The chunks producing this score are the same ones surfaced to the
  user as evidence in AC-11.5, so the score and the justification shown to the user cannot disagree.
- **AC-9.11a** — Calibrated semantic sub-scores must **discriminate within the scored candidate
  set**, not merely be bounded. A calibration that maps every candidate to 1.0 has the same practical
  effect as the raw narrow band it replaced.
  **Measured over the top 20, not the top 5 (v1.1, 2026-09-11).** With a small survivor pool a top-5
  can legitimately sit entirely above p95 — for the security profile, 5 of 229 survivors *is* the
  top 2.2% — so identical values there are a correct outcome, not a flat calibration. The property
  that matters is spread across the population the anchors describe, so the assertion is made over
  the top 20 and additionally requires at least one of the two semantic components to vary within
  the top 5.
- **AC-9.11** — Calibration (D8) maps raw cosine through
  `clip((raw - p5) / (p95 - p5), 0, 1)` using the persisted constants from AC-5.5. Scores do **not**
  depend on the composition of the candidate set: adding an unrelated job to the pool changes no other
  job's score. This is asserted directly by a test.
- **AC-9.12** — No input signal feeds two components. Tested directly: on a real top-5 the
  career-goals and résumé-evidence sub-scores must not be identical across results. The text embedded for the semantic components
  excludes the skills list, which already has its own components.
- **AC-9.13** — Every component has a defined behavior for missing input — drop-and-renormalize, or an
  explicit neutral 0.5. No component silently defaults to 1.0.
- **AC-9.14** — The scorer returns a full breakdown: each component's name, sub-score, weight, and
  weighted contribution, with the contributions summing to the reported score.

**Notes / Open Questions:**
The 8% preferred-skill weight is provisional pending AC-2.7's coverage measurement, and the 13/17
semantic split is provisional pending AC-13.6's correlation check. Both are revised through the
spec-change protocol if the evidence says so — not silently in code.

---

## REQ-10: ~~Grounded explanation layer~~ — **REMOVED**

**Status:** REMOVED 2026-09-08 (never implemented; no code was written against it)

This requirement specified a hosted-LLM explanation layer generating a narrative paragraph per
ranked job. It is removed from scope. The number is retained rather than reused so that earlier
commits, the changelog, and the report remain readable.

**Why it was removed.** The co-design prep's section (6) — "Practice with Agentic AI", an initial
prompt and a revised version with a fixed rubric — is an exercise in *directing* an AI, and belongs
in the report as exactly that. It was never a product requirement. Building it in would have added
an API key, a per-run cost, and a reproducibility barrier (AC-14.3: a grader must be able to run
this repo) in exchange for prose. Report §6 asks an LLM's role to be explained only *if* an LLM is
used.

**What is kept, and why nothing important is lost.** The evidence survives the removal — it was
never the LLM that produced it. REQ-4's semantic retrieval already finds the résumé chunks that best
support each job, and AC-4.2's character spans make every displayed quote provably verbatim. So the
user still sees *"you match Airflow — here is the line from your résumé that shows it"*, and the
system still improves markedly on the Stage 2 AI code, whose `match_summary` was string
concatenation (`agent-exercise:src/matcher.py`) carrying no evidence at all. What is given up is a
generated narrative paragraph, not the grounding. Display moves to **AC-11.5**.

**Stretch, if time allows:** a local Hugging Face model (no API key, no cost, runs offline) could
generate the narrative from the already-retrieved evidence. Listed under Non-Goals. Any such layer
is display-only and may never alter a score — D11 holds regardless of where the text comes from.

**The prompt-engineering exercise still appears in the report**, sourced from the co-design prep
document, as Stage 2 practice rather than as shipped code.

---

## REQ-11: Results page

**Status:** FROZEN v1.3 (2026-09-11)
**Traces to:** report §8 (final application), §1 (expected outputs)
**Tests:** manual; screenshots are the deliverable

**Description:**
A dedicated page rendering the ranked top 5 with full score transparency.

**Acceptance Criteria:**

- **AC-11.1** — Top 5 jobs render as cards showing title, company, location, salary range (or an
  "unlisted" badge), employment type, work setting (with an "inferred" badge where applicable), the
  0-100 match score, and the tier badge.
- **AC-11.2** — A funnel summary shows corpus size → survivors after each filter → candidates retrieved
  → final 5. This makes the pipeline legible and is a report figure.
- **AC-11.3** — Each card expands to a component breakdown table: component, sub-score, weight, weighted
  contribution, with contributions summing to the displayed score. Renormalized weights (AC-9.4) are
  shown as renormalized.
- **AC-11.4** — Matched skills display with their supporting résumé quote; missing skills display
  separately.
- **AC-11.5** — Each matched skill renders alongside the supporting résumé chunk retrieved for that
  job (REQ-4), with the matched span highlighted. Quotes are sliced from the source document by the
  character spans of AC-4.2 and are verbatim by construction — the UI never paraphrases, summarizes,
  or generates résumé text. A skill with no retrieved chunk above the similarity floor is shown as
  matched-without-evidence rather than given an invented justification.
- **AC-11.6** — A score-distribution chart across the returned results is shown.
- **AC-11.8** — **Job & Candidate comparison.** Each result shows the job's facts and the user's
  facts side by side — the comparison the score is *about*. Previously only the job was shown, which
  asked the reader to hold their own profile in their head.
- **AC-11.9** — **Evidence table.** Retrieved evidence renders as a typed table with **Type /
  Evidence / Source** columns rather than loose quotes: résumé sections carry their section name and
  document, job requirements carry "Job description", profile fields carry "Profile". Every résumé
  row is sliced by AC-4.2 character span and is verbatim by construction.
- **AC-11.10** — **Component bars.** Each component renders as a filled bar with its earned points
  out of its maximum (e.g. `21.3 / 30.0`), not just a decimal. Points still sum to the displayed
  score (AC-11.3).
- **AC-11.11** — **Gaps & unknowns table** with **Issue / Details / Impact** columns, covering two
  distinct things: *gaps* (a required skill the user lacks) and **unknowns** (facts the posting
  never stated). Unknowns are derived from the missing-data flags the pipeline already records —
  `salary_listed`, `work_setting_inferred`, `min_years_exp_source`, absent `education_required`.
  Surfacing them is the point: a score computed partly from neutral defaults must say so, or it
  implies a confidence the data does not support.
- **AC-11.12** — **Human decision panel.** Per-result actions (applied / saved for later / not
  interested) that take effect **only when explicitly confirmed** — selecting a radio must not file
  anything, since a stray click should not silently change what the user decided — plus an explicit
  weight-adjustment control that re-scores the same candidate set and re-ranks.
  Filed decisions are listed on their own **My jobs** page, grouped by status, removable, and
  exportable as CSV. An action with no destination is not an action (v1.3, 2026-09-11).
  This is **not** the feedback loop cut in Non-Goals: that was *learning* weights from thumbs
  up/down, which a handful of signals cannot support. This is the user setting weights directly and
  seeing the consequence — explicit control rather than inference, and it costs one re-score of an
  already-filtered set.
- **AC-11.14** — **Link and expiry disclosure.** Each result links to its original posting and
  states when it closed. The historical corpus means every posting is expired; the interface says so
  plainly rather than implying a live application is possible.
- **AC-11.13** — **Verdict.** Each result carries a one-line verdict naming what to check before
  applying, drawn from that job's own unknowns.
  Amended v1.2 (2026-09-11): the page-level "takeaway" paragraph explaining what an explainable
  match is *for* was removed. The application is a job-search tool someone uses, not a demonstration
  of its own method — that explanation belongs in the report, where it is assessed. The per-result
  verdict stays, because it tells the user something actionable about *this* job rather than about
  the system.
- **AC-11.7** — Every field collected on the profile page is either used by a filter or a score
  component, or is not collected. No field is stored and then ignored — the exact defect in the Stage 2
  AI code, where `education_level` and `min_salary` sat unused on `UserProfile`.

---

## REQ-12: Dataset analytics and visualization

**Status:** FROZEN v1.0 (2026-09-09)
**Traces to:** report §7 (Big Data Analytics and Visualization — **required**), Minimum Visual Evidence #4
**Tests:** `tests/test_req12_analytics.py` (aggregation correctness)

**Description:**
Descriptive analytics over the job corpus, computed as DuckDB aggregations and rendered on a dedicated page.
This is analysis *of the dataset*, distinct from the *result-level* charts in REQ-11.

**Acceptance Criteria:**

- **AC-12.1** — Six aggregations are computed as DuckDB `GROUP BY` queries and written to `data/processed/analytics/`:
  top 20 job titles; top 25 requested skills; geographic distribution by state; salary distribution
  and median by title family; remote vs. hybrid vs. on-site counts; experience-level distribution;
  top 20 hiring companies.
- **AC-12.2** — The analytics page renders at least four of these as charts, with at least one
  geographic and one salary visualization. Every chart is labelled with what the corpus actually is
  (D3: IT/engineering/analytics/QA/science job functions, ~25% strictly software/data) — not as
  "tech jobs", which would overstate it.
- **AC-12.3** — A skill co-occurrence view shows which skills appear together most often — the finding
  that makes the scoped tech corpus worth having.
- **AC-12.4** — Aggregation outputs are committed as CSVs so the report can cite exact numbers without
  a pipeline run.
- **AC-12.5** — Each aggregation is unit-tested against a small fixture with hand-computed expected values.

---

## REQ-13: Evaluation notebook

**Status:** FROZEN v1.1 (2026-09-09)
**Traces to:** report §9 (Results and Evaluation), §10 (comparison)
**Location:** `notebooks/evaluation.ipynb`

**Description:**
Side-by-side evaluation of *method quality*, distinct from the AC-derived tests, which verify *spec
compliance*.

**Acceptance Criteria:**

- **AC-13.1** — **Retrieval comparison** over the configurations built — BM25 only, dense only,
  hybrid RRF — reporting precision@10 on the
  **unfiltered** corpus, which is the honest way to compare retrieval methods.
  **Primary metric — recall of the top-scoring set (v1.1, 2026-09-09).** Score *every* filter
  survivor, take the true top-20 by final match score, and report each retrieval method's
  `recall@k` against that set: *does retrieval surface the jobs that scoring would rank highest?*
  This needs no human labels, is not circular (retrieval and scoring use different signals — BM25
  and embeddings versus eight weighted components), and answers the question the comparison exists
  for: whether hybrid retrieval earns its cost.
  **Secondary — pooled judgment.** Run every configuration, take the union of their top-10s, judge
  each job once, score all configurations against that shared pool. Pooling cannot under-credit a
  configuration for surfacing a good job nobody thought to pre-label.
  *Why the change:* the original title-family relevance proxy returned precision@10 of 1.00 for both
  BM25 and hybrid across all three profiles — a metric that scores nearly everything relevant cannot
  separate the methods. That was a limitation of the proxy, not evidence the methods are equivalent.
  The proxy is retained as a cross-check and the pools are still exported for manual labelling.
- **AC-13.2** — **End-to-end evaluation** with filters enabled, reporting the top 5 for each of the
  three preset profiles from AC-3.7.
- **AC-13.3** — **Latency**, measured per stage (filter, retrieve, score, evidence) as median and
  p95 over ≥20 runs.
- **AC-13.4** — **Scalability**: the ingestion job is timed over increasing corpus sizes — the ~30k
  tech subset, the full ~124k LinkedIn corpus, and the ~786k-row `data_jobs` corpus — and plotted,
  with peak memory recorded alongside wall-clock. This is where the Volume claim is actually
  evidenced. **Optional extension (stretch):** a minimal PySpark implementation of the same
  ingestion job, timed over the same three sizes, to locate the crossover point where distributed
  execution starts to win. A measured engine comparison is a stronger report result than an
  asserted tool choice.
- **AC-13.5** — **Calibration evidence**: raw-cosine and calibrated-score histograms side by side,
  demonstrating the correction of the Stage 2 magic-multiplier failure mode.
- **AC-13.6** — **Component sensitivity**: each score component is zeroed in turn and the change in
  the top 5 is reported, identifying components that cannot discriminate. **Run across every
  reference profile, not one (v1.1, 2026-09-09):** a single profile's result is not a property of
  the weight. The data-science profile's top results are mostly remote, so location is constant
  *for it* — a profile that rejects remote would see the same component discriminate. A conclusion
  about a weight requires agreement across profiles. Includes the correlation between
  the two semantic components; if r > 0.8 they are double-counting one signal and REQ-9 is revised
  through the spec-change protocol.
- **AC-13.8** — Every table and figure the report cites from this notebook is exported to
  `notebooks/figures/`.

---

## REQ-14: Reproducibility and documentation

**Status:** FROZEN v1.0 (2026-09-09)
**Traces to:** report §11 (GitHub and Reproducibility), §10 (comparison table)

**Acceptance Criteria:**

- **AC-14.1** — `README.md` independently explains the dataset, architecture, processing pipeline,
  analytics, execution steps, and example results. A reader never opens this spec to understand what
  was built.
- **AC-14.2** — `requirements.txt` pins every dependency to an exact version.
- **AC-14.3** — Raw data is not committed; the README gives exact download and placement instructions,
  and the pipeline regenerates every processed artifact from raw.
- **AC-14.4** — A single documented command runs ingestion end to end, and a second launches the app.
- **AC-14.5** — The Human vs. AI vs. Human-AI comparison table (report §10) is generated with evidence
  from `git diff human-ai-codesign agent-exercise -- src/`, citing specific files and line ranges rather
  than characterizing the AI code from memory.
- **AC-14.6** — Screenshots of the profile page, results page, and analytics page are committed under
  `docs/screenshots/`.

---

## Non-Goals

Explicitly out of scope for v1.0. The report's limitations section cites this list.

**Cut entirely:**
- Feedback / thumbs-up-down re-ranking loop. Session-weight adjustment was in the draft plan and is
  deliberately dropped — it is ungraded, and a handful of thumbs cannot support weight learning.
- GitHub repository ingestion into the knowledge base.
- Three-way side-by-side job comparison and radar charts.
- Any hosted-LLM dependency, and any API key requirement at runtime (REQ-10, removed).
- Company industry, posting recency, applicant count, equity/bonus, visa sponsorship, security
  clearance as score components (D12).
- Non-tech job domains (D3).
- Live geocoding APIs; persisted cross-session state; user accounts; real-time job APIs.

**Stretch — build only if the must-ship list is complete:**
- Local Hugging Face narrative generation over the already-retrieved evidence (REQ-10 stretch note).
- LLM-based résumé field extraction (manual entry is the guaranteed path).
- A PySpark implementation of the ingestion job, for the engine-comparison timing in AC-13.4.
- Lazy explanation generation below the top 5.

**Must ship:** REQ-1 through REQ-7, REQ-9, REQ-11 through REQ-14. REQ-8 ships disabled and evaluated; REQ-10 is removed.

---

## Open Questions

| # | Question | Blocks | Resolution path |
|---|---|---|---|
| ~~Q1~~ | ~~Does the preferred-skill weight of 8% survive AC-2.7's coverage measurement?~~ **CLOSED 2026-09-09** — measured 7.0% of postings yield preferred skills. **No change needed:** AC-9.4 already drops the component and renormalizes the other seven weights when preferred is empty, so on 93% of jobs the 8% redistributes automatically and the component fires only where it has data. The measurement instead quantifies the Stage 2 AI defect — its `if preferred_skills else 1.0` would have awarded a perfect score to **93% of postings** | — | — |
| ~~Q2~~ | ~~Are the two semantic components measuring one signal?~~ **CLOSED 2026-09-09** — they *were*: the career-goals statement is itself a KB chunk and won the evidence max, so both components returned identical values on most results. AC-9.10 now excludes it. Measured correlation after the fix: **−0.11, −0.00, +0.79** across the three profiles — largely independent for two, correlated for the third | — | — |
| Q3 | Where is the single-machine/distributed crossover for this workload? | Nothing — reporting only | AC-13.4's optional PySpark comparison, if time allows; otherwise stated as a documented limitation |
| Q4 | Is the ~30k tech corpus large enough for the retrieval comparison to differentiate methods? | AC-13.1 interpretation | Comparison runs unfiltered; if differences are marginal, that is itself a reportable finding |
| ~~Q6~~ | ~~Is education's 4% weight still right at 58.5% coverage?~~ **CLOSED 2026-09-09** — **keep the weight, discard the reasoning.** AC-13.6's sensitivity run shows zeroing education replaces **2 of 5** top results, so at 4% it already discriminates; it is not the decoration the original justification assumed. The same run surfaced the opposite problem elsewhere — see Q7 | — | — |
| Q7 | Location (8%) and preferred skills (8%) changed **no** top-5 result when zeroed — 16% of the weight is inert for the tested profile | Nothing shipped; reporting only | Explanations exist for both: only 7.0% of postings list preferred skills, and the top results are mostly remote so location is constant across them. Recorded as a measured limitation rather than re-tuned one day before the deadline — re-weighting without time to re-validate every AC would be worse than reporting the number. Report §12 future work |
| ~~Q5~~ | ~~Does `matched/total_required` over-reward thin postings?~~ **CLOSED 2026-09-09** — yes. Measured: 27.1% of postings yield exactly one required skill. Resolved by evidence damping in AC-9.3, not by raising AC-2.6's floor | — | — |
| ~~Q5-orig~~ | Does `matched/total_required` over-reward thin postings? A posting from which only one skill was extracted scores 1.0 on the heaviest component (30%) for matching that one skill, while a 7-skill posting matched in full scores the same | Whether AC-2.6 or AC-9.3 needs revising | AC-2.7's median-required-skills measurement decides it, inside REQ-2. If thin postings are rare, leave it. If common, either raise AC-2.6's floor from ≥1 to ≥3 skills, or damp the component by `× min(1, total_required/3)` so a 1-skill posting caps at 0.33. Note this is independent of D3's corpus decision — it applies equally in a pure software corpus |

---

## Spec Changelog

| Date | REQ/AC changed | What changed | Why |
|---|---|---|---|
| 2026-09-11 | AC-1.11 (new), AC-11.14 (new), AC-3.11 (new) | Carry `posting_url` and `expiry_date` through ingestion; link each result to its source posting and disclose expiry; profile persists across navigation with Clear and save-as-named | `job_posting_url` is 100% present in the raw data and was dropped from the schema, so the "Applied" action had no destination. Measured: the corpus ran Dec 2023 - Apr 2024 and **100% of it expired by October 2024** — letting a user click "Applied" on a two-year-dead posting is a lie the interface was telling. Separately, the form discarded everything on navigation, so changing one salary figure meant re-entering a résumé and twelve fields |
| 2026-09-11 | D13 (→v2), AC-5.5 (→v1.2), AC-2.2 (→v1.1), AC-6.8 (→v1.1), REQ-8 (reinstated v2.0), AC-3.10 (new) | Retrieval always runs at measured `k`; calibration anchors on filter survivors; `/` and `*` added to skill boundaries with single-letter terms gated on co-occurrence; generic office tools excluded from the skill floor; cross-encoder reinstated off-by-default and evaluated; résumé skills pre-fill the form | Five problems reported from real use, four reproduced. The common thread is that each was a measurement never taken: `k=200` was a round number (hybrid hits 100% recall at 400); calibration still anchored on the retrieval population after D13 changed what gets scored, zeroing 30% of the weight; `r` matched `P/R` and mangled text because the boundary set came from three test cases; and `excel` satisfying the skill floor was declined on 2026-09-08, then re-opened when it produced exactly the reported symptom |
| 2026-09-11 | AC-3.2, AC-3.9 (new) (REQ-3 → v1.1) | Display capitalisation for normalized values; job titles derived from the corpus with free text accepted; education split into two clearly-named fields; validation deferred until submit | The form exposed internal normalization ("python, sql, aws") as if it were user-facing text; the title list was 15 strings written by hand, so a real title like "data science engineer" simply did not exist; "Education" beside "Studying for" beside "Years of experience" read as three versions of one question; and warnings fired for fields the user had not reached yet |
| 2026-09-11 | AC-11.12 (REQ-11 → v1.3) | Decisions require explicit confirmation and now have a destination — a **My jobs** page grouped by status, with removal and CSV export | Selecting a radio filed the job immediately and confirmed it in a caption at the very bottom of the page, where the user would not see it. A stray click should not silently change a decision, and an action with no destination is not an action |
| 2026-09-11 | REQ-11 §layout, app structure | Sidebar navigation replaced with a top navigation bar carrying the product name ("Job Matcher") on every page; pages are Profile · Matches · My jobs · Insights | The sidebar consumed a large share of the viewport and the product had no persistent identity. Implemented with `st.navigation(position="hidden")` plus `st.page_link`, so routing stays declarative |
| 2026-09-11 | AC-11.13, REQ-11 §layout (REQ-11 → v1.2) | Results restructured to a job-board **list + detail** pattern; the standalone landing page was folded into the search form; the page-level "takeaway" paragraph was removed | The stacked-card layout made scanning impossible — comparing two results meant scrolling past two full breakdowns, and the sidebar consumed a large share of the viewport. Job boards separate scanning from reading for a reason, so the list carries what you scan by and the detail pane carries what you read. The takeaway paragraph explained the system to its own user: this is an application someone uses, not a demonstration of its method, and that explanation belongs in the report where it is assessed |
| 2026-09-09 | AC-11.8 - AC-11.13 (REQ-11 → v1.1) | Added Job & Candidate panel, typed evidence table, component bars with points/max, a Gaps **& unknowns** table, a human-decision panel with explicit weight adjustment, and verdict/takeaway lines | Course guidance specified the components an explainable-match UI should carry. Audit found roughly half present: score, breakdown, evidence and missing skills existed; the candidate side, evidence typing, unknowns, and any decision affordance did not. The **unknowns** gap was the substantive one — the pipeline already records `salary_listed`, `work_setting_inferred`, `min_years_exp_source` and absent `education_required`, and a score computed partly from neutral defaults must disclose that or it implies unearned confidence. The weight control is explicit user adjustment, not the learned feedback loop cut in Non-Goals |
| 2026-09-09 | D13 (new), pipeline | Retrieval skipped when filter survivors ≤ 2,500; every eligible job scored exactly | AC-13.1 v1.1's recall metric showed retrieval at k=200 recovering only 45-95% of the true top-20 by score — the approximate stage was discarding what the exact stage wanted. Scoring all survivors costs ~220 ms more and gave an identical top-5 on all three profiles. Retrieval retained above the threshold as the scalability path |
| 2026-09-09 | AC-13.1 (REQ-13 → v1.1) | Primary retrieval metric changed from precision@10 against a title-family proxy to **recall of the top-scoring set**; proxy and pooled judgment retained as secondary | The proxy returned precision@10 = 1.00 for both BM25 and hybrid on all three profiles — it could not separate the methods, which is a limitation of the metric rather than evidence about the methods. Recall-of-top-scored needs no human labels, is not circular (retrieval and scoring use different signals), and answers whether hybrid earns its cost |
| 2026-09-09 | AC-13.6 (REQ-13 → v1.1) | Component sensitivity must run across every reference profile, not one | Q7's "16% of the weight is inert" rested on a single profile whose top results are mostly remote — making location constant *for that profile* rather than inert in general. A conclusion about a weight needs agreement across profiles |
| 2026-09-09 | AC-9.10 | Evidence similarity computed against the job's stored index vector rather than a re-embedded copy of its text | Guarantees the evidence shown to the user and the evidence that produced the sub-score are the same measurement; also removes a redundant embedding pass (~178 ms of a 669 ms search) |
| 2026-09-09 | AC-9.10, AC-9.12 | Résumé-evidence similarity now excludes the career-goals chunk | With it included the two semantic components returned identical values on most results — the goals chunk won the max — so 30% of the weight was one signal counted twice. This is the concrete answer to Q2, found by printing both sub-scores side by side on a real search |
| 2026-09-09 | AC-5.5 (per-profile anchors) | Calibration anchors computed per profile at query time rather than pooled across reference profiles at build time | Pooled anchors failed on measurement: résumé length and vocabulary shift the similarity scale, so anchors that discriminated for the data-science profile mapped the security analyst's entire top-5 to 0.0. Per-profile anchors depend only on profile and corpus, so D8's pool-independence is preserved |
| 2026-09-09 | AC-5.5, AC-5.7 (REQ-5 → v1.1), AC-9.11a (new) | Calibration background changed from 2,000 random corpus jobs to each reference profile's top-`RETRIEVE_K` retrieved candidates. Added AC-9.11a requiring calibrated sub-scores to discriminate within a real top-5 | Measured end to end: the random-corpus anchors were p5=0.153 / p95=0.500, but the top-200 retrieved candidates had p5=0.501 — so ~95% of everything scored clipped to 1.0 and the two semantic components (30% of the weight) contributed nothing to the ranking. Calibration was measured on the wrong population; the constants stay fixed at build time so D8's pool-independence is unaffected |
| 2026-09-09 | AC-9.3 | Required-skill overlap damped by `min(1, \|required\| / 3)` | Measured on the built corpus: 27.1% of postings yield exactly one required skill, 43.1% two or fewer. Undamped, a one-skill posting scores 1.0 on the 30% component and outranks a ten-skill posting matched 8/10. A ratio from one observation should not weigh the same as one from ten. Rejected raising AC-2.6's floor to >=3, which would delete a further 56.3% of the corpus |
| 2026-09-08 | REQ-8 (removed), D10, AC-13.1, AC-13.3, Non-Goals | Cross-encoder removed entirely rather than kept as a stretch pruner | Not asked for by the assignment (§6 and §9 name BM25, embeddings and hybrid only); made redundant by filter-first (D2), since scoring 200 candidates costs milliseconds and unmeritorious jobs score low anyway; and opaque in either role — as a score component it adds an unjustifiable term to the breakdown, as a pruner it can silently drop a good job. Accepted from the AI at Stage 2 and reversed here on measured reasoning |
| 2026-09-08 | AC-1.6 (REQ-1 → FROZEN v1.3) | Experience regex now captures ranges and takes the lower bound | The frozen regex took the upper bound: "4-7 years related business experience" parsed as 7. The field is a *minimum*, so every ranged requirement was overstated, and AC-9.5 turns an inflated `job_min_years` into a wider `gap` — penalising candidates who actually qualify. Caught by sampling parsed output against source text |
| 2026-09-08 | D3, AC-12.2 | Corpus described as "LinkedIn's IT, engineering, analytics, QA and science job functions" rather than "tech roles"; REQ-12 charts must carry that label | Decision: keep the ~75% non-software rows rather than filter them. They cannot reach a top-5 — near-zero on required-skill overlap, title match, and both semantic components — so the match score already handles relevance. A core-vs-generic skill split was considered and rejected as unnecessary work. The residual obligation is descriptive accuracy, not filtering |
| 2026-09-08 | AC-1.2 (REQ-1 → FROZEN v1.2) | Title branch of AC-1.1 narrowed to compounds only and made precision-oriented; the `TECH_CODES` branch stays recall-oriented. Sales Engineer changed from a retention example to a drop example | Measured after phase 1a ran: the title branch contributed 6,157 rows of which only 18% were software/data — financial analysts (131), board-certified behavior analysts (67), office administrators (76), data entry clerks (31). It was widening the net rather than recovering miscoded tech roles, which was its stated purpose. Corpus 36,811 → 31,696 |
| 2026-09-08 | AC-1.3 (REQ-1 → FROZEN v1.1) | Dedupe key changed from `title_normalized` (seniority-stripped) to `title_key` (seniority preserved) | Found while implementing phase 1a: the frozen wording merged "Senior Data Engineer" and "Data Engineer" at the same company and location into a single row. Those are distinct openings, and the loss would have been silent — the dedupe drop count would simply have been higher, with nothing to indicate real postings had been destroyed. `title_normalized` still strips seniority for AC-1.1 and AC-9.7 |
| 2026-09-08 | AC-5.1, AC-5.2 | Dense vector now embeds `title + description` (skills excluded); BM25 indexes `title + skills + description` untruncated. Token limit corrected 512 → 256 | Two problems found when caching the model: the stated 512-token maximum was wrong (`all-MiniLM-L6-v2` is 256), and AC-5.1 contradicted AC-9.12 by embedding skills that already have their own score components. Splitting the two indexes resolves both at no cost and gives a better division of labor — the dense vector stops duplicating what BM25 does well |
| 2026-09-08 | AC-4.1 | Résumé chunking simplified from heading-driven semantic sections to paragraph blocks | Scope trim against a 3-day deadline. Résumés are already visually blocked, so paragraph splits land on nearly the same boundaries; heading detection across arbitrary résumé formats is brittle and would have been REQ-4's largest source of edge cases for no measurable retrieval gain. Capability unchanged |
| 2026-09-08 | AC-13.1 | Retrieval evaluation switched from ~30 pre-labeled pairs to pooled judgment over the union of configurations' top-10s | Less manual labeling *and* a sounder method — pooling cannot under-credit a configuration for surfacing a relevant job that was never in a pre-chosen label set |
| 2026-09-08 | REQ-10 (removed), D11, AC-4.2, AC-9.10, AC-11.5, AC-13.3, AC-13.7 (removed), Non-Goals | Removed the hosted-LLM explanation layer entirely. D11 restated as "no LLM in the application; evidence is retrieved, not generated". Evidence display moved to AC-11.5, backed by REQ-4 retrieval and AC-4.2 character spans. Local HF generation noted as stretch | The co-design prep's "Practice with Agentic AI" was a prompt-engineering exercise, not a product requirement — the assignment asks an LLM's role to be explained only *if* one is used. Shipping it would have added an API key, a per-run cost, and a reproducibility barrier for prose, while the grounding it was supposed to provide already comes from semantic retrieval. No code had been written against REQ-10 |
| 2026-09-08 | AC-1.1, AC-1.2 | Pinned `TECH_CODES = {IT, ENG, ANLS, QA, SCI}` (33,502 postings, 27%); required the join to semi-join/dedupe; recorded that the gazetteer, not `TECH_CODES`, sets final corpus size | Schema audit measured the code distribution and found `job_skills.csv` fans out at 1.69 rows/job. Only ~55% of the code set has a software/data title, but AC-2.6 drops the rest for having no gazetteer skills |
| 2026-09-08 | AC-1.5 | `remote_allowed` null now means "not remote" rather than "unknown"; `Unknown` asserted empty for the LinkedIn corpus | Audit found the column is a sparse flag with exactly two values (1.0 / null), not a nullable boolean. Routing null to `Unknown` would put 87.7% of the corpus there and, under AC-6.6's pass-and-tag policy, turn the work-setting filter into a no-op |
| 2026-09-08 | AC-1.6a (new), AC-3.3 | `formatted_work_type` has 7 values, not 4. Retained Full-time/Contract/Part-time/Temporary/Internship; excluded Volunteer and Other from the corpus; profile page gains a fifth checkbox | The enum was written from assumption. Volunteer and Other are not roles this system's users search for |
| 2026-09-08 | AC-2.3, AC-2.7 | Reversed extraction precedence: `description` parsing is primary, `skills_desc` a supplement where present. Added `skills_desc` coverage to reported stats as a report finding | `skills_desc` is 98.0% null — present on ~2,500 of 123,849 postings. The original AC would have extracted skills for 2% of the corpus |
| 2026-09-08 | D12 | Corrected the rationale for dropping industry from "labels are inconsistent" to "scope, not data quality" | The audit refuted the original justification: `job_industries.csv` is 0% null with 422 clean industry names. A wrong justification in a document the report cites is worse than none |
| 2026-09-08 | D9, AC-1.10, AC-12.1, AC-13.4, Q3 | Processing engine changed from PySpark to DuckDB for offline ingestion and analytics; PySpark demoted to an optional engine-comparison in AC-13.4 | The corpus fits single-machine, so Spark added real setup cost (JDK 17 vs. the installed JDK 24, ~300MB package, JVM startup per run) for no performance gain. Report §5 lists SQL among acceptable technologies and grades the *justification*, not the tool. Measuring the crossover is a stronger result than asserting the choice. REQ-1 was still DRAFT |
| 2026-09-08 | AC-1.4 | Salary normalization now prefers the dataset's existing `normalized_salary` column, with the pay_period derivation as fallback; added per-path row counts to coverage stats | User confirmed `postings.csv` carries `normalized_salary`. Reimplementing normalization the dataset already did would be wasted work and a worse veracity story than reporting how much was pre-normalized. REQ-1 was still DRAFT, so this is an amendment, not a post-freeze change |
| 2026-09-08 | — | Initial draft | Written from the review of `docs/context/job-matching-application-plan.md` against `human-design.md` and `human-ai-codesign-prep.md`; decisions D1-D12 settled in discussion |
