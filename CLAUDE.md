# Project: Job Search Application — Human-AI Co-Design Challenge (CS 5542)

## Current branch
`human-ai-codesign` — all Stage 3 (Human-AI Co-Design) implementation happens here.

## Context files (read these first, in this order)
1. `docs/context/human-design.md` — my original Stage 1 human design (features, matching method, workflow, data table). This is the baseline design.
2. `docs/context/human-ai-codesign-prep.md` — my Stage 2 evaluation of AI's design, plus the high-level plan for Stage 3 (keep/modify/reject decisions, matching strategy, retrieval strategy).
3. `docs/context/job-matching-application-plan.md` (gitignored) — a draft Stage 3 plan generated in an earlier session. This is a starting point/input, NOT the final spec. Treat it the same way you'd treat AI Design output: mine it for good ideas, but critically evaluate it — it has not been vetted and should not be copied wholesale.
4. `docs/context/challenge-1-report-instructions.md` — the assignment instructions for the final report I must produce. Everything we build and document should map back to this.

## Stage 2 AI-generated code
Lives on branch `agent-exercise`. Do NOT merge it into `human-ai-codesign`. Use git to inspect and compare instead:
```bash
git show agent-exercise:path/to/file
git diff human-ai-codesign agent-exercise -- src/
```
Use this to trace exactly what changed between the raw AI output and our co-designed version — this comparison is required in the final report (Section 10, and throughout Section 4).

---

## This project follows spec-driven development

The authoritative spec lives at `docs/project-plan.md` (committed — this is the single source of truth for what the system must do, not just a narrative plan). A template for this file is at `docs/project-plan-template.md`.

### Spec structure

The spec is organized into numbered requirements, each with numbered acceptance criteria:

- `REQ-n`: a discrete requirement (a component, behavior, or rule)
- `AC-n.m`: a specific, testable acceptance criterion under that requirement

Code and tests reference these IDs directly. See `docs/project-plan-template.md` for the exact format.

### Spec change protocol (critical — follow this exactly)

If implementation reveals that the spec needs to change, **do not just change the code**:

1. Edit the relevant `REQ`/`AC` in `docs/project-plan.md` first.
2. Bump the version/status note at the top of the spec (e.g., `DRAFT` → `FROZEN v1.0` → `FROZEN v1.1`).
3. Log the reason for the change in `docs/report-notes/progress-log.md`, referencing the REQ/AC ID.
4. Only then implement the corresponding code change.

Code changes are only valid *because* a requirement changed — the spec is never updated to retroactively justify code that was already written.

### Tests derived from acceptance criteria

Tests live in `tests/`, one file per requirement area, e.g.:
```
tests/
├── test_req1_filtering.py
├── test_req2_manual_score.py
```

For every `REQ`, write test cases covering each of its `AC` IDs — before or immediately alongside implementing that piece, not after the fact as an afterthought. Reference the AC ID in the test name or docstring. A phase is not "done" until its AC-derived tests pass.

### Traceability

- Commit messages reference the REQ ID they implement, e.g.:
  `git commit -m "Implement REQ-1 hard filtering (AC-1.1, AC-1.2, AC-1.3)"`
- Progress log entries reference REQ/AC IDs so the final report (especially Section 4: kept/accepted/rejected, and Section 9: Results and Evaluation) can cite specific spec items directly.

---

## Workflow for this project

1. **Plan review (do this first)**: Critically review the draft `docs/context/job-matching-application-plan.md` against `human-design.md` and `human-ai-codesign-prep.md`. Check whether it's internally consistent, technically sound, and covers everything the report instructions require. Flag gaps, contradictions, or overreach. This review feeds into a **brand-new spec document**, not a patched version of the draft. Discuss findings with me before writing the spec.

2. **Write the spec**: Using `docs/project-plan-template.md` as the format, write the actual spec at `docs/project-plan.md`, covering:
   - input format (user profile fields, job posting fields)
   - UI / page layout and user flow
   - data sources and how data flows through the system
   - features (filtering, ranking, retrieval, matching, RAG, etc.)
   - output format (what the user sees, explanations, scores)
   - specific technical details: ranking/scoring formulas, matching method, retrieval strategy (BM25, embeddings, hybrid, reranking), technologies used and why
   - Each item should be expressed as a `REQ` with testable `AC`s, not just prose.
   - This can be written all at once or section-by-section, but each `REQ` section should be reviewed with me before being marked `FROZEN`.

3. **Per-requirement implementation loop**: For each `REQ` section, in order:
   - Freeze that section of the spec (get my sign-off)
   - Write tests from its `AC`s
   - Implement the code
   - Run tests until they pass
   - Commit, referencing the REQ ID
   - Update the progress log
   - Only then move to the next `REQ`

   Do not implement the whole system in one uninterrupted pass, and do not implement ahead of a frozen spec section.

4. **Git hygiene**: Commit and push at each REQ checkpoint (see above) with clear, traceable commit messages, so the repo history shows real, reproducible, spec-linked progression.

5. **Evaluation**: The report requires comparing methods (e.g., keyword vs. BM25 vs. semantic vs. hybrid retrieval, response time, scalability). Build a Jupyter notebook under `notebooks/` for this side-by-side evaluation with tables/figures. This is evaluation of *method quality*, distinct from the AC-derived tests, which verify *spec compliance*.

6. **Report tracking**: After every meaningful decision, change, or result — especially anything involving what was kept/accepted from AI vs. rejected/redesigned, or any spec change — append an entry to `docs/report-notes/progress-log.md` (gitignored, local only) using the format below. This log is what the final report gets written from.

7. **Project structure**: Keep a standard layout — don't let scripts or data sprawl into the repo root.
   ```
   job-search-challenge/
   ├── src/                        # application/pipeline source code
   ├── tests/                      # AC-derived tests, one file per REQ area
   ├── data/                       # raw/processed data (or access instructions if not committed)
   ├── notebooks/                  # evaluation & comparison notebooks
   ├── docs/
   │   ├── project-plan.md          # FINAL spec — committed, authoritative
   │   ├── project-plan-template.md # spec format template — committed
   │   ├── context/                 # gitignored reference material (includes draft plan)
   │   └── report-notes/            # gitignored progress log
   ├── requirements.txt
   ├── README.md
   └── CLAUDE.md
   ```

## Progress log entry format

Append to `docs/report-notes/progress-log.md`:

```markdown
### [Date] — [REQ ID / Phase / Topic]
- What we decided or built
- What was kept / accepted / rejected / redesigned relative to the AI-generated Stage 2 code or AI suggestions during this step
- Any spec changes made, and why
- Why (human judgment / rationale)
- Test results, if any
```

## Reminders
- Do not merge `agent-exercise` into this branch — it's reference-only for comparison.
- Every `REQ` should be traceable back to something in `docs/context/challenge-1-report-instructions.md`.
- Never implement past a spec section that hasn't been frozen.
- The final README.md (separate from this CLAUDE.md) must independently explain dataset, architecture, pipeline, analytics, execution, and results per the report instructions — write that as part of final polish.
