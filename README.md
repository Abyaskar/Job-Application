# Job Application Strategy AI

**An intent-aware recommendation platform.** Upload a resume, type the role you actually want
("GenAI Engineer," "Data Analyst," "I want to move into data analytics"), and get a ranked,
explained, action-oriented shortlist of jobs — not a bare similarity score, and not a list
polluted with roles that merely *look* textually similar to your resume.

Built as a portfolio system demonstrating: Generative AI / RAG, vector search, hybrid
recommendation ranking, intent normalization (NLP), async Python (FastAPI), MongoDB, Redis,
evaluated retrieval (Precision@K / Recall@K / NDCG@K), and GCP-ready service design.

> **Demo mode note:** this project runs end-to-end with **zero external infrastructure and no
> API keys** — MongoDB, Redis, the embedding model, and the LLM explanation layer all have
> local, dependency-free implementations behind the same interfaces their production
> counterparts would use. See [Trade-offs](#trade-offs--why-things-are-built-this-way).

---

## Table of contents

1. [Quick start](#quick-start)
2. [Problem statement](#problem-statement)
3. [Three-layer architecture](#three-layer-architecture)
4. [Layer 1 — Career Intent Layer](#layer-1--career-intent-layer)
5. [Layer 2 — Job Intelligence Layer](#layer-2--job-intelligence-layer-dataset-integration)
6. [Layer 3 — Strategy Ranking Layer](#layer-3--strategy-ranking-layer)
7. [RAG flow — Why Apply / Why Not Apply](#rag-flow--why-apply--why-not-apply)
8. [API design](#api-design)
9. [Evaluation](#evaluation)
10. [Trade-offs — why things are built this way](#trade-offs--why-things-are-built-this-way)
11. [Failure modes](#failure-modes)
12. [Security & privacy](#security--privacy)
13. [GCP deployment](#gcp-deployment)
14. [Interview explanation guide](#interview-explanation-guide)
15. [Resume bullets](#resume-bullets-backed-by-measured-results)
16. [Project structure](#project-structure)

---

## Quick start

### Option A — Demo mode (no Docker, no database, 60 seconds)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

On startup the app seeds itself with 7 sample resumes and 12 sample job descriptions, fits the
embedding space, and builds the vector index and skill/role taxonomies. Open
**http://localhost:8000/docs** for interactive API docs.

```bash
# In another terminal
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** → Dashboard → type a career intent (e.g. "GenAI Engineer") in
the intent bar at the top.

### Option B — Full stack with real MongoDB + Redis (Docker)

```bash
docker compose up --build
```

### Run the tests

```bash
cd backend && pytest -v   # 26 tests: extraction, ranking, intent normalization, gating, evaluation metrics, full API integration
```

### Run the evaluation

```bash
python evaluation/run_eval.py --base-url http://localhost:8000
# Also try, with the backend running:
curl "http://localhost:8000/api/v1/evaluation/intent-impact" | python3 -m json.tool
```

### Postman

Import `postman/JobApplicationStrategyAI.postman_collection.json`. The `candidate_id` variable
defaults to `cand_pivot_analyst_01` — a data-analyst resume with a stated "GenAI Engineer"
intent, specifically so the collection demonstrates intent-aware ranking out of the box (see
[Evaluation](#evaluation)).

---

## Problem statement

Candidates applying to jobs at scale have no principled way to decide *which* jobs to
prioritize, and generic resume-vs-JD similarity actively misleads them: a resume that happens to
share vocabulary with a job ("Python," "distributed systems," "cloud" appear in both a backend
engineer JD and a data engineer JD) can rank that job highly even when it's not what the
candidate wants next. **Similarity is not intent.** This system treats "which job should I apply
to next" as three distinct problems stacked on top of each other:

1. **Understand what the candidate wants** (Career Intent Layer) — not inferred from the resume
   alone, but stated explicitly and normalized against a role taxonomy.
2. **Understand the job market** (Job Intelligence Layer) — structured requirements, skill
   taxonomy, and title↔skill relationships, informed by real-world dataset patterns.
3. **Rank, explain, and recommend an action** (Strategy Ranking Layer) — combine intent
   alignment with semantic fit, hard-skill coverage, experience, education, and location into a
   deterministic score, then explain it with grounded evidence and a concrete next action.

---

## Three-layer architecture

```
+------------------------------------------------------------------------------------+
|  LAYER 1 -- CAREER INTENT LAYER                                                     |
|  Free-text role input ("GenAI Engineer") -> normalize_intent() -> IntentProfile     |
|  (role_family, canonical_title, related_titles, intent_skills, intent_keywords,     |
|  match_method, confidence). Cascade: exact alias -> fuzzy -> keyword overlap ->     |
|  embedding fallback. Stored per-candidate, independent of resume re-uploads.        |
+-------------------------------------------+-----------------------------------------+
                                             | IntentProfile
                                             v
+------------------------------------------------------------------------------------+
|  LAYER 2 -- JOB INTELLIGENCE LAYER                                                  |
|  Job ingestion -> rule-based requirement extraction (skills, experience, education, |
|  location) against a skill taxonomy expanded from the "Job Skill Set" dataset.      |
|  Embedding fit + vector index over the job corpus. role_title_skills.json (dataset- |
|  derived) enriches intent_skills for a resolved role family's canonical title.      |
+-------------------------------------------+-----------------------------------------+
                                             | ParsedJob + ExtractedRequirements + embeddings
                                             v
+------------------------------------------------------------------------------------+
|  LAYER 3 -- STRATEGY RANKING LAYER                                                  |
|  score_candidate_job(resume, job, intent) -> ScoreBreakdown (deterministic, no LLM):|
|  intent_alignment . semantic_similarity . hard_skill_match . experience_match .     |
|  education_match . location_match -> weighted final_score, with hard intent GATING  |
|  for clearly off-intent jobs. -> recommend_action() -> apply_now / tailor_resume_first|
|  / build_missing_evidence / low_priority. -> RAG explainer (LLM, evidence-grounded,  |
|  ranking-independent) generates Why Apply / Why Not Apply from retrieved evidence.   |
+------------------------------------------------------------------------------------+
```

**What's new vs. the previous (pre-intent) architecture:** the existing FastAPI/MongoDB/Redis
service skeleton, embedding provider abstraction, hybrid ranking core, RAG explanation pipeline,
and evaluation harness are all **preserved** — this upgrade adds Layer 1 as a new service +
repository + router, extends Layer 3's scoring function with one new deterministic component
(`intent_alignment`) and a gating rule, and extends the RAG explainer's output shape
(`why_apply`/`why_not_apply`) without changing its grounding/generation architecture. Nothing
about the Mongo/Redis dual-adapter pattern, the pluggable embedding/LLM providers, or the
ranking/generation separation changed — see the original design rationale still documented
inline in `ranking.py` and `rag.py`.

---

## Layer 1 — Career Intent Layer

**Where it lives:** `app/services/intent.py`, `data/role_taxonomy.json`,
`app/api/routers/intent.py`, `IntentRepository`.

### Normalization cascade

`normalize_intent(candidate_id, free_text)` resolves free text to a role family via four
increasingly fuzzy strategies, cheapest/most-precise first:

| Stage | Method | Example |
|---|---|---|
| 1 | Exact alias match against `role_taxonomy.json` | `"genai engineer"` → `genai_engineer` |
| 2 | Fuzzy match (rapidfuzz token-sort ratio, threshold 78) | `"Data Analist"` (typo) → `data_analyst` |
| 3 | Keyword overlap against each family's keywords/skills | `"I want to work with large language models and RAG"` → `genai_engineer` |
| 4 | Embedding similarity (same provider ranking uses) | catches intent text sharing no taxonomy vocabulary at all |

Deliberately **not an LLM call**: this needs to run synchronously and cheaply on every intent
submission, be deterministic enough to unit test (`tests/test_intent.py`), and cost nothing per
request for what is fundamentally classification into a bounded, known taxonomy. An LLM-based
normalizer is the natural extension for genuinely open-ended text the taxonomy can't resolve
(`IntentMatchMethod.UNRESOLVED`) — documented as a follow-up rather than built, to keep this
layer as evaluable as ranking is.

### What a resolved intent contains

```json
{
  "role_family": "genai_engineer",
  "canonical_title": "GenAI Engineer",
  "related_titles": ["AI Engineer", "Machine Learning Engineer", "NLP Engineer", "..."],
  "intent_skills": ["fastapi", "llm", "machine_learning", "nlp", "prompt_engineering", "python", "rag", "vector_search"],
  "intent_keywords": ["generative ai", "large language model", "retrieval augmented generation", "..."],
  "match_method": "exact_alias",
  "confidence": 1.0
}
```

`intent_skills` is enriched from the dataset-derived `role_title_skills.json` (Layer 2) for the
resolved family's *canonical title only* — not its related titles. That scoping decision fixed a
real bug found while building this: several related titles are legitimately shared across more
than one role family (e.g. "Full-Stack Engineer, AI Products" is a related title for both
`genai_engineer` and `frontend_engineer`), so enriching from every related title leaked one
family's skills into another's intent profile via that shared title. Scoping to the canonical
title only avoids the leak.

### Storage — separate from the resume

Intent is stored in its own `intents` collection (`IntentRepository`), keyed by `candidate_id`,
not as a field on `ParsedResume`. A candidate can explore "GenAI Engineer" this week and "ML
Engineer" next week against the same uploaded resume without re-uploading anything — intent is
an independently-updatable signal layered on top of the resume, not baked into it.

### API

- `POST /api/v1/intent` `{candidate_id, free_text}` → resolves and **persists** the intent.
- `GET /api/v1/intent/preview?text=...` → resolves **without persisting**, used by the frontend
  for a live "did you mean: GenAI Engineer?" preview before the candidate confirms.
- `GET /api/v1/intent/{candidate_id}` → fetch the candidate's currently active intent.
- `POST /api/v1/recommendations/rank` takes `use_intent: bool` (default `true`) — set `false` to
  see pure resume-based ranking with intent disabled, for direct comparison (this is exactly
  what `/evaluation/intent-impact` automates across the whole eval set).

---

## Layer 2 — Job Intelligence Layer (dataset integration)

Two Kaggle datasets inform this layer. **Full disclosure up front:** this sandbox has no network
access to `kaggle.com` (verified directly — outbound requests are rejected), so neither raw CSV
is bundled in this repo. Rather than fake the numbers or skip the requirement, this layer is
built the same way the embedding/LLM providers are: a real adapter with a documented schema and
column-mapping, and a schema-matching **synthetic fallback** so the full pipeline runs and is
testable today, with an exact, scripted path to plug in the real data. See
`backend/data/external/README.md` for the full writeup; summary below.

### Dataset 1 — Job Skill Set

<https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set>

**Used for:** expanding `skill_taxonomy.json` and building `role_title_skills.json` (job title →
associated skills), which Layer 1 uses to enrich a resolved intent's skill list.

**Expected schema:** `job_title`, `skill` columns (configurable via `JOB_SKILL_SET_COLUMN_MAP`
in `app/services/dataset_loaders.py`).

**To use the real dataset:**
```bash
cp ~/Downloads/job_skill_set.csv backend/data/external/job_skill_set.csv
python backend/scripts/build_taxonomy_from_job_skill_set.py
```
The script is **additive and idempotent**: it resolves each dataset skill string against the
*existing* taxonomy's aliases first (via the same `alias_to_canonical()` lookup extraction uses)
and only creates a new taxonomy entry for genuinely new terms — it does not blindly overwrite or
duplicate existing entries. Run today (against the synthetic fallback, since the real file isn't
present) it reports: `Loaded 248 pairs [SYNTHETIC fallback] -> skill_taxonomy.json: 55 total
skills (0 newly added) -> role_title_skills.json: 23 job titles mapped` — zero new skills because
the synthetic generator is deliberately derived *from* the existing taxonomy (so it's internally
consistent), not an independent source; the real dataset would be expected to surface genuinely
new terms.

### Dataset 2 — Resume Data for Ranking

<https://www.kaggle.com/datasets/thejohnwick001/resume-data-for-ranking>

**Used for:** building additional resume↔job relevance cases for the evaluation harness.

**Expected schema:** `resume_text`, `job_title`, `job_description`, `match_label` (configurable
via `RESUME_RANKING_COLUMN_MAP`).

**To use the real dataset:**
```bash
cp ~/Downloads/resume_data_for_ranking.csv backend/data/external/resume_ranking.csv
python backend/scripts/build_eval_set_from_resume_ranking.py
```
This writes `data/eval_labels_dataset_augmented.json` — deliberately **kept separate** from the
hand-labeled `data/eval_labels.json` rather than merged in-place. **This is the "don't blindly
merge datasets" requirement in concrete form:** the hand-labeled set encodes "would a career
counselor prioritize this," while this dataset's `match_label` encodes whatever that dataset's
own labeling methodology was — silently merging both into one Precision@K number would conflate
two different definitions of "relevant." The evaluation endpoint and README report them
distinctly.

### Why synthetic fallback, not a scraped copy

Redistributing a Kaggle dataset's contents isn't appropriate regardless of network access. The
adapter pattern (`DatasetLoader`-shaped functions, real CSV path, schema-matching synthetic
generator) is the same one used for `EmbeddingProvider`/`LLMProvider` elsewhere in this project —
one interface, a documented real implementation, a dependency-free default.

---

## Layer 3 — Strategy Ranking Layer

### Ranking formula

```
final_score = w_intent x intent_alignment
            + w_sem    x semantic_similarity
            + w_skill  x hard_skill_match
            + w_exp    x experience_match
            + w_edu    x education_match
            + w_loc    x location_match
```

Default weights (configurable via env vars):

| Component | Weight | What it captures |
|---|---|---|
| Intent alignment | 0.20 | Deterministic title/domain match against the candidate's resolved role family |
| Semantic similarity | 0.25 | Cosine similarity between resume and JD embeddings |
| Hard skill match | 0.25 | `matched_required / total_required` against the skill taxonomy |
| Experience match | 0.12 | Candidate years vs. required years (floored, not zeroed, when under) |
| Education match | 0.08 | Candidate's degree level vs. required level |
| Location match | 0.10 | Remote/preferred-location match |

### Intent gating — the concrete mechanism, not just a weighted signal

Everything above is a **soft** weighted signal — except intent, deliberately. When a candidate
has a resolved intent and a job's `intent_alignment` score is ≤ 0.2 (title/domain shares
essentially nothing with the stated intent), the job is **gated**: its final score is hard-capped
at 0.35 regardless of how well the resume text otherwise matches, and `ScoreBreakdown.intent_gated
= true` is set so the UI can show *why* it was deprioritized. This is what makes "don't recommend
unrelated roles just because of generic resume similarity" a real mechanism rather than a
marketing sentence — see [Evaluation](#evaluation) for a measured before/after example.

Retrieval is also intent-aware, not just scoring: `_expand_shortlist_for_intent()` backfills the
vector-search shortlist with jobs that title-match the stated intent even if the resume text
doesn't yet strongly resemble them (e.g. a data analyst's resume pivoting toward GenAI won't
contain much GenAI vocabulary yet) — so intent-relevant jobs reach the *ranking* layer to be
judged, rather than being silently excluded at *retrieval* time by content-similarity alone.

### Ranking stays deterministic and separate from the LLM

`ranking.py` — including `intent_alignment` and gating — has zero LLM calls. `rag.py` only
*explains* an already-computed score/gap; it never influences it. This means: (1) evaluation
metrics are reproducible regardless of LLM provider/availability, (2) an LLM outage degrades
explanation fluency, never ranking correctness, and (3) unit-testing ranking never requires
mocking an LLM (see `tests/test_intent.py::test_intent_gating_caps_score_for_off_intent_job`).

### Action recommendation

| Condition | Action |
|---|---|
| Intent-gated | `low_priority` (forced, regardless of other scores) |
| `final_score >= 0.75` and `skill_coverage >= 0.8` | `apply_now` |
| `final_score >= 0.55` and coverage `< 0.8` and `<=2` missing required skills | `tailor_resume_first` |
| `final_score >= 0.4` and `>2` missing required skills | `build_missing_evidence` |
| otherwise | `low_priority` |

---

## RAG flow — Why Apply / Why Not Apply

```
   ScoreBreakdown + SkillGap + IntentProfile (already computed -- RAG never touches these)
                    |
                    v
        retrieve_evidence()  -- scoped to matched/missing focus skills (max 4+4) + a
                                 dedicated intent-alignment evidence snippet naming the
                                 stated intent and the job's actual title
                    |
                    v
        LLMProvider.generate_explanation()
          LocalTemplateLLMProvider (default): every sentence constructed directly from
          computed scores/evidence -- intent-gated jobs get an explicit "deprioritized
          for intent, not resume content" line in why_not_apply; strong intent matches
          get a "directly matches your stated intent" line in why_apply. No free-text
          generation step exists to invent anything.
                    |
                    v
        _groundedness_filter()  -- drops any why_apply/why_not_apply/reasons item sharing
                                    no vocabulary with retrieved evidence, per-list
                    |
                    v
              Explanation(summary, reasons[], why_apply[], why_not_apply[], evidence[], grounded, confidence)
```

Example (real output, `cand_pivot_analyst_01` / "Data Analyst" job, intent = "GenAI Engineer"):

```
why_apply:
  - "Your resume shows direct evidence of data visualization, python, sql, statistics,
     which this role lists as required."

why_not_apply:
  - "This role's title/domain doesn't align with your stated intent of 'GenAI Engineer' —
     it was deprioritized specifically for that reason, not because of your resume content."
  - "No evidence found for required skill(s): excel."
  - "This job's location doesn't match your stated preferences."
```

Note this job had 85% semantic similarity and 80% skill coverage — a generic-similarity system
would call this a strong match. The intent-aware explanation correctly tells the candidate why it
isn't the priority anyway.

---

## API design

Base path: `/api/v1`. Full interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health`, `/health/live`, `/health/ready` | Liveness/readiness probes |
| `POST` | `/candidates/resume` | Ingest resume, run structured extraction |
| `GET` | `/candidates/{id}/resume` | Fetch parsed resume |
| **`POST`** | **`/intent`** | **Normalize + persist career intent (Layer 1)** |
| **`GET`** | **`/intent/preview?text=`** | **Preview normalization without persisting** |
| **`GET`** | **`/intent/{candidate_id}`** | **Fetch candidate's active intent** |
| `POST` | `/jobs`, `/jobs/bulk` | Ingest job description(s) |
| `POST` | `/jobs/reindex` | Rebuild embedding fit + vector index |
| `GET` | `/jobs/{id}`, `/jobs` | Fetch / list jobs |
| `POST` | `/recommendations/rank` | Rank jobs for a candidate — `use_intent: bool` toggles Layer 1 |
| `POST` | `/recommendations/compare` | keyword/vector/hybrid side-by-side, intent-aware |
| `GET` | `/recommendations/{candidate_id}`, `/{candidate_id}/{job_id}` | Fetch persisted recommendation(s) |
| `POST` | `/feedback` | Record accept/reject signal |
| `GET` | `/evaluation/run?use_intent=` | Precision/Recall/NDCG for keyword/vector/hybrid |
| **`GET`** | **`/evaluation/intent-impact`** | **Hybrid-with-intent vs. hybrid-without-intent, isolated** |
| `GET` | `/evaluation/cache-stats`, `/evaluation/summary` | Cache hit rate, acceptance rate |

---

## Evaluation

### Methodology

`backend/data/eval_labels.json` hand-labels, for 7 candidates × 12 sample jobs, which jobs a
career counselor would prioritize — **respecting stated intent over generic similarity** where an
`intended_role` is set. One case, `cand_pivot_analyst_01`, is a deliberate stress test: a
data-analyst resume with a stated intent to move into GenAI engineering, used specifically to
measure whether the Career Intent Layer correctly overrides the resume's generic best textual
match. This is a small, manually curated set — see [Failure modes](#failure-modes) for its
limits and the dataset-augmented scaling path (Layer 2).

### Keyword vs. vector vs. hybrid (live, `python evaluation/run_eval.py`)

| Mode | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | NDCG@1 | NDCG@3 | NDCG@5 | Avg latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Keyword | 0.714 | 0.429 | 0.314 | 0.476 | 0.738 | 0.929 | 0.714 | 0.716 | 0.806 | 4.02 |
| Vector  | 1.000 | 0.381 | 0.314 | 0.691 | 0.762 | 0.929 | 1.000 | 0.814 | 0.906 | 4.02 |
| Hybrid  | 0.857 | 0.524 | 0.314 | 0.548 | 0.929 | 0.929 | 0.857 | **0.880** | 0.880 | 3.98 |

Hybrid wins on P@3, R@3, and NDCG@3 — the metrics that matter most for "which few jobs should I
look at first" — by combining keyword-match precision with embedding-similarity recall, same
conclusion as before this upgrade, now computed on the enlarged 7-candidate set with intent
applied by default (`use_intent=true`).

### Intent impact (isolated) — `GET /evaluation/intent-impact`

Runs every case that has a stated `intended_role` through **hybrid ranking twice**: once with the
Career Intent Layer applied, once with it fully disabled.

| Variant | P@3 | R@3 | NDCG@3 | Avg latency (ms) |
|---|---|---|---|---|
| Without intent | 0.429 | 0.738 | 0.760 | 7.97 |
| **With intent** | **0.524** | **0.929** | **0.881** | 4.42 |

**NDCG@3 improves 0.76 → 0.88, Recall@3 improves 0.74 → 0.93** with the Career Intent Layer
applied. Concretely, for `cand_pivot_analyst_01` (data-analyst resume, GenAI intent):
`intended_role="GenAI Engineer"`, and *without* intent the top-ranked job is **"Data Analyst"**
(score 0.804, `apply_now`) — the generic-similarity trap this layer exists to fix. *With* intent
applied, "Data Analyst" is gated out of contention (`intent_gated: true`, final score capped at
0.35, action forced to `low_priority`) and the top result becomes **"Full-Stack Engineer, AI
Products"** — a role actually aligned with the stated goal.

This isn't a universal free lunch, and the numbers above are reported honestly including the one
case where it isn't: `cand_nlp_researcher_01` (intent: "NLP Research Scientist") had "AI Engineer
- Generative AI Platform" gated out even though it was hand-labeled relevant for them — a PhD
researcher's adjacent-but-differently-titled opportunity got suppressed by strict title-based
gating. The *aggregate* metrics above are still net-positive across all 7 cases, but this is a
real, quantified precision/recall trade-off, not a strictly free improvement — documented further
in [Failure modes](#failure-modes).

### Cache hit rate

`GET /evaluation/cache-stats` returns live hit/miss counts from the result cache (now
intent-aware — the cache key includes the intent signature so an intent change correctly
invalidates cached rankings rather than serving stale ones).

---

## Trade-offs — why things are built this way

| Decision | Why | Production upgrade path |
|---|---|---|
| Rule-based/taxonomy intent normalization instead of an LLM call | Runs synchronously on every intent submission; needs to be fast, deterministic, and testable without mocking an LLM | Add an LLM-based normalizer specifically for `IntentMatchMethod.UNRESOLVED` cases the taxonomy cascade can't resolve |
| Intent gating is a hard cap, not just another weight | A weighted-only approach lets a strong enough resume match still overpower a clearly wrong-domain job; a stated intent of "GenAI Engineer" should make "Data Analyst" a `low_priority`, not merely a slightly-lower-scored option | Could be made a learned threshold once enough feedback data on accepted/rejected off-intent recommendations exists |
| Synthetic fallback for both Kaggle datasets | No network access to kaggle.com in this sandbox (verified) | Drop the real CSV into `data/external/` and re-run the two build scripts — zero code changes needed |
| `role_title_skills.json` enrichment scoped to canonical title only | Related titles are legitimately shared across role families in `role_taxonomy.json`; enriching from all of them leaked cross-family skills — found by actually running the pipeline, not a hypothetical concern | If related-title enrichment is wanted later, de-duplicate shared titles across families first |
| Local TF-IDF+SVD embeddings, template-based RAG explanations, brute-force vector index | Unchanged from the original architecture — see inline rationale in `embeddings.py`, `rag.py`, `vector_index.py` | Unchanged — `EMBEDDING_PROVIDER=vertex_ai`, `LLM_PROVIDER=anthropic`, Atlas/Vertex Vector Search |

---

## Failure modes

- **Intent gating can suppress a legitimately relevant, differently-titled job.** Quantified in
  [Evaluation](#evaluation): `cand_nlp_researcher_01`'s hand-labeled-relevant "AI Engineer -
  Generative AI Platform" got gated out under a strict "NLP Research Scientist" intent. Title-
  based alignment is deliberately simple and inspectable, which means it's also occasionally
  blunt. Mitigation: the 0.2 gate threshold and 0.35 cap are conservative rather than aggressive
  (a soft demotion, not exclusion — the job still appears, just deprioritized), and both are env-
  configurable.
- **Intent normalization can resolve to the wrong family on ambiguous input.** "Analyst" alone
  could plausibly mean data or business analyst; the taxonomy's alias/keyword lists are curated
  to minimize this but aren't exhaustive. The `/intent/preview` endpoint exists specifically so
  the frontend can show a "did you mean X?" confirmation step before an intent is applied.
- **Dataset-derived taxonomy/eval augmentation currently runs against synthetic fallback data**,
  not the real Kaggle datasets (no network access in this sandbox) — see
  [Job Intelligence Layer](#layer-2--job-intelligence-layer-dataset-integration) for the exact,
  scripted path to swap in the real files, which requires no code changes.
- **Small, hand-labeled eval set** (7 candidates × 12 jobs) — enough to demonstrate methodology
  and catch large regressions, not enough for strong statistical confidence on small ranking
  changes. `data/eval_labels_dataset_augmented.json` (Layer 2) is the documented scaling path.
- Other pre-existing failure modes (thin resumes, taxonomy coverage gaps, cold-start embeddings,
  rule-based action thresholds not yet learned from feedback, provider outage fallback) are
  unchanged from the original architecture and still apply — see inline comments in
  `ranking.py`, `rag.py`, and `services/seed.py`.

---

## Security & privacy

Unchanged from the original architecture (no auth wired into routers — flagged explicitly as a
gap; rate limiting via Redis fixed-window; CORS allow-listed; groundedness filtering bounds what
the LLM layer can assert). One addition: **intent text is free-form user input persisted per
candidate** — it should be included in any future data-retention/deletion policy alongside resume
text, and is not currently redacted or encrypted at rest in demo mode (flagged, not implemented,
same as resume PII).

---

## GCP deployment

Unchanged — see the original mapping (Cloud Run for backend/frontend, Memorystore for Redis,
Atlas/Vertex AI Vector Search for the vector index at scale, Vertex AI/Anthropic for embeddings
and LLM explanations, Secret Manager for credentials, Cloud Build for CI/CD). The Career Intent
Layer adds no new infrastructure requirements — it's pure application logic plus one new Mongo
collection.

---

## Interview explanation guide

**"How is this different from just adding a filter for job title?"**
A hard title filter would exclude jobs whose title doesn't exactly match — too brittle (a
candidate wanting "GenAI Engineer" should still see "AI Engineer" or "LLM Engineer" roles). The
Career Intent Layer resolves free text to a *role family* with a curated set of related titles,
and gating operates on alignment *score* (title + domain token overlap), not exact string
matching, so adjacent titles still surface while genuinely unrelated ones get suppressed.

**"Why gate instead of just weighting intent like the other signals?"**
Tried and measured: a pure-weight approach (intent as just another 20%-weighted signal) still let
strong skill/semantic matches on off-intent jobs outrank on-intent ones in early testing. Gating
is the mechanism that actually delivers "don't recommend unrelated roles because of generic
similarity" — see the Data Analyst / GenAI Engineer example in Evaluation, where 85% semantic
similarity and 80% skill coverage were *not* enough to prevent gating.

**"Where does the LLM actually touch this?"**
Nowhere in scoring, gating, or the action decision — all deterministic. The LLM (or its local
template stand-in) only turns already-computed scores/evidence into readable Why Apply / Why Not
Apply sentences, through a groundedness filter that drops anything not traceable to retrieved
evidence.

**"You don't have the real Kaggle datasets — doesn't that undermine the Job Intelligence Layer?"**
The *pipeline* is real and tested: real CSV loader with documented schema, idempotent merge
script, synthetic fallback for reviewability without setup. What's synthetic is the *data*, not
the *code path* — dropping the real CSVs into `data/external/` and re-running two scripts is the
entire integration step, by design.

**"What would you build next if you had another week?"**
Scale the eval set using `eval_labels_dataset_augmented.json` once the real Resume Data for
Ranking dataset is available; add an LLM-based intent normalizer for the `UNRESOLVED` cascade
tail; replace the rule-based action thresholds with a re-ranker trained on accumulated feedback
data, evaluated against the current thresholds before replacing them.

---

## Resume bullets (backed by measured results)

- Designed and shipped a Career Intent Layer that normalizes free-text career goals into a role
  taxonomy via a 4-stage resolution cascade (exact/fuzzy/keyword/embedding), and used it to gate
  job recommendations by intent alignment — **improving NDCG@3 by 16% (0.76→0.88) and Recall@3 by
  26% (0.74→0.93)** on a labeled evaluation set isolating the effect of intent-aware vs.
  intent-naive ranking.
- Built a dataset-integration layer for two external (Kaggle) datasets with a documented schema,
  configurable column-mapping, and idempotent merge scripts, kept structurally separate from
  hand-labeled evaluation data to avoid conflating different labeling methodologies.
- Extended a hybrid ranking model (semantic + skill + experience + education + location) with a
  deterministic intent-alignment component and hard gating for off-intent roles, keeping ranking
  fully separate from the LLM explanation layer so evaluation metrics remain reproducible
  regardless of LLM availability.
- Added grounded "Why Apply" / "Why Not Apply" explanation sections through a RAG pipeline with
  scoped evidence retrieval and a post-generation groundedness filter, surfaced in a Next.js
  dashboard with a live intent-normalization preview.

---

## Project structure

```
job-application-strategy-ai/
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── intent.py              Career Intent Layer (Layer 1)
│   │   │   ├── dataset_loaders.py     Job Intelligence Layer dataset adapters (Layer 2)
│   │   │   ├── ranking.py             Strategy Ranking Layer (Layer 3) — intent-aware
│   │   │   ├── rag.py                 Why Apply / Why Not Apply grounded explanations
│   │   │   ├── recommender.py         Orchestration — intent-aware retrieval + ranking
│   │   │   └── ... (embeddings, vector_index, extraction, evaluation, seed, taxonomy)
│   │   ├── api/routers/
│   │   │   ├── intent.py              POST /intent, GET /intent/preview, GET /intent/{id}
│   │   │   └── ... (candidates, jobs, recommendations, feedback, evaluation, health)
│   │   ├── models/schemas.py          IntentProfile, extended ScoreBreakdown/Explanation
│   │   └── repositories/repositories.py   IntentRepository + existing repos
│   ├── data/
│   │   ├── role_taxonomy.json         11 role families for intent resolution
│   │   ├── role_title_skills.json     dataset-derived (Job Skill Set), built by script
│   │   ├── eval_labels.json           hand-labeled + intended_role per case
│   │   ├── eval_labels_dataset_augmented.json   dataset-derived (Resume Data for Ranking)
│   │   ├── external/README.md         exact real-dataset integration instructions
│   │   └── skill_taxonomy.json        extended with GenAI/business-analyst vocabulary
│   ├── scripts/
│   │   ├── build_taxonomy_from_job_skill_set.py
│   │   └── build_eval_set_from_resume_ranking.py
│   └── tests/test_intent.py           11 tests: normalization, alignment, gating
├── frontend/
│   ├── components/IntentBar.tsx       career intent input + live preview
│   └── app/...                        dashboard (intent-aware), recommendation detail
│                                       (Why Apply / Why Not Apply sections)
├── evaluation/
│   ├── run_eval.py
│   └── results.md
├── postman/JobApplicationStrategyAI.postman_collection.json   incl. /intent/* endpoints
├── docker-compose.yml
└── README.md
```
