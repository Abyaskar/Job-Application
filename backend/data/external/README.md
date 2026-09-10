# External datasets (Job Intelligence Layer)

This project's Job Intelligence Layer is designed around two Kaggle datasets. **This sandbox
has no network access to kaggle.com**, so the real CSVs are not bundled here — this directory
documents exactly how to plug them in, and `app/services/dataset_loaders.py` runs against a
schema-matching **synthetic fallback** when the real files are absent, so the pipeline is fully
functional and evaluable today.

## 1. Job Skill Set

<https://www.kaggle.com/datasets/batuhanmutlu/job-skill-set>

**Used for:** expanding `data/skill_taxonomy.json` (new skill aliases/categories) and building
`data/role_title_skills.json` (job title → associated skills), which the Career Intent Layer
uses to expand a resolved role family's related titles into skills.

**Expected schema** (column-mapping is configurable — see `JOB_SKILL_SET_COLUMN_MAP` in
`dataset_loaders.py` if your downloaded copy uses different column names):

| Column | Meaning |
|---|---|
| `job_title` | Free-text job title |
| `skill` | A single skill/keyword associated with that title |

**To use the real dataset:**
```bash
# 1. Download job-skill-set.csv from Kaggle and place it here:
cp ~/Downloads/job_skill_set.csv /workspaces/Job-Application/backend/data/external/job_skill_set.csv

# 2. Run the build script (idempotent — safe to re-run)
python backend/scripts/build_taxonomy_from_job_skill_set.py
```
This merges newly-discovered skills into `skill_taxonomy.json` (skills already present by alias
are left untouched — no blind overwrite) and writes/updates `role_title_skills.json`.

## 2. Resume Data for Ranking

<https://www.kaggle.com/datasets/thejohnwick001/resume-data-for-ranking>

**Used for:** building realistic candidate/job relevance examples for evaluating resume-to-job
matching — i.e. augmenting `data/eval_labels.json`-style Precision@K/Recall@K/NDCG@K evaluation
with dataset-derived cases rather than only hand-labeled ones.

**Expected schema** (configurable — see `RESUME_RANKING_COLUMN_MAP`):

| Column | Meaning |
|---|---|
| `resume_text` | Full resume text |
| `job_title` / `job_description` | The job the resume was matched against |
| `match_label` | Binary or ordinal relevance signal (e.g. 1 = relevant/shortlisted, 0 = not) |

**To use the real dataset:**
```bash
cp ~/Downloads/resume_data_for_ranking.csv /workspaces/Job-Application/backend/data/external/resume_ranking.csv
python backend/scripts/build_eval_set_from_resume_ranking.py
```
This writes `data/eval_labels_dataset_augmented.json` in the same shape as `eval_labels.json`,
kept **separate** from the hand-labeled set rather than merged in-place, so the evaluation
endpoint can report hand-labeled and dataset-derived metrics distinctly (see README
"Evaluation" — merging datasets with different labeling methodologies into one metric silently
conflates two different notions of "relevant," which is exactly the "don't blindly merge
datasets" instruction this project is built to respect).

## Why synthetic fallback, not a scraped/redistributed copy

Redistributing a Kaggle dataset's data isn't something this repo should do regardless of network
access. The adapter pattern here (`DatasetLoader` interface, real CSV loader, schema-matching
synthetic generator) is the same one used for `EmbeddingProvider`/`LLMProvider` elsewhere in this
project — one interface, a documented real implementation, and a dependency-free default so the
project is reviewable without any external setup.
