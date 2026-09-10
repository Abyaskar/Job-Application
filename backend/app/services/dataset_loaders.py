"""
Job Intelligence Layer — dataset adapters.

Two Kaggle datasets inform this layer (see `data/external/README.md` for
full documentation and the exact expected schema of each):

  1. "Job Skill Set" (batuhanmutlu/job-skill-set) — job_title -> skill pairs,
     used to expand the skill taxonomy and build a job-title -> skills prior
     that the Career Intent Layer uses when resolving related titles.

  2. "Resume Data for Ranking" (thejohnwick001/resume-data-for-ranking) —
     resume/job/relevance triples, used to build additional evaluation
     cases for the ranking evaluation harness.

This sandbox has no network access to kaggle.com (confirmed: `kaggle.com`
is not in the allowed-domains list this environment can reach), so neither
raw CSV ships in this repo. Each loader below:

  - Reads the real CSV from `data/external/` if present, using a
    documented, overridable column-name mapping.
  - Falls back to a small, schema-matching synthetic generator if the
    file isn't present, so every downstream consumer remains testable.

The real Resume Data for Ranking dataset uses `matched_score` as a
continuous relevance score rather than a binary label. For the evaluation
adapter, scores >= 0.70 are treated as relevant (label 1), while scores
below 0.70 are treated as not relevant (label 0).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger


logger = get_logger("services.dataset_loaders")

EXTERNAL_DIR = Path(__file__).resolve().parents[2] / "data" / "external"

JOB_SKILL_SET_PATH = EXTERNAL_DIR / "job_skill_set.csv"
RESUME_RANKING_PATH = EXTERNAL_DIR / "resume_data_for_ranking.csv"


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

# Column-name overrides: if your downloaded CSV uses different headers,
# change these rather than the loader logic.
JOB_SKILL_SET_COLUMN_MAP = {
    "job_title": "job_title",
    "skill": "skill",
}

# Actual columns in:
# data/external/resume_data_for_ranking.csv
RESUME_RANKING_COLUMN_MAP = {
    "resume_text": "career_objective",
    "job_title": "job_position_name",
    "job_description": "responsibilities.1",
    "match_label": "matched_score",
}


@dataclass(frozen=True)
class JobSkillPair:
    job_title: str
    skill_text: str


@dataclass(frozen=True)
class ResumeRankingRow:
    resume_text: str
    job_title: str
    job_description: str
    match_label: int  # 1 = relevant, 0 = not


# ---------------------------------------------------------------------------
# Job Skill Set
# ---------------------------------------------------------------------------

def load_job_skill_set() -> tuple[list[JobSkillPair], bool]:
    """Returns (pairs, is_synthetic)."""

    if JOB_SKILL_SET_PATH.exists():
        pairs = []

        with open(
            JOB_SKILL_SET_PATH,
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)

            title_col = JOB_SKILL_SET_COLUMN_MAP["job_title"]
            skill_col = JOB_SKILL_SET_COLUMN_MAP["skill"]

            for row in reader:
                if (
                    title_col in row
                    and skill_col in row
                    and row[title_col]
                    and row[skill_col]
                ):
                    pairs.append(
                        JobSkillPair(
                            row[title_col].strip(),
                            row[skill_col].strip(),
                        )
                    )

        logger.info(
            "dataset.job_skill_set.loaded_real",
            n_rows=len(pairs),
        )

        return pairs, False

    logger.warning(
        "dataset.job_skill_set.synthetic_fallback",
        detail=(
            "Real dataset not found at "
            "data/external/job_skill_set.csv — using synthetic "
            "schema-matching data. See data/external/README.md "
            "to plug in the real Kaggle dataset."
        ),
    )

    return _synthetic_job_skill_set(), True


def _synthetic_job_skill_set() -> list[JobSkillPair]:
    """Representative, schema-matching stand-in.

    Covers the same role families this project's role taxonomy already
    defines, generated from that taxonomy so it's internally consistent
    rather than arbitrary.

    This is NOT a substitute for the real dataset's coverage or scale.
    It only exists to exercise the loader/merge pipeline.
    """

    from app.services.taxonomy import (
        canonical_label,
        load_role_taxonomy,
    )

    pairs = []

    for meta in load_role_taxonomy().values():
        titles = [
            meta["canonical_title"],
            *meta.get("related_titles", []),
        ]

        for title in titles:
            for skill_id in meta.get("core_skills", []):
                pairs.append(
                    JobSkillPair(
                        title,
                        canonical_label(skill_id),
                    )
                )

    return pairs


# ---------------------------------------------------------------------------
# Resume Data for Ranking
# ---------------------------------------------------------------------------

def load_resume_ranking_dataset() -> tuple[list[ResumeRankingRow], bool]:
    """Returns (rows, is_synthetic).

    The real dataset contains a continuous `matched_score` rather than
    a binary relevance label.

    Evaluation rule:
        matched_score >= 0.70 -> label 1
        matched_score <  0.70 -> label 0
    """

    if RESUME_RANKING_PATH.exists():
        rows = []

        with open(
            RESUME_RANKING_PATH,
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)

            cols = RESUME_RANKING_COLUMN_MAP

            for row in reader:
                if cols["resume_text"] not in row:
                    continue

                try:
                    matched_score = float(
                        row.get(cols["match_label"], 0) or 0
                    )
                except (ValueError, TypeError):
                    matched_score = 0.0

                label = 1 if matched_score >= 0.70 else 0

                rows.append(
                    ResumeRankingRow(
                        resume_text=row[cols["resume_text"]].strip(),
                        job_title=row.get(
                            cols["job_title"],
                            "",
                        ).strip(),
                        job_description=row.get(
                            cols["job_description"],
                            "",
                        ).strip(),
                        match_label=label,
                    )
                )

        logger.info(
            "dataset.resume_ranking.loaded_real",
            n_rows=len(rows),
        )

        return rows, False

    logger.warning(
        "dataset.resume_ranking.synthetic_fallback",
        detail=(
            "Real dataset not found at "
            "data/external/resume_data_for_ranking.csv — "
            "using synthetic schema-matching data. See "
            "data/external/README.md to plug in the real "
            "Kaggle dataset."
        ),
    )

    return _synthetic_resume_ranking_dataset(), True


def _synthetic_resume_ranking_dataset() -> list[ResumeRankingRow]:
    """Schema-matching synthetic dataset.

    Reuses this project's own sample resumes/jobs data so the generated
    rows are at least internally coherent text, paired with simple
    keyword-overlap-derived labels.

    This exists purely to exercise
    `scripts/build_eval_set_from_resume_ranking.py` end to end.

    It is explicitly NOT presented as evaluation signal on its own.
    """

    import json

    data_dir = Path(__file__).resolve().parents[2] / "data"

    with open(
        data_dir / "sample_resumes.json",
        encoding="utf-8",
    ) as f:
        resumes = json.load(f)

    with open(
        data_dir / "sample_jobs.json",
        encoding="utf-8",
    ) as f:
        jobs = json.load(f)

    rows = []

    for resume in resumes:
        resume_tokens = set(
            resume["raw_text"].lower().split()
        )

        for job in jobs:
            job_tokens = set(
                job["raw_description"].lower().split()
            )

            overlap = len(
                resume_tokens & job_tokens
            )

            label = 1 if overlap >= 12 else 0

            rows.append(
                ResumeRankingRow(
                    resume_text=resume["raw_text"],
                    job_title=job["title"],
                    job_description=job["raw_description"],
                    match_label=label,
                )
            )

    return rows