'''
Build dataset-derived candidate × job training examples.

IMPORTANT:
- This script PREPARES and INSPECTS data. It DOES NOT train the ML model.
- It does NOT write to MongoDB.
- It does NOT modify the live recommendation/scraping pipeline.
- It uses the SAME `extract_candidate_job_features()` function that
  production inference uses, so the feature definition stays consistent.

Primary datasets:
    data/external/training_data.csv
    data/external/job_roles.csv

Output:
    data/processed/dataset_training_examples.csv

Each resume gets:
    1 positive pair  -> its labelled Job Role
    N negative pairs -> plausible competing-role negatives plus some easy negatives

Example:
    Resume A + Data Scientist -> label 1
    Resume A + Java Developer -> label 0 (hard negative)
    Resume A + HR Manager      -> label 0 (easy negative)

The goal of this first stage is to verify that the dataset can be converted
into the project's existing 15-feature representation before we train anything.
'''

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Make `app/...` imports work when this script is run from the backend folder.
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import (  # noqa: E402
    ExtractedRequirements,
    ParsedJob,
    ParsedResume,
)
from app.services.embeddings import get_embedding_provider  # noqa: E402
from app.services.extraction import extract_resume_profile  # noqa: E402
from app.services.learned_ranker import (  # noqa: E402
    CandidateJobFeatures,
    extract_candidate_job_features,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXTERNAL_DIR = BACKEND_DIR / "data" / "external"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed"

RESUME_DATA_PATH = EXTERNAL_DIR / "training_data.csv"
JOB_ROLE_DATA_PATH = EXTERNAL_DIR / "job_roles.csv"
OUTPUT_PATH = OUTPUT_DIR / "dataset_training_examples_contrastive_negatives.csv"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Column aliases.
#
# Different downloads sometimes use slightly different capitalization or
# naming. The script tries these alternatives instead of forcing one exact
# spelling.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "resume_text": [
        "Resume Text",
        "resume_text",
        "Resume",
        "resume",
        "Text",
        "text",
    ],
    "education": [
        "Education",
        "education",
        "Education Level",
        "education_level",
    ],
    "experience": [
        "Experience Years",
        "experience_years",
        "Years Experience",
        "years_experience",
        "Experience",
        "experience",
    ],
    "skills": [
        "Skills",
        "skills",
        "Skill",
        "skill",
    ],
    "job_role": [
        "Job Role",
        "job_role",
        "Job Role Name",
        "job_role_name",
        "Role",
        "role",
    ],
    "job_title": [
        "Job Title",
        "job_title",
        "Job Role",
        "job_role",
        "Title",
        "title",
    ],
    "job_skills": [
        "Required Skills",
        "required_skills",
        "Required Skills",
        "Skills",
        "skills",
        "Skill Set",
        "skill_set",
    ],
    "job_experience": [
        "Experience Years",
        "experience_years",
        "Required Experience",
        "required_experience",
        "Min Experience",
        "min_experience",
        "Minimum Experience",
        "minimum_experience",
    ],
    "job_education": [
        "Education Requirement",
        "education_requirement",
        "Education Level",
        "education_level",
        "Required Education",
        "required_education",
    ],
    "job_location": [
        "Location",
        "location",
        "Job Location",
        "job_location",
    ],
    "job_domain": [
        "Domain",
        "domain",
        "Category",
        "category",
    ],
    "job_seniority": [
        "Seniority",
        "seniority",
        "Experience Level",
        "experience_level",
    ],
}


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------
def find_column(df: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    """Find the first matching column name."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find a required column.\n"
            f"Tried: {aliases}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def clean_text(value) -> str:
    """Convert NaN/None/etc. into safe text."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_number(value, default: float = 0.0) -> float:
    """Safely convert a value such as '2 years' or '2.5' to a number."""
    if pd.isna(value):
        return default

    text = str(value).strip().lower()

    # Direct numeric conversion first.
    try:
        return float(text)
    except ValueError:
        pass

    # Extract the first number from text.
    import re

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))

    return default


def parse_skill_list(value) -> list[str]:
    """
    Convert common skill formats into a clean Python list.

    Handles examples such as:
        "Python, SQL, Machine Learning"
        "['Python', 'SQL']"
        '["Python", "SQL"]'
        "Python | SQL | Machine Learning"
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()

        if not text:
            return []

        raw_items = None

        # Try Python-list / JSON-list syntax.
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    raw_items = parsed
                    break
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue

        if raw_items is None:
            # Most CSV skill columns are comma/semicolon/pipe separated.
            import re

            raw_items = re.split(r"[,;|]", text)

    result = []
    seen = set()

    for item in raw_items:
        skill = str(item).strip()
        if not skill:
            continue

        key = skill.lower()
        if key not in seen:
            seen.add(key)
            result.append(skill)

    return result


def normalize_role(role: str) -> str:
    """Normalize a role title for matching between the two datasets."""
    return " ".join(clean_text(role).lower().split())


def experience_bucket(years: float) -> str:
    """Project-specific experience groups."""
    if years < 1:
        return "Fresher (<1 year)"
    if years <= 3:
        return "Early Career (1-3 years)"
    return "Experienced (>3 years)"


# ---------------------------------------------------------------------------
# Load and validate datasets
# ---------------------------------------------------------------------------
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 80)
    print("STEP 1 - LOADING DATASETS")
    print("=" * 80)

    if not RESUME_DATA_PATH.exists():
        raise FileNotFoundError(f"Resume dataset not found: {RESUME_DATA_PATH}")

    if not JOB_ROLE_DATA_PATH.exists():
        raise FileNotFoundError(f"Job-role dataset not found: {JOB_ROLE_DATA_PATH}")

    print(f"[1/2] Reading resume dataset:\n      {RESUME_DATA_PATH}")
    resumes = pd.read_csv(RESUME_DATA_PATH)

    print(f"[2/2] Reading job-role dataset:\n      {JOB_ROLE_DATA_PATH}")
    jobs = pd.read_csv(JOB_ROLE_DATA_PATH)

    print("\nDataset sizes:")
    print(f"  training_data.csv : {len(resumes):,} rows × {len(resumes.columns)} columns")
    print(f"  job_roles.csv     : {len(jobs):,} rows × {len(jobs.columns)} columns")

    print("\nResume columns:")
    print(list(resumes.columns))

    print("\nJob-role columns:")
    print(list(jobs.columns))

    return resumes, jobs


def resolve_columns(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
) -> dict[str, str | None]:
    """Resolve all required/optional column names once."""
    print("\n" + "=" * 80)
    print("STEP 2 - RESOLVING COLUMN NAMES")
    print("=" * 80)

    resolved = {
        "resume_text": find_column(resumes, COLUMN_ALIASES["resume_text"]),
        "education": find_column(resumes, COLUMN_ALIASES["education"]),
        "experience": find_column(resumes, COLUMN_ALIASES["experience"]),
        "skills": find_column(resumes, COLUMN_ALIASES["skills"]),
        "job_role": find_column(resumes, COLUMN_ALIASES["job_role"]),
        "job_title": find_column(jobs, COLUMN_ALIASES["job_title"]),
        "job_skills": find_column(jobs, COLUMN_ALIASES["job_skills"]),
        "job_experience": find_column(jobs, COLUMN_ALIASES["job_experience"], required=False),
        "job_education": find_column(jobs, COLUMN_ALIASES["job_education"], required=False),
        "job_location": find_column(jobs, COLUMN_ALIASES["job_location"], required=False),
        "job_domain": find_column(jobs, COLUMN_ALIASES["job_domain"], required=False),
        "job_seniority": find_column(jobs, COLUMN_ALIASES["job_seniority"], required=False),
    }

    for key, value in resolved.items():
        print(f"  {key:18s} -> {value}")

    return resolved


# ---------------------------------------------------------------------------
# Build structured ParsedResume objects
# ---------------------------------------------------------------------------
def build_resume_objects(
    resumes: pd.DataFrame,
    columns: dict[str, str | None],
    limit: int | None,
) -> list[dict]:
    """
    Convert dataset rows into the same ParsedResume representation used by
    the application.

    IMPORTANT:
    We intentionally DO NOT include the labelled Job Role in raw_text.
    Including the answer inside the input would create target leakage.
    """
    print("\n" + "=" * 80)
    print("STEP 3 - CONVERTING DATASET RESUMES TO ParsedResume")
    print("=" * 80)

    if limit:
        source = resumes.head(limit).copy()
        print(f"Using first {len(source):,} resume rows because --limit={limit}.")
    else:
        source = resumes.copy()
        print(f"Using all {len(source):,} resume rows.")

    objects = []

    for index, row in source.iterrows():
        resume_text = clean_text(row[columns["resume_text"]])
        education_text = clean_text(row[columns["education"]])
        skills_text = clean_text(row[columns["skills"]])
        years = parse_number(row[columns["experience"]])
        labelled_role = clean_text(row[columns["job_role"]])

        # Build input text WITHOUT the target Job Role.
        raw_text_parts = [
        resume_text,
        f"Education: {education_text}" if education_text else "",
        f"Skills: {skills_text}" if skills_text else "",
        f"Experience: {years:g} years of experience",
        ]
        raw_text = "\n".join(part for part in raw_text_parts if part)

        # Run the project's existing resume extractor.
        profile = extract_resume_profile(raw_text)

        # The source dataset already contains structured skills.
        # Parse them explicitly so "|" separated skills remain separate.
        #
        # This fix is kept inside the offline dataset builder rather than
        # changing the production resume extractor. Real uploaded resumes may
        # use different skill formats.
        parsed_dataset_skills = parse_skill_list(skills_text)

        if parsed_dataset_skills:
            profile["skills"] = parsed_dataset_skills

        candidate_id = f"dataset_resume_{index + 1:06d}"

        resume = ParsedResume(
            candidate_id=candidate_id,
            raw_text=raw_text,
            preferred_locations=[],
            preferred_domains=[],
            **profile,
        )

        objects.append(
            {
                "candidate_id": candidate_id,
                "resume": resume,
                "labelled_role": labelled_role,
                "labelled_role_normalized": normalize_role(labelled_role),
                "source_row": index,
                "experience_bucket": experience_bucket(years),
            }
        )

        if len(objects) <= 3:
            print(f"\nExample resume #{len(objects)}:")
            print(f"  candidate_id : {candidate_id}")
            print(f"  target role  : {labelled_role}")
            print(f"  experience   : {years}")
            print(f"  bucket       : {experience_bucket(years)}")
            print(f"  extracted skills: {resume.skills[:10]}")

    print(f"\nCreated {len(objects):,} ParsedResume objects.")

    bucket_counts = pd.Series([x["experience_bucket"] for x in objects]).value_counts()
    print("\nExperience distribution in selected resumes:")
    for bucket, count in bucket_counts.items():
        print(f"  {bucket:25s}: {count:,}")

    return objects


# ---------------------------------------------------------------------------
# Build ParsedJob objects from job_roles.csv
# ---------------------------------------------------------------------------
def build_job_objects(
    jobs: pd.DataFrame,
    columns: dict[str, str | None],
) -> dict[str, ParsedJob]:
    """
    Convert job-role rows into ParsedJob objects.

    job_roles.csv is already structured, so we do not need to scrape anything
    here. This is an offline representation of job requirements.
    """
    print("\n" + "=" * 80)
    print("STEP 4 - CONVERTING JOB ROLES TO ParsedJob")
    print("=" * 80)

    job_map: dict[str, ParsedJob] = {}
    duplicate_titles = 0

    for index, row in jobs.iterrows():
        title = clean_text(row[columns["job_title"]])
        if not title:
            continue

        normalized_title = normalize_role(title)

        if normalized_title in job_map:
            duplicate_titles += 1
            continue

        skills = parse_skill_list(row[columns["job_skills"]])

        min_experience = (
            parse_number(row[columns["job_experience"]])
            if columns["job_experience"]
            else 0.0
        )

        education_level = 1
        if columns["job_education"]:
            education_text = clean_text(row[columns["job_education"]]).lower()

            if any(x in education_text for x in ["phd", "doctorate"]):
                education_level = 3
            elif any(x in education_text for x in ["master", "msc", "mba", "m.tech", "m.eng"]):
                education_level = 2
            elif any(x in education_text for x in ["bachelor", "bsc", "b.tech", "b.eng", "undergraduate"]):
                education_level = 1

        location = (
            clean_text(row[columns["job_location"]])
            if columns["job_location"]
            else "Remote"
        ) or "Remote"

        domain = (
            clean_text(row[columns["job_domain"]])
            if columns["job_domain"]
            else None
        ) or None

        seniority = (
            clean_text(row[columns["job_seniority"]])
            if columns["job_seniority"]
            else None
        ) or None

        # Build a readable description for the semantic embedding.
        # The actual structured requirements remain authoritative.
        description_parts = [
            f"Job Title: {title}",
            f"Required Skills: {', '.join(skills)}" if skills else "",
            f"Minimum Experience: {min_experience:g} years",
            f"Education Requirement: {clean_text(row[columns['job_education']])}"
            if columns["job_education"] and clean_text(row[columns["job_education"]])
            else "",
            f"Location: {location}",
        ]
        raw_description = "\n".join(x for x in description_parts if x)

        requirements = ExtractedRequirements(
            required_skills=skills,
            preferred_skills=[],
            min_experience_years=min_experience,
            education_level_required=education_level,
            location=location,
            domain=domain,
            seniority=seniority,
        )

        job = ParsedJob(
            job_id=f"dataset_job_{index + 1:06d}",
            title=title,
            company="Dataset",
            location=location,
            domain=domain,
            raw_description=raw_description,
            requirements=requirements,
        )

        job_map[normalized_title] = job

    print(f"Unique job roles created: {len(job_map):,}")
    print(f"Duplicate job titles skipped: {duplicate_titles:,}")

    return job_map


# ---------------------------------------------------------------------------
# Create positive + negative candidate-job pairs
# ---------------------------------------------------------------------------
def _normalize_skill_for_pairing(skill: str) -> str:
    """Normalize a skill for approximate overlap checks during negative sampling."""
    text = clean_text(skill).lower()
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    return " ".join(text.split())


def _candidate_skill_coverage(resume: ParsedResume, job: ParsedJob) -> float:
    """Estimate candidate coverage of a job's required skills for sampling only."""
    candidate_skills = {
        _normalize_skill_for_pairing(skill)
        for skill in (resume.skills or [])
        if _normalize_skill_for_pairing(skill)
    }
    required_skills = {
        _normalize_skill_for_pairing(skill)
        for skill in (job.requirements.required_skills or [])
        if _normalize_skill_for_pairing(skill)
    }

    if not required_skills:
        return 0.0

    return len(candidate_skills & required_skills) / len(required_skills)


def _hard_negative_score(resume: ParsedResume, job: ParsedJob) -> float:
    """Score a plausible competing role for contrastive negative sampling.

    Unlike the previous version, FULL skill coverage is NOT rejected.
    A candidate can be qualified for multiple roles; for this offline
    benchmark, non-target roles are treated as competing alternatives.
    This prevents required_skill_coverage from becoming a perfect label
    separator.
    """
    coverage = _candidate_skill_coverage(resume, job)

    candidate_years = float(resume.total_experience_years or 0.0)
    required_years = float(job.requirements.min_experience_years or 0.0)

    experience_closeness = 1.0 / (
        1.0 + abs(candidate_years - required_years)
    )

    candidate_education = len(
        getattr(resume, "education", []) or []
    )
    required_education = int(
        job.requirements.education_level_required or 0
    )

    education_match = 0.0
    if candidate_education and required_education:
        education_match = (
            1.0
            if candidate_education >= required_education
            else 0.0
        )

    # Skill overlap is still the strongest signal, but FULL coverage is
    # intentionally allowed. Experience and education make the competing
    # role more realistic.
    return (
        0.55 * coverage
        + 0.25 * experience_closeness
        + 0.20 * education_match
    )


def _select_negative_roles(
    resume: ParsedResume,
    target_role: str,
    job_map: dict[str, ParsedJob],
    negatives_per_positive: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Select competing-role negatives, including full-skill-coverage alternatives."""

    if negatives_per_positive <= 0:
        return []

    target_job = job_map[target_role]
    target_category = clean_text(target_job.domain).lower()

    same_category = []
    different_category = []
    full_coverage_roles = []

    for role, job in job_map.items():
        if role == target_role:
            continue

        category = clean_text(job.domain).lower()
        score = _hard_negative_score(resume, job)

        coverage = _candidate_skill_coverage(resume, job)

        # Explicitly collect fully covered alternative roles.
        if coverage >= 1.0:
            full_coverage_roles.append((score, role))

        if target_category and category == target_category:
            same_category.append((score, role))
        elif category != target_category:
            different_category.append((score, role))

    same_category.sort(key=lambda item: (-item[0], item[1]))
    full_coverage_roles.sort(key=lambda item: (-item[0], item[1]))

    selected = []

    # ------------------------------------------------------------
    # 1. FORCE ONE FULL-COVERAGE COMPETING ROLE
    # ------------------------------------------------------------
    if full_coverage_roles:
        selected.append((full_coverage_roles[0][1], "hard"))

    # ------------------------------------------------------------
    # 2. Fill remaining slots with strong same-category roles
    # ------------------------------------------------------------
    hard_count = min(2, negatives_per_positive)

    if len(selected) < hard_count and same_category:
        hard_pool_size = min(
            len(same_category),
            max(hard_count * 5, hard_count),
        )

        hard_pool = same_category[:hard_pool_size]

        available = [
            item
            for item in hard_pool
            if item[1] not in {role for role, _ in selected}
        ]

        needed = hard_count - len(selected)

        if available:
            chosen = rng.sample(
                available,
                min(needed, len(available)),
            )

            selected.extend(
                (role, "hard")
                for _, role in chosen
            )

    # ------------------------------------------------------------
    # 3. Fill remaining slots with easy negatives
    # ------------------------------------------------------------
    remaining = negatives_per_positive - len(selected)

    if remaining > 0 and different_category:
        selected_roles = {role for role, _ in selected}

        easy_roles = [
            role
            for _, role in different_category
            if role not in selected_roles
        ]

        n = min(remaining, len(easy_roles))

        if n > 0:
            selected.extend(
                (role, "easy")
                for role in rng.sample(easy_roles, n)
            )

    # ------------------------------------------------------------
    # 4. Final fallback
    # ------------------------------------------------------------
    remaining = negatives_per_positive - len(selected)

    if remaining > 0:
        selected_roles = {role for role, _ in selected}

        fallback_pool = [
            role
            for role in sorted(job_map)
            if role != target_role
            and role not in selected_roles
        ]

        if fallback_pool:
            n = min(remaining, len(fallback_pool))

            selected.extend(
                (role, "fallback")
                for role in rng.sample(fallback_pool, n)
            )

    return selected[:negatives_per_positive]

def create_pairs(
    resume_objects: list[dict],
    job_map: dict[str, ParsedJob],
    negatives_per_positive: int,
) -> list[dict]:
    print("\n" + "=" * 80)
    print("STEP 5 - CREATING POSITIVE AND HARD-NEGATIVE CANDIDATE × JOB PAIRS")
    print("=" * 80)

    rng = random.Random(RANDOM_SEED)

    pairs = []
    skipped = 0
    sampling_counts = {"hard": 0, "easy": 0, "fallback": 0}

    for item in resume_objects:
        candidate_id = item["candidate_id"]
        target_role = item["labelled_role_normalized"]
        resume = item["resume"]

        if target_role not in job_map:
            skipped += 1
            continue

        # ---------------------------------------------------------------
        # Positive pair:
        # Resume's labelled role == selected job role
        # ---------------------------------------------------------------
        positive_job = job_map[target_role]
        pairs.append(
            {
                "candidate_id": candidate_id,
                "job_id": positive_job.job_id,
                "candidate_role": item["labelled_role"],
                "job_title": positive_job.title,
                "label": 1,
                "source": "dataset_training_data_positive",
                "experience_bucket": item["experience_bucket"],
            }
        )

        # ---------------------------------------------------------------
        # Negative pairs:
        # 1) Prefer same-category competing-role hard negatives.
        # 2) Fill remaining slots with different-category easy negatives.
        # 3) Use a fallback only if the role taxonomy is too small.
        # ---------------------------------------------------------------
        negative_roles = _select_negative_roles(
            resume=resume,
            target_role=target_role,
            job_map=job_map,
            negatives_per_positive=negatives_per_positive,
            rng=rng,
        )

        for negative_role, negative_type in negative_roles:
            job = job_map[negative_role]
            sampling_counts[negative_type] += 1

            pairs.append(
                {
                    "candidate_id": candidate_id,
                    "job_id": job.job_id,
                    "candidate_role": item["labelled_role"],
                    "job_title": job.title,
                    "label": 0,
                    "source": f"dataset_{negative_type}_negative_sampling",
                    "experience_bucket": item["experience_bucket"],
                }
            )

    print(f"Resume rows skipped because their role was not found: {skipped:,}")
    print(f"Total candidate-job pairs created: {len(pairs):,}")

    print("\nNegative sampling breakdown:")
    print(f"  Hard negatives    : {sampling_counts['hard']:,}")
    print(f"  Easy negatives    : {sampling_counts['easy']:,}")
    print(f"  Fallback negatives: {sampling_counts['fallback']:,}")

    if pairs:
        labels = pd.Series([p["label"] for p in pairs]).value_counts().sort_index()
        print("\nLabel distribution:")
        print(f"  label 0 (negative): {int(labels.get(0, 0)):,}")
        print(f"  label 1 (positive): {int(labels.get(1, 0)):,}")

    return pairs


# ---------------------------------------------------------------------------
# Candidate-level train/test split
# ---------------------------------------------------------------------------
def assign_candidate_splits(pairs: list[dict], test_size: float = 0.20) -> None:
    """
    Split by candidate, NOT by individual pair.

    This is critical. If Resume A has 4 job pairs, all 4 must be in either
    train or test. Otherwise the same person's resume can leak into both.
    """
    print("\n" + "=" * 80)
    print("STEP 6 - CANDIDATE-LEVEL TRAIN / TEST SPLIT")
    print("=" * 80)

    candidate_ids = sorted({p["candidate_id"] for p in pairs})

    train_candidates, test_candidates = train_test_split(
        candidate_ids,
        test_size=test_size,
        random_state=RANDOM_SEED,
    )

    train_set = set(train_candidates)
    test_set = set(test_candidates)

    for pair in pairs:
        if pair["candidate_id"] in train_set:
            pair["split"] = "train"
        elif pair["candidate_id"] in test_set:
            pair["split"] = "test"
        else:
            raise RuntimeError("Candidate was not assigned to train or test.")

    print(f"Unique candidates : {len(candidate_ids):,}")
    print(f"Train candidates  : {len(train_candidates):,}")
    print(f"Test candidates   : {len(test_candidates):,}")

    print("\nPair split:")
    print(
        pd.Series([p["split"] for p in pairs])
        .value_counts()
        .to_string()
    )


# ---------------------------------------------------------------------------
# Generate the project's 15 features
# ---------------------------------------------------------------------------
def generate_features(
    pairs: list[dict],
    resume_map: dict[str, dict],
    job_map_by_id: dict[str, ParsedJob],
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("STEP 7 - GENERATING THE EXISTING 15 FEATURES")
    print("=" * 80)

    feature_names = CandidateJobFeatures.get_feature_names()

    print("\nCanonical feature order:")
    for number, name in enumerate(feature_names, start=1):
        print(f"  {number:2d}. {name}")

    # ---------------------------------------------------------------
    # Fit the project's embedding provider once.
    #
    # For this preparation stage we fit on all available text. In the
    # final ML training pipeline, the embedding/TF-IDF fitting step
    # should be performed using TRAIN candidates only to avoid leakage.
    # ---------------------------------------------------------------
    provider = get_embedding_provider()

    print("\nPreparing embeddings...")
    all_text = []

    for item in resume_map.values():
        all_text.append(item["resume"].raw_text)

    for job in job_map_by_id.values():
        all_text.append(job.raw_description)

    print(f"Texts supplied to embedding provider: {len(all_text):,}")
    provider.fit(all_text)

    # Embed resumes once.
    print("Embedding resumes...")
    for item in resume_map.values():
        resume = item["resume"]
        resume.embedding = provider.embed(resume.raw_text)

    # Embed jobs once.
    print("Embedding jobs...")
    for job in job_map_by_id.values():
        job.embedding = provider.embed(job.raw_description)

    print("Embeddings ready.")

    # ---------------------------------------------------------------
    # Calculate features.
    # ---------------------------------------------------------------
    output_rows = []

    total = len(pairs)

    for counter, pair in enumerate(pairs, start=1):
        resume_item = resume_map[pair["candidate_id"]]
        resume = resume_item["resume"]
        job = job_map_by_id[pair["job_id"]]

        # IMPORTANT:
        # We intentionally pass intent=None in this first dataset bridge.
        #
        # Why?
        # The source dataset's "Job Role" is already the supervised target.
        # Using that same target as career intent would leak the answer into
        # the features.
        features = extract_candidate_job_features(
            resume=resume,
            job=job,
            intent=None,
        )

        row = {
            "candidate_id": pair["candidate_id"],
            "job_id": pair["job_id"],
            "candidate_role": pair["candidate_role"],
            "job_title": pair["job_title"],
            "label": pair["label"],
            "source": pair["source"],
            "experience_bucket": pair["experience_bucket"],
            "split": pair["split"],
        }

        values = features.to_array(feature_names)

        for name, value in zip(feature_names, values):
            row[name] = float(value)

        output_rows.append(row)

        if counter <= 3:
            print(f"\nExample feature vector #{counter}:")
            print(f"  Candidate : {pair['candidate_id']}")
            print(f"  Job       : {pair['job_title']}")
            print(f"  Label     : {pair['label']}")
            for name, value in zip(feature_names, values):
                print(f"  {name:32s}: {float(value):.4f}")

        if counter % 1000 == 0 or counter == total:
            print(f"  Processed {counter:,}/{total:,} pairs")

    result = pd.DataFrame(output_rows)

    print(f"\nFeature matrix created: {result.shape[0]:,} rows × {result.shape[1]:,} columns")
    return result


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def run_quality_checks(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("STEP 8 - QUALITY CHECKS")
    print("=" * 80)

    feature_names = CandidateJobFeatures.get_feature_names()

    print("\n1. Missing values in feature columns:")
    missing = df[feature_names].isna().sum()
    print(missing.to_string())

    print("\n2. Feature ranges:")
    for feature in feature_names:
        minimum = df[feature].min()
        maximum = df[feature].max()
        mean = df[feature].mean()
        print(
            f"  {feature:32s} "
            f"min={minimum:.4f} "
            f"max={maximum:.4f} "
            f"mean={mean:.4f}"
        )

    print("\n3. Label distribution by experience group:")
    table = pd.crosstab(
        df["experience_bucket"],
        df["label"],
        margins=True,
    )
    print(table.to_string())

    print("\n4. Mean feature values by label:")
    label_means = df.groupby("label")[feature_names].mean().T
    print(label_means.round(4).to_string())

    print("\n5. Positive vs negative skill coverage:")
    skill_summary = df.groupby("label")["required_skill_coverage"].mean()
    print(skill_summary.round(4).to_string())

    print("\n6. Positive vs negative experience match:")
    exp_summary = df.groupby("label")["experience_match"].mean()
    print(exp_summary.round(4).to_string())

    print("\n7. Candidate leakage check:")
    train_candidates = set(df.loc[df["split"] == "train", "candidate_id"])
    test_candidates = set(df.loc[df["split"] == "test", "candidate_id"])
    overlap = train_candidates & test_candidates

    print(f"  Train candidates: {len(train_candidates):,}")
    print(f"  Test candidates : {len(test_candidates):,}")
    print(f"  Overlap         : {len(overlap):,}")

    if overlap:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED: some candidates appear in both train and test."
        )

    print("  ✓ No candidate overlap detected.")

    print("\n8. Constant features:")
    constant_features = [
        feature
        for feature in feature_names
        if df[feature].nunique(dropna=False) <= 1
    ]

    if constant_features:
        print("  WARNING - constant features:")
        for feature in constant_features:
            print(f"    {feature}")
    else:
        print("  ✓ No constant features.")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_output(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("STEP 9 - SAVING DATASET TRAINING EXAMPLES")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved successfully:")
    print(f"  {OUTPUT_PATH}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset-derived candidate-job examples using the existing 15-feature extractor."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of resume rows to process. Default: all rows.",
    )

    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=3,
        help="Number of negative jobs per positive job. The builder prefers 2 plausible competing roles from the same category and fills remaining slots with different-category negatives. Default: 3.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    if args.negatives_per_positive < 1:
        raise ValueError("--negatives-per-positive must be at least 1.")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "#" * 80)
    print("# DATASET → CANDIDATE × JOB → 15-FEATURE PREPARATION")
    print("# NO MODEL TRAINING IN THIS SCRIPT")
    print("# NO MONGODB CHANGES")
    print("# NO SCRAPING")
    print("#" * 80)

    resumes, jobs = load_datasets()

    columns = resolve_columns(resumes, jobs)

    resume_objects = build_resume_objects(
        resumes=resumes,
        columns=columns,
        limit=args.limit,
    )

    job_map = build_job_objects(
        jobs=jobs,
        columns=columns,
    )

    pairs = create_pairs(
        resume_objects=resume_objects,
        job_map=job_map,
        negatives_per_positive=args.negatives_per_positive,
    )

    if not pairs:
        raise RuntimeError(
            "No candidate-job pairs were created. "
            "Check whether Job Role values match job_roles.csv titles."
        )

    assign_candidate_splits(pairs)

    resume_map = {
        item["candidate_id"]: item
        for item in resume_objects
    }

    job_map_by_id = {
        job.job_id: job
        for job in job_map.values()
    }

    feature_df = generate_features(
        pairs=pairs,
        resume_map=resume_map,
        job_map_by_id=job_map_by_id,
    )

    run_quality_checks(feature_df)

    save_output(feature_df)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        """
What this script proved:

1. The resume dataset can be converted into ParsedResume objects.
2. The job-role dataset can be converted into ParsedJob objects.
3. Positive and negative candidate-job pairs can be created.
4. The same 15-feature extractor used by production inference can
   calculate features for the dataset.
5. Train/test splitting is done at candidate level to prevent resume leakage.
6. The result is saved as a normal CSV that we can inspect before training.

What this script DID NOT do:

- It did not train LogisticRegression.
- It did not write anything to MongoDB.
- It did not modify real-user learning_examples.
- It did not scrape live jobs.
- It did not change recommendation behaviour.

The next stage, after you inspect this output, is to connect this CSV
to LearnedRanker.train() and train/evaluate the model.
"""
    )


if __name__ == "__main__":
    main()
