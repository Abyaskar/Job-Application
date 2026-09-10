"""
Experience Distribution, EDA, and Job-Priority Analysis
==========================================================
Project: Job Application Strategy AI

Purpose
-------
This script studies the resume dataset before we use it for recommendation
or machine-learning work.

Main questions answered:
1. How many resume records look like freshers?
2. How many have 1-3 years of experience?
3. How many have more than 3 years of experience?
4. What does the overall resume/job dataset look like?
5. What experience levels are most common in the job postings?
6. How can jobs be prioritized differently for freshers, early-career
   candidates, and experienced candidates?

IMPORTANT DATA NOTE
-------------------
The resume CSV is a resume-job ranking dataset. A row should therefore be
interpreted as a RESUME RECORD / resume-job observation unless you have
independent evidence that every row represents a unique person.

This script does NOT use `matched_score` as a real-world ground-truth label.
It is treated as an existing dataset score for EDA only. Real-user feedback
should remain the production learning signal in `learning_examples`.

Experience calculation
----------------------
Experience is estimated from `start_dates`, `end_dates`, and `positions`.
Overlapping employment periods are merged so that the same time period is
not counted twice.

Internship-only positions are excluded from professional experience because
this project is specifically interested in job-eligibility priority for
freshers. This is a project decision, not a universal definition of
"experience".

Reference date
--------------
For open-ended jobs such as "Current" / "Till Date", the script uses the
latest explicit YEAR found in the dataset as an analysis reference year.
This prevents a historical dataset from being inflated simply because the
analysis is being run today.

Outputs
-------
- Console EDA and interpretation
- CSV summary files
- PNG charts (if matplotlib is installed)

Run from the project/backend directory, for example:
    python experience_eda_and_priority_analysis.py

If your CSV files are somewhere else, change the two paths in `main()`.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. FILE CONFIGURATION
# ---------------------------------------------------------------------------

# The script is currently prepared for the two CSV files used in this project.
# Change these paths if you move the files.
RESUME_CSV = Path("/workspaces/Job-Application//workspaces/Job-Application/backend/data/external/resume_data_for_ranking.csv")
JOB_CSV = Path("/workspaces/Job-Application//workspaces/Job-Application/backend/data/external/all_job_post.csv")

# All analysis outputs will be placed here.
OUTPUT_DIR = Path("experience_eda_output")


# ---------------------------------------------------------------------------
# 2. BASIC DISPLAY HELPERS
# ---------------------------------------------------------------------------


def print_header(title: str) -> None:
    """Print a clear section heading so an unfamiliar reader can follow the EDA."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a smaller section heading."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def print_percentage(count: int | float, total: int | float) -> str:
    """Return a safely formatted percentage."""
    if total == 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


# ---------------------------------------------------------------------------
# 3. LOAD DATA
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both datasets and perform basic column validation."""
    print_header("STEP 1 — LOADING DATA")

    print(f"Resume dataset: {RESUME_CSV}")
    print(f"Job dataset:    {JOB_CSV}")

    if not RESUME_CSV.exists():
        raise FileNotFoundError(
            f"Resume CSV was not found: {RESUME_CSV.resolve()}\n"
            "Place the CSV beside this script or update RESUME_CSV."
        )

    if not JOB_CSV.exists():
        raise FileNotFoundError(
            f"Job CSV was not found: {JOB_CSV.resolve()}\n"
            "Place the CSV beside this script or update JOB_CSV."
        )

    resumes = pd.read_csv(RESUME_CSV)
    jobs = pd.read_csv(JOB_CSV)

    print(f"\nResume dataset loaded successfully: {resumes.shape[0]:,} rows × {resumes.shape[1]} columns")
    print(f"Job dataset loaded successfully:    {jobs.shape[0]:,} rows × {jobs.shape[1]} columns")

    required_resume = {
        "start_dates",
        "end_dates",
        "positions",
        "matched_score",
        "experiencere_requirement",
    }
    required_jobs = {"job_id", "category", "job_title", "job_description", "job_skill_set"}

    missing_resume = required_resume - set(resumes.columns)
    missing_jobs = required_jobs - set(jobs.columns)

    if missing_resume:
        raise ValueError(f"Resume dataset is missing required columns: {sorted(missing_resume)}")
    if missing_jobs:
        raise ValueError(f"Job dataset is missing required columns: {sorted(missing_jobs)}")

    return resumes, jobs


# ---------------------------------------------------------------------------
# 4. GENERAL EDA
# ---------------------------------------------------------------------------


def run_general_eda(resumes: pd.DataFrame, jobs: pd.DataFrame) -> None:
    """Perform general dataset-quality and distribution checks."""
    print_header("STEP 2 — GENERAL EDA BEFORE EXPERIENCE CATEGORIZATION")

    print_subheader("Dataset dimensions")
    print(f"Resume records : {len(resumes):,}")
    print(f"Job postings   : {len(jobs):,}")

    print_subheader("Duplicate and missing-value overview")
    print(f"Exact duplicate resume rows : {resumes.duplicated().sum():,}")
    print(f"Exact duplicate job rows    : {jobs.duplicated().sum():,}")

    resume_missing = resumes.isna().mean().sort_values(ascending=False) * 100
    job_missing = jobs.isna().mean().sort_values(ascending=False) * 100

    print("\nTop 10 resume columns with missing values:")
    print(resume_missing.head(10).round(2).to_string())

    print("\nJob columns with missing values:")
    print(job_missing.round(2).to_string())

    print_subheader("Existing matched_score distribution")
    if pd.api.types.is_numeric_dtype(resumes["matched_score"]):
        score = resumes["matched_score"].dropna()
        print(score.describe().round(4).to_string())
        print(
            "\nInterpretation: `matched_score` is useful for understanding the existing "
            "dataset and for evaluation, but this script does NOT treat it as "
            "real user feedback."
        )

    print_subheader("Job category distribution")
    if "category" in jobs.columns:
        category_counts = jobs["category"].fillna("Unknown").value_counts()
        print(category_counts.to_string())

    print_subheader("Important interpretation before categorization")
    print(
        "The resume dataset contains 9,544 resume records and the job dataset "
        "contains 1,167 job postings. Experience should be derived from the "
        "resume work-history fields, not from `experiencere_requirement`, "
        "because that column describes what a JOB requires."
    )
    print(
        "We therefore calculate candidate/resume experience first, and separately "
        "calculate the experience requirement of each job."
    )


# ---------------------------------------------------------------------------
# 5. PARSE LIST-LIKE CSV FIELDS
# ---------------------------------------------------------------------------


def parse_list_field(value: Any) -> list[Any]:
    """Convert strings such as "['Jan 2020', 'Feb 2021']" into Python lists."""
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (ValueError, SyntaxError):
        # If the field is not a valid Python-style list, keep it as one item.
        return [text]


# ---------------------------------------------------------------------------
# 6. INFER A REFERENCE DATE FOR HISTORICAL DATA
# ---------------------------------------------------------------------------


def infer_reference_date(resumes: pd.DataFrame) -> pd.Timestamp:
    """
    Infer a sensible historical reference date from explicit dates in the data.

    We intentionally do not use today's date because that could turn an old
    "Current" role into several extra years of experience merely because the
    analysis is being run later.
    """
    print_subheader("Inferring analysis reference date")

    # For employment experience, use work-history dates only. Education dates
    # such as `passing_years` must NOT make an old "Current" job appear newer.
    candidate_date_columns = ["start_dates", "end_dates"]
    parsed_dates: list[pd.Timestamp] = []

    for column in candidate_date_columns:
        if column not in resumes.columns:
            continue

        for value in resumes[column].dropna():
            for item in parse_list_field(value):
                if item is None:
                    continue

                text = str(item).strip()
                if not text:
                    continue

                # Ignore open-ended markers because they are not explicit dates.
                if text.lower() in {"current", "till date", "till today", "present", "now"}:
                    continue

                parsed = pd.to_datetime(text, errors="coerce")
                if pd.notna(parsed):
                    parsed_dates.append(parsed)

    if not parsed_dates:
        # Conservative fallback if no usable date exists.
        reference = pd.Timestamp("2021-12-31")
        print(f"No explicit dates were found. Using fallback reference date: {reference.date()}")
        return reference

    latest = max(parsed_dates)

    # Passing years can contain January timestamps even though only a year was
    # supplied. We use the end of that year as the analysis reference.
    reference = pd.Timestamp(year=latest.year, month=12, day=31)

    print(f"Latest explicit date/year found : {latest.date()}")
    print(f"Analysis reference date         : {reference.date()}")
    print(
        "This reference is used only for open-ended work history such as "
        "'Current' or 'Till Date'."
    )

    return reference


# ---------------------------------------------------------------------------
# 7. EXPERIENCE CALCULATION
# ---------------------------------------------------------------------------


def parse_date(value: Any, reference_date: pd.Timestamp) -> pd.Timestamp | None:
    """Parse a date unless it represents an open-ended/current marker."""
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    open_markers = {"current", "till date", "till today", "present", "now"}
    if text.lower() in open_markers:
        return reference_date

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None

    return parsed


def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping work periods so overlapping jobs are not double-counted."""
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[list[pd.Timestamp]] = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


def calculate_experience_years(row: pd.Series, reference_date: pd.Timestamp) -> float:
    """
    Estimate professional experience in years.

    Rules:
    - Use start_dates + end_dates.
    - Open-ended end dates use the reference date.
    - Internship positions are excluded from professional experience.
    - Invalid/missing periods are skipped.
    - Overlapping periods are merged.
    """
    starts = parse_list_field(row.get("start_dates"))
    ends = parse_list_field(row.get("end_dates"))
    positions = parse_list_field(row.get("positions"))

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for index, start_value in enumerate(starts):
        end_value = ends[index] if index < len(ends) else None
        position_value = positions[index] if index < len(positions) else ""

        position_text = str(position_value or "")

        # Internship experience is deliberately excluded for this project.
        if "intern" in position_text.lower():
            continue

        start_date = parse_date(start_value, reference_date)
        end_date = parse_date(end_value, reference_date)

        if start_date is None or end_date is None:
            continue

        if end_date < start_date:
            continue

        intervals.append((start_date, end_date))

    merged = merge_intervals(intervals)
    total_days = sum((end - start).days for start, end in merged)

    return max(total_days / 365.25, 0.0)


def categorize_experience(years: float) -> str:
    """Map estimated experience into the project's three main candidate bands."""
    if years < 1:
        return "Fresher (<1 year)"
    if years <= 3:
        return "Early Career (1-3 years)"
    return "Experienced (>3 years)"


def add_candidate_experience_categories(
    resumes: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add estimated experience and the three project categories."""
    print_header("STEP 3 — CANDIDATE EXPERIENCE CATEGORIZATION")

    result = resumes.copy()

    print("Calculating professional experience from work-history dates...")
    result["estimated_experience_years"] = result.apply(
        lambda row: calculate_experience_years(row, reference_date), axis=1
    )
    result["experience_category"] = result["estimated_experience_years"].apply(
        categorize_experience
    )

    counts = result["experience_category"].value_counts()
    total = len(result)

    print_subheader("Three target candidate categories")
    ordered_categories = [
        "Fresher (<1 year)",
        "Early Career (1-3 years)",
        "Experienced (>3 years)",
    ]

    for category in ordered_categories:
        count = int(counts.get(category, 0))
        print(f"{category:28s}: {count:6,d} records ({print_percentage(count, total)})")

    print_subheader("Experience statistics")
    print(result["estimated_experience_years"].describe().round(2).to_string())

    print_subheader("Project interpretation")
    fresher_count = int(counts.get("Fresher (<1 year)", 0))
    early_count = int(counts.get("Early Career (1-3 years)", 0))
    experienced_count = int(counts.get("Experienced (>3 years)", 0))

    print(
        f"• Freshers represent {print_percentage(fresher_count, total)} of resume records."
    )
    print(
        f"• Early-career candidates (1-3 years) represent {print_percentage(early_count, total)}."
    )
    print(
        f"• Experienced candidates (>3 years) represent {print_percentage(experienced_count, total)}."
    )
    print(
        "\nImportant: these are RESUME RECORD counts. The dataset structure does "
        "not independently prove that every row is a different person."
    )

    print(
        "\nRecommendation for the product: freshers should receive explicit job "
        "priority rather than being mixed into the same ranking pool as highly "
        "experienced candidates. The exact priority should still respect the "
        "candidate's skills, education, location, and semantic job match."
    )

    return result


# ---------------------------------------------------------------------------
# 8. JOB EXPERIENCE REQUIREMENT CATEGORIZATION
# ---------------------------------------------------------------------------


def parse_min_experience_requirement(value: Any) -> float | np.nan:
    """Extract the minimum stated experience from a job requirement string."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "n/a"}:
        return np.nan

    # Examples handled:
    # "At least 5 years"
    # "At least 1 year"
    # "1 to 3 years"
    # "2 to 5 years"
    # "5 year(s)"
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan

    return float(match.group(1))


def categorize_job_requirement(min_years: float | np.nan, job_title: Any = "") -> str:
    """
    Categorize a job using its minimum experience requirement.

    If the requirement is missing but the title clearly says Intern, Graduate,
    Fresher, Entry Level, Trainee, or Apprentice, we classify it as
    fresher-friendly. This is useful for this project because internship and
    trainee postings are often missing an explicit numeric experience value.
    """
    if pd.isna(min_years):
        title = str(job_title or "").lower()
        fresher_title_pattern = (
            r"\bintern\b|\bgraduate\b|\bfresher\b|"
            r"entry[- ]level|\btrainee\b|\bapprentice\b"
        )
        if re.search(fresher_title_pattern, title):
            return "Fresher-Friendly (<1 year)"
        return "Unknown / Not Specified"
    if min_years < 1:
        return "Fresher-Friendly (<1 year)"
    if min_years <= 3:
        return "1-3 Years"
    return "Experienced (>3 years)"


def add_job_experience_categories(
    resumes: pd.DataFrame, jobs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze experience requirements found in the resume-job ranking dataset.

    The standalone `all_job_post` CSV does not contain an experience-
    requirement column or a job_id that links it to the resume ranking CSV.
    Therefore we cannot honestly assign an exact experience category to each
    of the 1,167 job postings from that file alone.

    Instead, we classify the `experiencere_requirement` observations present
    in the resume ranking dataset and clearly label them as requirement
    observations, not unique job counts.
    """
    print_header("STEP 4 — JOB EXPERIENCE REQUIREMENT ANALYSIS")

    result = resumes.copy()
    result["minimum_experience_years"] = result["experiencere_requirement"].apply(
        parse_min_experience_requirement
    )
    result["job_experience_category"] = result.apply(
        lambda row: categorize_job_requirement(
            row["minimum_experience_years"], row.get("job_position_name", "")
        ),
        axis=1,
    )

    counts = result["job_experience_category"].value_counts()
    total = len(result)

    print_subheader("Experience requirement distribution in ranking observations")
    ordered_categories = [
        "Fresher-Friendly (<1 year)",
        "1-3 Years",
        "Experienced (>3 years)",
        "Unknown / Not Specified",
    ]

    for category in ordered_categories:
        count = int(counts.get(category, 0))
        print(f"{category:30s}: {count:6,d} observations ({print_percentage(count, total)})")

    print_subheader("Important dataset limitation")
    print(
        "The standalone all_job_post CSV has 1,167 jobs but does not contain "
        "the experience requirement field. The resume ranking CSV contains "
        "experiencere_requirement, but it does not contain job_id. Therefore "
        "this script does NOT pretend these observations are 1,167 uniquely "
        "classified jobs. A later ETL step should create a proper job-level "
        "experience table when a reliable job_id mapping is available."
    )

    print_subheader("Example experience requirements")
    sample_columns = [
        c for c in [
            "job_position_name",
            "experiencere_requirement",
            "job_experience_category",
        ]
        if c in result.columns
    ]
    print(result[sample_columns].drop_duplicates().head(20).to_string(index=False))

    return result, jobs


# ---------------------------------------------------------------------------
# 9. PRODUCT PRIORITY LOGIC
# ---------------------------------------------------------------------------


def candidate_job_priority(candidate_category: str, job_category: str) -> int:
    """
    Return product priority 1/2/3 for a candidate-job experience pairing.

    Priority is an EXPERIENCE-LEVEL PRIORITY ONLY.
    It must NOT replace the actual recommendation score.

    The final recommendation system should combine this with:
    - semantic similarity
    - skill coverage
    - education match
    - experience match
    - location match
    - intent alignment
    - and eventually learned real-user behavior.
    """
    if candidate_category == "Fresher (<1 year)":
        if job_category == "Fresher-Friendly (<1 year)":
            return 1
        if job_category == "1-3 Years":
            return 2
        if job_category == "Experienced (>3 years)":
            return 3
        return 2

    if candidate_category == "Early Career (1-3 years)":
        if job_category == "1-3 Years":
            return 1
        if job_category == "Fresher-Friendly (<1 year)":
            return 2
        if job_category == "Experienced (>3 years)":
            return 3
        return 2

    # Experienced candidate
    if job_category == "Experienced (>3 years)":
        return 1
    if job_category == "1-3 Years":
        return 2
    if job_category == "Fresher-Friendly (<1 year)":
        return 3
    return 2


def show_priority_strategy() -> None:
    """Print the proposed product strategy in plain English."""
    print_header("STEP 5 — PRODUCT PRIORITY STRATEGY")

    priority_table = pd.DataFrame(
        [
            ["Fresher (<1 year)", "Fresher-Friendly (<1 year)", 1],
            ["Fresher (<1 year)", "1-3 Years", 2],
            ["Fresher (<1 year)", "Experienced (>3 years)", 3],
            ["Early Career (1-3 years)", "1-3 Years", 1],
            ["Early Career (1-3 years)", "Fresher-Friendly (<1 year)", 2],
            ["Early Career (1-3 years)", "Experienced (>3 years)", 3],
            ["Experienced (>3 years)", "Experienced (>3 years)", 1],
            ["Experienced (>3 years)", "1-3 Years", 2],
            ["Experienced (>3 years)", "Fresher-Friendly (<1 year)", 3],
        ],
        columns=["Candidate Category", "Job Category", "Priority"],
    )

    print(priority_table.to_string(index=False))

    print_subheader("How this should work in the real recommendation system")
    print(
        "Priority 1 does NOT mean 'always recommend this job'. It means the job "
        "gets a favorable experience-level position in the candidate's ranking."
    )
    print(
        "For a fresher, a well-matched fresher-friendly job should appear before "
        "a job requiring 3+ years. However, skill and semantic mismatch should "
        "still be able to push a bad job down the ranking."
    )
    print(
        "This protects the product from a common failure: showing a fresher a "
        "large list of jobs that technically exist but are unrealistic because "
        "they demand several years of experience."
    )


# ---------------------------------------------------------------------------
# 10. SCORE DISTRIBUTION BY EXPERIENCE GROUP
# ---------------------------------------------------------------------------


def analyze_score_by_candidate_group(resumes: pd.DataFrame) -> None:
    """Show how the dataset's existing matched score varies by experience group."""
    print_header("STEP 6 — EXISTING MATCHED SCORE BY CANDIDATE EXPERIENCE")

    summary = (
        resumes.groupby("experience_category", observed=True)["matched_score"]
        .agg(["count", "mean", "median", "min", "max"])
        .reindex(
            [
                "Fresher (<1 year)",
                "Early Career (1-3 years)",
                "Experienced (>3 years)",
            ]
        )
    )

    print(summary.round(3).to_string())

    print(
        "\nInterpretation: this is descriptive EDA only. A higher or lower "
        "matched_score in one group does not prove that the ranking system is "
        "fair or correct. It tells us what the existing dataset looks like."
    )


# ---------------------------------------------------------------------------
# 11. FRESHER-FOCUSED JOB AVAILABILITY ANALYSIS
# ---------------------------------------------------------------------------


def analyze_fresher_job_pool(resumes: pd.DataFrame, jobs: pd.DataFrame) -> None:
    """
    Measure the fresher-friendly experience-requirement pool.

    Because `all_job_post` has no experience-requirement column, this section
    reports ranking observations rather than claiming exact counts of unique
    jobs.
    """
    print_header("STEP 7 — FRESHER-FRIENDLY OPPORTUNITY SIGNAL")

    total = len(resumes)
    fresher_obs = resumes[
        resumes["job_experience_category"] == "Fresher-Friendly (<1 year)"
    ]
    early_obs = resumes[resumes["job_experience_category"] == "1-3 Years"]
    experienced_obs = resumes[
        resumes["job_experience_category"] == "Experienced (>3 years)"
    ]

    print(f"Resume-job ranking observations : {total:,}")
    print(
        f"Fresher-friendly observations : {len(fresher_obs):,} "
        f"({print_percentage(len(fresher_obs), total)})"
    )
    print(
        f"1-3 year observations          : {len(early_obs):,} "
        f"({print_percentage(len(early_obs), total)})"
    )
    print(
        f">3 year observations           : {len(experienced_obs):,} "
        f"({print_percentage(len(experienced_obs), total)})"
    )

    print(
        "\nThese percentages show how often each experience requirement appears "
        "in the ranking dataset. They are not the exact percentage of the "
        "1,167 unique jobs because the CSVs do not provide a reliable job_id "
        "mapping between the two datasets."
    )

    if not fresher_obs.empty:
        print_subheader("Sample fresher-friendly requirement observations")
        cols = [
            c for c in [
                "job_position_name",
                "experiencere_requirement",
                "job_experience_category",
            ]
            if c in fresher_obs.columns
        ]
        print(fresher_obs[cols].drop_duplicates().head(20).to_string(index=False))


# ---------------------------------------------------------------------------
# 12. OPTIONAL CHARTS
# ---------------------------------------------------------------------------


def create_charts(resumes: pd.DataFrame, jobs: pd.DataFrame) -> None:
    """Create simple charts for a report/presentation if matplotlib is available."""
    print_header("STEP 8 — CREATING EDA CHARTS")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping charts.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Candidate experience distribution.
    candidate_order = [
        "Fresher (<1 year)",
        "Early Career (1-3 years)",
        "Experienced (>3 years)",
    ]
    candidate_counts = resumes["experience_category"].value_counts().reindex(candidate_order).fillna(0)

    plt.figure(figsize=(9, 5))
    candidate_counts.plot(kind="bar")
    plt.title("Resume Records by Experience Category")
    plt.xlabel("Experience Category")
    plt.ylabel("Number of Resume Records")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    candidate_chart = OUTPUT_DIR / "candidate_experience_distribution.png"
    plt.savefig(candidate_chart, dpi=160)
    plt.close()
    print(f"Saved: {candidate_chart}")

    # Job experience requirement distribution.
    job_order = [
        "Fresher-Friendly (<1 year)",
        "1-3 Years",
        "Experienced (>3 years)",
        "Unknown / Not Specified",
    ]
    job_counts = resumes["job_experience_category"].value_counts().reindex(job_order).fillna(0)

    plt.figure(figsize=(9, 5))
    job_counts.plot(kind="bar")
    plt.title("Ranking Observations by Minimum Experience Requirement")
    plt.xlabel("Job Experience Category")
    plt.ylabel("Number of Ranking Observations")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    job_chart = OUTPUT_DIR / "job_experience_requirement_distribution.png"
    plt.savefig(job_chart, dpi=160)
    plt.close()
    print(f"Saved: {job_chart}")

    # Existing matched score by candidate group.
    plt.figure(figsize=(9, 5))
    resumes.boxplot(column="matched_score", by="experience_category", grid=False)
    plt.suptitle("")
    plt.title("Existing Matched Score by Candidate Experience Category")
    plt.xlabel("Experience Category")
    plt.ylabel("Matched Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    score_chart = OUTPUT_DIR / "matched_score_by_experience.png"
    plt.savefig(score_chart, dpi=160)
    plt.close()
    print(f"Saved: {score_chart}")


# ---------------------------------------------------------------------------
# 13. SAVE ANALYSIS DATA
# ---------------------------------------------------------------------------


def save_outputs(resumes: pd.DataFrame, jobs: pd.DataFrame) -> None:
    """Save enriched datasets and compact summaries for later analysis."""
    print_header("STEP 9 — SAVING ANALYSIS OUTPUTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enriched_resume_path = OUTPUT_DIR / "resume_experience_analysis.csv"
    enriched_job_path = OUTPUT_DIR / "job_experience_analysis.csv"

    resumes.to_csv(enriched_resume_path, index=False)
    # `job_experience_category` lives on the ranking observations because the
    # standalone all_job_post CSV has no experience requirement column.
    resumes[[
        c for c in [
            "job_position_name",
            "experiencere_requirement",
            "minimum_experience_years",
            "job_experience_category",
        ]
        if c in resumes.columns
    ]].drop_duplicates().to_csv(enriched_job_path, index=False)

    print(f"Saved: {enriched_resume_path}")
    print(f"Saved: {enriched_job_path}")

    candidate_summary = (
        resumes["experience_category"]
        .value_counts()
        .rename_axis("experience_category")
        .reset_index(name="resume_records")
    )
    candidate_summary["percentage"] = (
        candidate_summary["resume_records"] / len(resumes) * 100
    ).round(2)

    job_summary = (
        resumes["job_experience_category"]
        .value_counts()
        .rename_axis("job_experience_category")
        .reset_index(name="ranking_observations")
    )
    job_summary["percentage"] = (
        job_summary["ranking_observations"] / len(resumes) * 100
    ).round(2)

    candidate_summary_path = OUTPUT_DIR / "candidate_experience_summary.csv"
    job_summary_path = OUTPUT_DIR / "job_experience_summary.csv"

    candidate_summary.to_csv(candidate_summary_path, index=False)
    job_summary.to_csv(job_summary_path, index=False)

    print(f"Saved: {candidate_summary_path}")
    print(f"Saved: {job_summary_path}")


# ---------------------------------------------------------------------------
# 14. FINAL PROJECT INTERPRETATION
# ---------------------------------------------------------------------------


def print_final_interpretation(resumes: pd.DataFrame, jobs: pd.DataFrame) -> None:
    """Give a concise conclusion that can also help with the project presentation."""
    print_header("FINAL INTERPRETATION FOR JOB APPLICATION STRATEGY AI")

    candidate_counts = resumes["experience_category"].value_counts()
    job_counts = resumes["job_experience_category"].value_counts()

    fresher_candidates = int(candidate_counts.get("Fresher (<1 year)", 0))
    early_candidates = int(candidate_counts.get("Early Career (1-3 years)", 0))
    experienced_candidates = int(candidate_counts.get("Experienced (>3 years)", 0))

    fresher_jobs = int(job_counts.get("Fresher-Friendly (<1 year)", 0))
    early_jobs = int(job_counts.get("1-3 Years", 0))
    experienced_jobs = int(job_counts.get("Experienced (>3 years)", 0))

    print("\nCANDIDATE SIDE")
    print(f"  Freshers (<1 year)       : {fresher_candidates:,}")
    print(f"  Early career (1-3 years) : {early_candidates:,}")
    print(f"  Experienced (>3 years)   : {experienced_candidates:,}")

    print("\nJOB SIDE")
    print(f"  Fresher-friendly requirement observations: {fresher_jobs:,}")
    print(f"  1-3 year requirement observations       : {early_jobs:,}")
    print(f"  >3 year requirement observations        : {experienced_jobs:,}")

    print("\nPRODUCT DECISION")
    print(
        "  1. Make experience category an explicit feature of the recommendation pipeline."
    )
    print(
        "  2. Give freshers first visibility into fresher-friendly opportunities."
    )
    print(
        "  3. Keep 1-3 year opportunities as the second priority for freshers."
    )
    print(
        "  4. Keep >3 year opportunities as the third priority for freshers unless "
        "the candidate's skills/profile create a strong reason to surface them."
    )
    print(
        "  5. Do the same logic in reverse for experienced candidates rather than "
        "making the platform fresher-only."
    )

    print("\nML DECISION")
    print(
        "  Do NOT train the production learned ranker directly from `matched_score`."
    )
    print(
        "  Use this CSV analysis for development/evaluation, while real candidate "
        "feedback continues to populate `learning_examples`."
    )
    print(
        "  Once there are enough genuine positive and negative examples from multiple "
        "real candidates, train and evaluate the learned ranker from those examples."
    )

    print("\nNEXT RECOMMENDED STEP")
    print(
        "  Use the output of this script to design the experience-priority feature "
        "inside the recommendation scorer, then evaluate the baseline ranking "
        "before training the learned ranker."
    )


# ---------------------------------------------------------------------------
# 15. MAIN PROGRAM
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete EDA → categorization → priority analysis pipeline."""
    print("\n" + "#" * 80)
    print("JOB APPLICATION STRATEGY AI")
    print("EXPERIENCE-FOCUSED EDA + JOB PRIORITY ANALYSIS")
    print("#" * 80)
    print("\nThe analysis order is intentional:")
    print("  1. Load and understand the data")
    print("  2. Check data quality")
    print("  3. Categorize candidate experience")
    print("  4. Categorize job experience requirements")
    print("  5. Define candidate-specific job priorities")
    print("  6. Inspect existing scores")
    print("  7. Save reusable analysis outputs")

    resumes, jobs = load_data()

    # EDA comes before categorization so that the reader understands what the
    # columns mean and does not accidentally confuse job requirements with
    # candidate experience.
    run_general_eda(resumes, jobs)

    reference_date = infer_reference_date(resumes)
    resumes = add_candidate_experience_categories(resumes, reference_date)
    resumes, jobs = add_job_experience_categories(resumes, jobs)

    show_priority_strategy()
    analyze_score_by_candidate_group(resumes)
    analyze_fresher_job_pool(resumes, jobs)

    create_charts(resumes, jobs)
    save_outputs(resumes, jobs)
    print_final_interpretation(resumes, jobs)

    print_header("ANALYSIS COMPLETE")
    print(f"All reusable output files are in: {OUTPUT_DIR.resolve()}")
    print("No model was trained by this script.")
    print("That is intentional: this file is for EDA and product-priority analysis.")


if __name__ == "__main__":
    main()
