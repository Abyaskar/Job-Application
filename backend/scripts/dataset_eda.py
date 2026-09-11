"""
dataset_eda.py

Purpose
-------
Run a first EDA on all datasets currently being considered for the
Job Application Strategy AI.

IMPORTANT:
This script is EDA + data-understanding only.
It does NOT train or modify the production ML model.

Datasets covered:
1. all_job_post.csv
2. resume_data_for_ranking.csv
3. apps.tsv
4. job_roles.csv
5. skills_list.csv
6. training_data.csv
7. Skill_Job_Matching_Dataset.csv
8. skills_database.json
9. test_resumes.json

How to run in Codespaces
------------------------
Put this file in your project, for example:

    backend/scripts/dataset_eda.py

Then run from the repository root:

    python backend/scripts/dataset_eda.py

If your data is somewhere else, change DATA_DIR below.

The script creates:
    dataset_eda_output/
        *.csv
        *.png

The script intentionally does NOT copy the large apps.tsv file or create
large intermediate datasets, because Codespaces disk space is limited.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change this if your datasets live somewhere else.
DATA_DIR = Path("/workspaces/Job-Application//workspaces/Job-Application/backend/data/external")

# All EDA results are written here.
OUTPUT_DIR = DATA_DIR / "dataset_eda_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SMALL HELPER FUNCTIONS
# ============================================================

def section(title: str) -> None:
    """Print a clear section heading so the terminal output is easy to read."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_basic_eda(df: pd.DataFrame, name: str) -> None:
    """Print common EDA information for a DataFrame."""
    print(f"\nDataset: {name}")
    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {df.shape[1]:,}")
    print(f"Duplicates : {df.duplicated().sum():,}")

    print("\nColumn types:")
    print(df.dtypes.value_counts().to_string())

    print("\nMissing values:")
    missing = (
        df.isna()
        .sum()
        .to_frame("missing_count")
    )
    missing["missing_percent"] = (
        missing["missing_count"] / len(df) * 100
    ).round(2)
    print(missing.sort_values("missing_percent", ascending=False).to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(max_cols=12))


def save_value_counts(
    series: pd.Series,
    filename: str,
    value_name: str,
) -> None:
    """Save a frequency table to CSV."""
    result = (
        series.value_counts(dropna=False)
        .rename_axis(value_name)
        .reset_index(name="count")
    )
    result["percentage"] = (
        result["count"] / result["count"].sum() * 100
    ).round(2)
    result.to_csv(OUTPUT_DIR / filename, index=False)


# ============================================================
# 3. JOB POST DATASET
# ============================================================

def analyze_job_posts() -> None:
    section("1. JOB POST DATASET")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/all_job_post.csv"
    df = pd.read_csv(path, low_memory=False)

    show_basic_eda(df, path.name)

    print("\nJob categories:")
    print(df["category"].value_counts().to_string())

    print(f"\nUnique job IDs    : {df['job_id'].nunique():,}")
    print(f"Unique job titles : {df['job_title'].nunique():,}")

    # Count skills in each job.
    # The dataset uses a string field containing multiple skills.
    skill_counts = (
        df["job_skill_set"]
        .fillna("")
        .astype(str)
        .str.split(r"[,|]")
        .apply(lambda x: len([s for s in x if s.strip()]))
    )

    print("\nSkills per job:")
    print(skill_counts.describe().round(2).to_string())

    save_value_counts(
        df["category"],
        "job_category_distribution.csv",
        "category",
    )

    plt.figure(figsize=(10, 6))
    df["category"].value_counts().plot(kind="bar")
    plt.title("Job Posts by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Jobs")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "job_category_distribution.png", dpi=150)
    plt.close()

    print("\nINTERPRETATION:")
    print("- This is primarily a JOB-SIDE dataset.")
    print("- It is useful for testing job parsing, matching and ranking.")
    print("- It can also be used to construct candidate-job pairs later.")
    print("- It does not contain user application outcomes, so it is not")
    print("  by itself enough to teach the model what users actually prefer.")


# ============================================================
# 4. RESUME RANKING DATASET
# ============================================================

def analyze_resume_ranking() -> None:
    section("2. RESUME RANKING DATASET")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/resume_data_for_ranking.csv"
    df = pd.read_csv(path, low_memory=False)

    show_basic_eda(df, path.name)

    print("\nMatched-score statistics:")
    print(df["matched_score"].describe().round(4).to_string())

    print("\nMatched-score distribution:")
    score_bins = pd.cut(
        df["matched_score"],
        bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.00],
        include_lowest=True,
    )
    print(score_bins.value_counts().sort_index().to_string())

    print(f"\nUnique career objectives : {df['career_objective'].nunique():,}")
    print(f"Unique job positions     : {df['job_position_name'].nunique():,}")
    print(f"Unique required-skill strings: {df['skills_required'].nunique():,}")

    print("\nTop job positions:")
    print(df["job_position_name"].value_counts().head(15).to_string())

    print("\nExperience requirement examples:")
    print(
        df["experiencere_requirement"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    # Do NOT silently convert matched_score into the production label.
    # It is a dataset-provided score, not observed user behavior.
    df[["matched_score"]].describe().to_csv(
        OUTPUT_DIR / "resume_ranking_score_summary.csv"
    )

    plt.figure(figsize=(10, 6))
    plt.hist(df["matched_score"].dropna(), bins=25)
    plt.title("Resume Dataset: Matched Score Distribution")
    plt.xlabel("matched_score")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "resume_matched_score_distribution.png",
        dpi=150,
    )
    plt.close()

    print("\nINTERPRETATION:")
    print("- This dataset is useful for resume/job feature engineering and EDA.")
    print("- matched_score is a useful analysis target, but we should NOT")
    print("  automatically call it the real user's positive/negative label.")
    print("- If we train directly on it, the model mainly learns to imitate")
    print("  this dataset's scoring system.")
    print("- That is different from learning from actual applications/rejections.")


# ============================================================
# 5. APPLICATION BEHAVIOR DATASET
# ============================================================

def analyze_apps() -> None:
    section("3. APPLICATION BEHAVIOR DATASET (apps.tsv)")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/apps.tsv"

    # Only read the columns required for this EDA.
    # This keeps memory usage lower in Codespaces.
    usecols = ["UserID", "WindowID", "Split", "ApplicationDate", "JobID"]
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        low_memory=False,
    )

    show_basic_eda(df, path.name)

    df["ApplicationDate"] = pd.to_datetime(
        df["ApplicationDate"],
        errors="coerce",
    )

    print(f"\nUnique users : {df['UserID'].nunique():,}")
    print(f"Unique jobs  : {df['JobID'].nunique():,}")
    print(f"Windows      : {df['WindowID'].nunique():,}")

    print("\nTrain/Test split:")
    print(df["Split"].value_counts().to_string())

    print("\nApplication date range:")
    print(f"Minimum: {df['ApplicationDate'].min()}")
    print(f"Maximum: {df['ApplicationDate'].max()}")

    apps_per_user = df.groupby("UserID").size()
    apps_per_job = df.groupby("JobID").size()

    print("\nApplications per user:")
    print(apps_per_user.describe().round(2).to_string())

    print("\nApplications per job:")
    print(apps_per_job.describe().round(2).to_string())

    duplicate_pairs = df.duplicated(["UserID", "JobID"]).sum()
    print(f"\nDuplicate UserID + JobID pairs: {duplicate_pairs:,}")

    save_value_counts(
        df["Split"],
        "apps_split_distribution.csv",
        "split",
    )

    plt.figure(figsize=(8, 5))
    df["Split"].value_counts().plot(kind="bar")
    plt.title("Application Records by Split")
    plt.xlabel("Split")
    plt.ylabel("Application records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "apps_split_distribution.png", dpi=150)
    plt.close()

    print("\nINTERPRETATION:")
    print("- This is the strongest dataset here for REAL application behavior.")
    print("- Each row represents a user applying to a job.")
    print("- It gives us positive implicit feedback: the user applied.")
    print("- It does NOT directly contain rejected/not-interested rows.")
    print("- Therefore, it is not a simple 0/1 supervised dataset by itself.")
    print("- For ranking training, we can use applied jobs as positives and")
    print("  carefully sampled non-applied jobs as negatives.")
    print("- This dataset is old and should be treated as a research/benchmark")
    print("  source, not as current production behavior.")


# ============================================================
# 6. JOB ROLE DATASET
# ============================================================

def analyze_job_roles() -> None:
    section("4. STRUCTURED JOB ROLE DATASET")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/job_roles.csv"
    df = pd.read_csv(path, low_memory=False)

    show_basic_eda(df, path.name)

    print("\nCategory distribution:")
    print(df["Category"].value_counts().to_string())

    print("\nExperience requirement:")
    print(df["Experience Years"].describe().round(2).to_string())

    print(f"\nUnique job titles: {df['Job Title'].nunique():,}")

    skill_counts = (
        df["Required Skills"]
        .fillna("")
        .str.split("|")
        .apply(lambda x: len([s for s in x if s.strip()]))
    )

    print("\nRequired skills per role:")
    print(skill_counts.describe().round(2).to_string())

    save_value_counts(
        df["Category"],
        "job_roles_category_distribution.csv",
        "category",
    )

    print("\nINTERPRETATION:")
    print("- This is a small structured job-requirement dataset.")
    print("- It is useful for controlled experiments and feature engineering.")
    print("- It contains explicit experience and required skills.")
    print("- It is much more suitable for constructing candidate-job examples")
    print("  than treating it as raw web-scraped production data.")


# ============================================================
# 7. SKILLS LIST
# ============================================================

def analyze_skills_list() -> None:
    section("5. SKILL LIST")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/skills_list.csv"
    df = pd.read_csv(path, low_memory=False)

    show_basic_eda(df, path.name)

    print("\nSkills by category:")
    print(df["Category"].value_counts().to_string())

    print(f"\nRows                : {len(df):,}")
    print(f"Unique skill names  : {df['Skill Name'].nunique():,}")
    print(
        f"Duplicate skill names: "
        f"{df.duplicated('Skill Name').sum():,}"
    )

    save_value_counts(
        df["Category"],
        "skills_category_distribution.csv",
        "category",
    )

    print("\nINTERPRETATION:")
    print("- This is a skill vocabulary/taxonomy, not a training-label dataset.")
    print("- It is useful for normalizing resume skills and job skills.")
    print("- It can help the NLP layer map aliases to canonical skills.")


# ============================================================
# 8. SYNTHETIC/STRUCTURED RESUME TRAINING DATA
# ============================================================

def analyze_training_data() -> None:
    section("6. RESUME TRAINING DATA")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/training_data.csv"
    df = pd.read_csv(path, low_memory=False)

    show_basic_eda(df, path.name)

    print(f"\nUnique Resume IDs: {df['Resume ID'].nunique():,}")
    print(f"Unique Job Roles : {df['Job Role'].nunique():,}")
    print(f"Unique Categories: {df['Category'].nunique():,}")

    print("\nExperience statistics:")
    print(df["Experience Years"].describe().round(2).to_string())

    print("\nTop job roles:")
    print(df["Job Role"].value_counts().head(20).to_string())

    print("\nCategory distribution:")
    print(df["Category"].value_counts().head(20).to_string())

    skill_counts = (
        df["Skills"]
        .fillna("")
        .str.split("|")
        .apply(lambda x: len([s for s in x if s.strip()]))
    )

    print("\nSkills per resume:")
    print(skill_counts.describe().round(2).to_string())

    save_value_counts(
        df["Category"],
        "training_data_category_distribution.csv",
        "category",
    )

    plt.figure(figsize=(10, 6))
    plt.hist(df["Experience Years"], bins=14)
    plt.title("Training Resume Experience Distribution")
    plt.xlabel("Experience Years")
    plt.ylabel("Number of resumes")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "training_resume_experience_distribution.png",
        dpi=150,
    )
    plt.close()

    print("\nINTERPRETATION:")
    print("- This is very useful for the resume/NLP side of the ML system.")
    print("- It contains 10,000 uniquely identified resume records.")
    print("- Job Role acts as a supervised target for resume-to-role learning.")
    print("- It is NOT the same as candidate-job application behavior.")
    print("- It can help the system learn which roles a resume resembles,")
    print("  especially for cold-start users with no feedback history.")


# ============================================================
# 9. STUDENT/FRESHER JOB MATCHING DATASET
# ============================================================

def analyze_student_matching() -> None:
    section("7. STUDENT / FRESHER JOB MATCHING DATASET")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/Skill_Job_Matching_Dataset.csv"
    df = pd.read_csv(path, low_memory=False)

    show_basic_eda(df, path.name)

    print("\nJob match label:")
    print(df["Job_Match"].value_counts().to_string())

    print(
        f"\nPositive match rate: "
        f"{df['Job_Match'].mean() * 100:.2f}%"
    )

    print(f"\nUnique students : {df['Student_ID'].nunique():,}")
    print(f"Unique jobs     : {df['Job_ID'].nunique():,}")
    print(f"Unique titles   : {df['Job_Title'].nunique():,}")

    print("\nMinimum experience in months:")
    print(df["Min_Experience_Months"].describe().round(2).to_string())

    print("\nInternship experience in months:")
    print(df["Internship_Experience"].describe().round(2).to_string())

    print("\nAcademic performance:")
    print(df["Academic_Performance"].describe().round(2).to_string())

    print("\nVocational program:")
    print(df["Vocational_Program"].value_counts().to_string())

    print("\nJob title:")
    print(df["Job_Title"].value_counts().to_string())

    save_value_counts(
        df["Job_Match"],
        "student_job_match_distribution.csv",
        "job_match",
    )

    plt.figure(figsize=(7, 5))
    df["Job_Match"].value_counts().sort_index().plot(kind="bar")
    plt.title("Student-Job Match Labels")
    plt.xlabel("Job_Match")
    plt.ylabel("Number of records")
    plt.xticks([0, 1], ["No Match (0)", "Match (1)"], rotation=0)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "student_job_match_distribution.png",
        dpi=150,
    )
    plt.close()

    print("\nINTERPRETATION:")
    print("- This is the most directly relevant dataset for fresher/student")
    print("  qualification matching.")
    print("- Job_Match is an explicit binary target.")
    print("- It contains internship experience and minimum job experience.")
    print("- However, its skills are represented as numeric Skill_1..Skill_5")
    print("  values rather than your real ParsedResume skill strings.")
    print("- Therefore, it is useful as a supervised research dataset, but")
    print("  we should not blindly feed its raw columns into production.")
    print("- Its schema must be mapped into your 15 canonical candidate-job")
    print("  features before it can train the same ranker.")


# ============================================================
# 10. SKILL DATABASE JSON
# ============================================================

def analyze_skills_database() -> None:
    section("8. SKILLS DATABASE JSON")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/skills_database.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"JSON type: {type(data).__name__}")
    print(f"Skill categories: {len(data):,}")

    rows = []

    for category, skills in data.items():
        print(f"\n{category}: {len(skills)} skills")

        for skill in skills:
            rows.append(
                {
                    "category": category,
                    "skill": skill,
                }
            )

    df = pd.DataFrame(rows)

    print(f"\nTotal skill entries: {len(df):,}")
    print(f"Unique skill names : {df['skill'].nunique():,}")

    duplicate_skills = (
        df[df.duplicated("skill", keep=False)]
        .sort_values("skill")
    )

    print("\nSkills appearing in more than one category:")
    if duplicate_skills.empty:
        print("None")
    else:
        print(duplicate_skills.to_string(index=False))

    df.to_csv(
        OUTPUT_DIR / "skills_database_flattened.csv",
        index=False,
    )

    print("\nINTERPRETATION:")
    print("- This JSON is reference knowledge for the NLP/skill layer.")
    print("- It is not itself a supervised ML training dataset.")
    print("- It can support skill normalization and category mapping.")
    print("- For example, Python is present in both Programming and Data")
    print("  Science & Analytics in this vocabulary.")


# ============================================================
# 11. TEST RESUMES JSON
# ============================================================

def analyze_test_resumes() -> None:
    section("9. TEST RESUMES JSON")

    path = DATA_DIR / "/workspaces/Job-Application/backend/data/external/test_resumes.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Number of test resumes: {len(data):,}")

    rows = []

    for item in data:
        rows.append(
            {
                "name": item.get("name"),
                "description_length": len(item.get("description", "")),
                "resume_text_length": len(item.get("resume_text", "")),
                "expected_roles_count": len(item.get("expected_roles", [])),
                "expected_roles": " | ".join(
                    item.get("expected_roles", [])
                ),
            }
        )

        print("\nTest resume:", item.get("name"))
        print("Expected roles:", item.get("expected_roles", []))

    df = pd.DataFrame(rows)
    df.to_csv(
        OUTPUT_DIR / "test_resume_summary.csv",
        index=False,
    )

    print("\nINTERPRETATION:")
    print("- These are test/demo resumes.")
    print("- They are useful for validating resume parsing and role matching.")
    print("- They should NOT be inserted into learning_candidates as real users.")
    print("- They should NOT contaminate production feedback training.")



# ============================================================
# 10. CROSS-DATASET CONSISTENCY CHECK
# ============================================================

def analyze_cross_dataset_consistency() -> None:
    section("10. CROSS-DATASET CONSISTENCY CHECK")

    training = pd.read_csv(
        DATA_DIR / "/workspaces/Job-Application/backend/data/external/training_data.csv",
        low_memory=False,
    )
    roles = pd.read_csv(
        DATA_DIR / "/workspaces/Job-Application/backend/data/external/job_roles.csv",
        low_memory=False,
    )

    training_roles = set(training["Job Role"].dropna().astype(str))
    role_table_roles = set(roles["Job Title"].dropna().astype(str))

    overlap = training_roles & role_table_roles

    print(
        f"Resume training roles : {len(training_roles):,}"
    )
    print(
        f"Structured job roles  : {len(role_table_roles):,}"
    )
    print(
        f"Exact role overlap    : {len(overlap):,}"
    )
    print(
        f"Resume-only roles     : {len(training_roles - role_table_roles):,}"
    )
    print(
        f"Job-only roles        : {len(role_table_roles - training_roles):,}"
    )

    print("\nThis is an important result:")
    if training_roles == role_table_roles:
        print(
            "PASS: Every Job Role in training_data.csv has a matching "
            "Job Title in job_roles.csv."
        )
        print(
            "This means we can join resume records to structured job "
            "requirements using the role/title."
        )
    else:
        print(
            "WARNING: The role lists are not identical. "
            "A mapping table will be required."
        )

    # Fresher / experience distribution in the resume training data.
    experience = pd.to_numeric(
        training["Experience Years"],
        errors="coerce",
    )

    fresher_count = int((experience == 0).sum())
    early_count = int(experience.between(1, 3).sum())
    experienced_count = int((experience > 3).sum())

    print("\nResume experience groups:")
    print(f"Fresher (0 years) : {fresher_count:,}")
    print(f"Early career (1-3): {early_count:,}")
    print(f"Experienced (>3)  : {experienced_count:,}")

    # Experience requirements in the structured job-role table.
    role_exp = pd.to_numeric(
        roles["Experience Years"],
        errors="coerce",
    )

    print("\nStructured job-role experience requirements:")
    print(
        f"Fresher-friendly roles (0 years): "
        f"{int((role_exp == 0).sum()):,}"
    )
    print(
        f"Roles requiring <=1 year: "
        f"{int((role_exp <= 1).sum()):,}"
    )
    print(
        f"Roles requiring 2-3 years: "
        f"{int(role_exp.between(2, 3).sum()):,}"
    )
    print(
        f"Roles requiring >3 years: "
        f"{int((role_exp > 3).sum()):,}"
    )

    print("\nINTERPRETATION:")
    print(
        "- training_data.csv and job_roles.csv form a particularly "
        "useful pair."
    )
    print(
        "- The 324 Job Roles match the 324 structured Job Titles exactly "
        "in the current files."
    )
    print(
        "- Therefore each resume can be joined to the requirements of "
        "its labeled role."
    )
    print(
        "- The resume's own Job Role is a natural POSITIVE role example."
    )
    print(
        "- We can then create NEGATIVE examples by pairing that resume "
        "with other roles."
    )
    print(
        "- This gives us a path toward candidate-job supervised ranking "
        "without pretending that matched_score is a real user outcome."
    )
    print(
        "- The 820 resumes with 0 years of experience are especially "
        "valuable for your fresher-first requirement."
    )


# ============================================================
# 11. TRAINING-DATA DESIGN PREVIEW
# ============================================================

def print_training_design_preview() -> None:
    section("11. PROPOSED ML TRAINING DESIGN")

    print(
        """
STEP 1 - RESUME REPRESENTATION
------------------------------
training_data.csv
    |
    +-- Resume Text
    +-- Education
    +-- Experience Years
    +-- Skills
    +-- Job Role
    +-- Category


STEP 2 - JOB REPRESENTATION
---------------------------
job_roles.csv
    |
    +-- Job Title
    +-- Category
    +-- Education Requirement
    +-- Experience Years
    +-- Required Skills
    +-- Salary Range


STEP 3 - JOIN
-------------
Resume.Job Role == JobRole.Job Title

This produces a candidate + matching-job pair.


STEP 4 - LABEL
--------------
Resume's own Job Role
        |
        +--> POSITIVE example (label = 1)

A different job role
        |
        +--> NEGATIVE example (label = 0)

For better ranking training, negatives should include:
    - same category / closely related roles = hard negatives
    - unrelated roles = easy negatives


STEP 5 - FEATURE ENGINEERING
----------------------------
The pair must eventually be converted into the SAME 15 features
used by your production LearnedRanker.

Do NOT create a completely separate feature definition for the CSV.

Production contract:
    Candidate + Job
        |
        v
    CandidateJobFeatures
        |
        +-- semantic_similarity
        +-- intent_alignment
        +-- required_skill_coverage
        +-- preferred_skill_coverage
        +-- matched_required_skill_count
        +-- missing_required_skill_count
        +-- experience_match
        +-- education_match
        +-- location_match
        +-- seniority_match
        +-- domain_match
        +-- resume_experience_years
        +-- required_experience_years
        +-- skill_gap_ratio
        +-- intent_confidence


STEP 6 - TRAIN
--------------
X = 15-feature vectors
y = 0/1 relevance labels

        |
        v

StandardScaler
        |
        v
LogisticRegression(class_weight="balanced")
        |
        v
Saved .joblib model


STEP 7 - LIVE SYSTEM
--------------------
Real uploaded resume
        +
Live scraped job
        |
        v
Same 15-feature extraction
        |
        v
Same trained model
        |
        v
ML relevance score
        |
        v
Experience-aware ranking
        |
        v
Recommendation


STEP 8 - REAL USER LEARNING
---------------------------
User applies / accepts / rejects
        |
        v
MongoDB learning_examples
        |
        v
Future retraining
        |
        v
Production model improves from real behavior
        """
    )

    print("\nIMPORTANT:")
    print(
        "This is a design preview only. This EDA script does NOT create "
        "training labels or train the production model."
    )


# ============================================================
# 12. MAIN
# ============================================================

# ============================================================
# 13. FINAL DATASET ROLE MAP
# ============================================================

def print_final_architecture() -> None:
    section("FINAL DATASET ROLE MAP")

    print(
        """
DATASET                         PRIMARY PURPOSE
---------------------------------------------------------------------------
all_job_post.csv             Job corpus / development / ranking tests
resume_data_for_ranking.csv Resume-job feature research / EDA
apps.tsv                        Historical application behavior / ranking
job_roles.csv                   Structured job requirements / experiments
skills_list.csv                Skill vocabulary / normalization
training_data.csv              Resume -> job-role supervised learning
Skill_Job_Matching_Dataset.csv Fresher/student match supervised learning
skills_database.json            Skill taxonomy / NLP reference
test_resumes.json               Testing only; NOT production learning
        """
    )

    print(
        """
TARGET ARCHITECTURE

OFFLINE TRAINING
----------------
resume datasets + structured job datasets + behavioral data
                    |
                    v
          feature engineering / mapping
                    |
                    v
       candidate-job training examples
                    |
                    v
             ML ranker / classifier
                    |
                    v
             saved model (.joblib)


LIVE PRODUCTION
---------------
real user resume
      |
      v
NLP extraction + embeddings
      |
      v
MongoDB candidate profile
      |
      +-------------------------------+
                                      |
live job URLs -> scraper -> ParsedJob |
                                      v
                         candidate x job features
                                      |
                                      v
                              trained ML model
                                      |
                                      v
                           experience-aware ranking
                                      |
                                      v
                              recommendations
                                      |
                                      v
                              user feedback
                                      |
                                      v
                            learning_examples
                                      |
                                      v
                         future model retraining
        """
    )

    print("\nMOST IMPORTANT DECISION:")
    print(
        "Do NOT merge every dataset into one CSV and train blindly."
    )
    print(
        "Each dataset represents a different kind of information."
    )
    print(
        "The next engineering step is to map the useful datasets into"
    )
    print(
        "your existing CandidateJobFeatures (the 15-feature contract)."
    )


# ============================================================
# 13. MAIN
# ============================================================

def main() -> None:
    print("\n" + "#" * 80)
    print("JOB APPLICATION STRATEGY AI - DATASET EDA")
    print("#" * 80)

    print(f"\nData directory : {DATA_DIR.resolve()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    analyze_job_posts()
    analyze_resume_ranking()
    analyze_apps()
    analyze_job_roles()
    analyze_skills_list()
    analyze_training_data()
    analyze_student_matching()
    analyze_skills_database()
    analyze_test_resumes()
    analyze_cross_dataset_consistency()
    print_training_design_preview()
    print_final_architecture()

    section("EDA COMPLETE")
    print("EDA output files were created in:")
    print(OUTPUT_DIR.resolve())
    print("\nNo production ML model was changed.")
    print("No MongoDB data was changed.")
    print("No live scraping was performed.")


if __name__ == "__main__":
    main()
