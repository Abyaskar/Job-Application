from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from app.services.ranking import _normalize_skill


BACKEND_DIR = Path(__file__).resolve().parents[1]

TRAINING_DATA_PATH = (
    BACKEND_DIR / "data" / "external" / "training_data.csv"
)

JOB_ROLES_PATH = (
    BACKEND_DIR / "data" / "external" / "job_roles.csv"
)


def parse_list(value):
    """Convert a CSV skill field into a Python list."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            items = []

            for item in parsed:
                # Dataset stores multiple skills inside one string
                # separated by "|".
                parts = str(item).split("|")

                for part in parts:
                    part = part.strip()

                    if part:
                        items.append(part)

            return items

    except (ValueError, SyntaxError):
        pass

    # Fallback for plain text.
    return [
        item.strip()
        for item in text.replace("|", ",").split(",")
        if item.strip()
    ]

def main():
    print("\n" + "=" * 100)
    print("REAL SKILL MATCH INVESTIGATION")
    print("=" * 100)

    resumes = pd.read_csv(TRAINING_DATA_PATH)
    jobs = pd.read_csv(JOB_ROLES_PATH)

    print(f"\nTraining resumes : {len(resumes)}")
    print(f"Job roles        : {len(jobs)}")

    # Normalize column names so the script is easier to maintain.
    resumes.columns = [str(c).strip() for c in resumes.columns]
    jobs.columns = [str(c).strip() for c in jobs.columns]

    # Build lookup by Job Role.
    job_lookup = {}

    for _, job in jobs.iterrows():
        title = str(job["Job Title"]).strip()

        required_skills = parse_list(job["Required Skills"])

        normalized_required = {
            _normalize_skill(skill)
            for skill in required_skills
            if skill.strip()
        }

        job_lookup[title.lower()] = {
            "title": title,
            "required_skills": required_skills,
            "normalized_required": normalized_required,
        }

    # Only inspect positive examples.
    positive_resumes = resumes.head(0)

    print("\nFinding positive resume/job pairs with ZERO skill matches...")

    zero_match_examples = []

    for index, resume in resumes.iterrows():

        job_role = str(resume["Job Role"]).strip()

        job = job_lookup.get(job_role.lower())

        if job is None:
            continue

        resume_skills = parse_list(resume["Skills"])

        normalized_resume = {
            _normalize_skill(skill)
            for skill in resume_skills
            if skill.strip()
        }

        matches = sorted(
            normalized_resume & job["normalized_required"]
        )

        if len(matches) == 0:

            zero_match_examples.append(
                {
                    "index": index,
                    "job_role": job_role,
                    "resume_skills": resume_skills,
                    "required_skills": job["required_skills"],
                    "normalized_resume": sorted(normalized_resume),
                    "normalized_required": sorted(
                        job["normalized_required"]
                    ),
                }
            )

        if len(zero_match_examples) >= 20:
            break

    print(
        f"\nShowing {len(zero_match_examples)} examples "
        "where the labelled job has ZERO exact skill matches."
    )

    for example in zero_match_examples:

        print("\n" + "-" * 100)

        print(f"Resume row       : {example['index']}")
        print(f"Job Role         : {example['job_role']}")

        print(
            f"\nResume Skills:\n"
            f"  {example['resume_skills']}"
        )

        print(
            f"\nRequired Job Skills:\n"
            f"  {example['required_skills']}"
        )

        print(
            f"\nNormalized Resume Skills:\n"
            f"  {example['normalized_resume']}"
        )

        print(
            f"\nNormalized Required Skills:\n"
            f"  {example['normalized_required']}"
        )

        print("\nActual Exact Matches:")
        print("  NONE")

    print("\n" + "=" * 100)
    print("INVESTIGATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()