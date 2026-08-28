"""
Build `data/eval_labels_dataset_augmented.json` from the "Resume Data for
Ranking" Kaggle dataset (or its synthetic fallback — see
`data/external/README.md`).

Usage:
    python scripts/build_eval_set_from_resume_ranking.py

Deliberately written to a SEPARATE file from the hand-labeled
`data/eval_labels.json` rather than merged into it: the two label sources
have different methodologies (hand-authored "would a career counselor
prioritize this" judgments vs. this dataset's own match_label signal), and
silently merging different labeling methodologies into one metric would
misrepresent what the resulting Precision@K/NDCG@K numbers mean. The
evaluation endpoint (`GET /evaluation/run`) can be pointed at either file
explicitly; the README documents both and reports them separately.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.dataset_loaders import load_resume_ranking_dataset  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_PATH = DATA_DIR / "eval_labels_dataset_augmented.json"


def main() -> None:
    rows, is_synthetic = load_resume_ranking_dataset()
    print(f"Loaded {len(rows)} resume-ranking rows [{'SYNTHETIC fallback' if is_synthetic else 'real dataset'}]")

    # Group by resume text (as a stand-in candidate identity) and collect
    # the set of job titles labeled relevant for it.
    by_resume: dict[str, list] = defaultdict(list)
    for row in rows:
        by_resume[row.resume_text].append(row)

    cases = []
    for i, (resume_text, rows_for_resume) in enumerate(by_resume.items()):
        relevant_titles = [r.job_title for r in rows_for_resume if r.match_label == 1]
        if not relevant_titles:
            continue
        cases.append(
            {
                "synthetic_candidate_id": f"dataset_cand_{i:03d}",
                "resume_text": resume_text,
                "relevant_job_titles": relevant_titles,
            }
        )

    output = {
        "methodology": (
            "Cases derived from the 'Resume Data for Ranking' Kaggle dataset "
            f"({'SYNTHETIC fallback — see data/external/README.md' if is_synthetic else 'real dataset'}). "
            "Relevance here comes from the dataset's own match_label column, a different "
            "methodology from data/eval_labels.json's hand-authored judgments — kept in a "
            "separate file rather than merged so the two label sources are never silently "
            "conflated into one metric. These cases reference resume text and job TITLES "
            "directly (not this project's internal candidate_id/job_id), since they weren't "
            "ingested into the running system's store — a harness wanting to evaluate against "
            "them would first POST each resume_text to /candidates/resume and match job titles "
            "to ingested job_ids."
        ),
        "is_synthetic": is_synthetic,
        "cases": cases,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(cases)} dataset-augmented eval cases to {OUTPUT_PATH}")
    if is_synthetic:
        print(
            "\nNOTE: ran against synthetic fallback data. Place the real CSV at "
            "data/external/resume_ranking.csv and re-run for real dataset coverage."
        )


if __name__ == "__main__":
    main()
