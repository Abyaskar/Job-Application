from __future__ import annotations

from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    BACKEND_DIR / "data" / "processed" / "dataset_training_examples.csv"
)


SKILL_FEATURES = [
    "required_skill_coverage",
    "matched_required_skill_count",
    "missing_required_skill_count",
    "skill_gap_ratio",
]


def main() -> None:
    print("\n" + "=" * 80)
    print("SKILL FEATURE INVESTIGATION")
    print("=" * 80)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print(f"\nDataset shape: {df.shape}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nSkill features by label:")
    print("-" * 80)

    summary = (
        df.groupby("label")[SKILL_FEATURES]
        .agg(["mean", "median", "min", "max"])
        .round(4)
    )

    print(summary.to_string())

    print("\nSimple positive-vs-negative comparison:")
    print("-" * 80)

    for feature in SKILL_FEATURES:
        positive_mean = df.loc[df["label"] == 1, feature].mean()
        negative_mean = df.loc[df["label"] == 0, feature].mean()

        difference = positive_mean - negative_mean

        print(
            f"{feature:35s} "
            f"positive={positive_mean:.4f}  "
            f"negative={negative_mean:.4f}  "
            f"difference={difference:+.4f}"
        )

    print("\nSkill feature distributions:")
    print("-" * 80)

    for feature in SKILL_FEATURES:
        print(f"\n{feature}")
        print(df.groupby("label")[feature].value_counts().head(15).to_string())

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()