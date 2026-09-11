"""
Hard-Negative Dataset Diagnostic

Purpose:
    Diagnose whether the hard-negative dataset is actually harder
    and whether the synthetic labels are still too easy to separate.

This script:
- Loads dataset_training_examples_hard_negatives.csv
- Checks feature redundancy
- Checks constant features
- Compares positive vs negative distributions
- Checks skill-coverage separation
- Checks candidate label structure
- Measures hard-negative difficulty
- Runs controlled LogisticRegression experiments
- Checks feature/label correlation
- Compares hard negatives against positives

NO model is saved.
NO production model is changed.
NO MongoDB changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "semantic_similarity",
    "intent_alignment",
    "required_skill_coverage",
    "preferred_skill_coverage",
    "matched_required_skill_count",
    "missing_required_skill_count",
    "experience_match",
    "education_match",
    "location_match",
    "seniority_match",
    "domain_match",
    "resume_experience_years",
    "required_experience_years",
    "skill_gap_ratio",
    "intent_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the hard-negative ranking dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset_training_examples_hard_negatives.csv",
        help="Path to the hard-negative training dataset.",
    )

    return parser.parse_args()


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_dataset(path_string: str) -> pd.DataFrame:
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required_columns = FEATURE_NAMES + ["label"]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    if "split" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'split' column."
        )

    return df


# ---------------------------------------------------------------------------
# 1. FEATURE REDUNDANCY
# ---------------------------------------------------------------------------

def check_feature_redundancy(df: pd.DataFrame) -> None:
    print_header("1. FEATURE REDUNDANCY CHECK")

    X = df[FEATURE_NAMES].astype(float)

    found = False

    for i, feature_a in enumerate(FEATURE_NAMES):
        for feature_b in FEATURE_NAMES[i + 1:]:

            a = X[feature_a]
            b = X[feature_b]

            if np.array_equal(a.values, b.values):
                print(
                    f"EXACT DUPLICATE: "
                    f"{feature_a} == {feature_b}"
                )
                found = True
                continue

            correlation = a.corr(b)

            if pd.notna(correlation) and abs(correlation) >= 0.95:
                print(
                    f"{feature_a:<36}"
                    f"{feature_b:<36}"
                    f"r={correlation:.4f}"
                )
                found = True

    if not found:
        print("No highly correlated feature pairs found.")


# ---------------------------------------------------------------------------
# 2. CONSTANT FEATURES
# ---------------------------------------------------------------------------

def check_constant_features(df: pd.DataFrame) -> None:
    print_header("2. CONSTANT FEATURES")

    found = False

    for feature in FEATURE_NAMES:
        unique_count = df[feature].nunique(dropna=False)

        if unique_count <= 1:
            print(
                f"  {feature:<36}"
                f"unique_values={unique_count} "
                f"value={df[feature].iloc[0]}"
            )
            found = True

    if not found:
        print("No constant features found.")


# ---------------------------------------------------------------------------
# 3. POSITIVE VS NEGATIVE DISTRIBUTIONS
# ---------------------------------------------------------------------------

def positive_negative_distribution(
    df: pd.DataFrame,
) -> None:

    print_header("3. POSITIVE vs NEGATIVE DISTRIBUTIONS")

    means = df.groupby("label")[FEATURE_NAMES].mean().T

    result = pd.DataFrame(
        {
            "negative_mean": means.get(
                0,
                pd.Series(index=FEATURE_NAMES),
            ),
            "positive_mean": means.get(
                1,
                pd.Series(index=FEATURE_NAMES),
            ),
        }
    )

    print(
        result.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ---------------------------------------------------------------------------
# 4. SKILL COVERAGE SEPARATION
# ---------------------------------------------------------------------------

def skill_coverage_analysis(
    df: pd.DataFrame,
) -> None:

    print_header("4. SKILL COVERAGE SEPARATION")

    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    positive_full = (
        positives["required_skill_coverage"] == 1.0
    ).mean()

    negative_full = (
        negatives["required_skill_coverage"] == 1.0
    ).mean()

    print(
        "Positive coverage == 1.0 : "
        f"{positive_full:.4%}"
    )

    print(
        "Negative coverage == 1.0 : "
        f"{negative_full:.4%}"
    )

    print()

    print("Coverage quantiles:")

    quantiles = (
        df.groupby("label")[
            "required_skill_coverage"
        ]
        .quantile(
            [0.00, 0.25, 0.50, 0.75, 1.00]
        )
        .unstack()
    )

    quantiles.columns = [
        "0.00",
        "0.25",
        "0.50",
        "0.75",
        "1.00",
    ]

    print(
        quantiles.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()

    print(
        "Negative examples with coverage >= 0.50: "
        f"{(negatives['required_skill_coverage'] >= 0.50).mean():.4%}"
    )

    print(
        "Negative examples with coverage >= 0.67: "
        f"{(negatives['required_skill_coverage'] >= 0.67).mean():.4%}"
    )

    print(
        "Negative examples with coverage >= 0.83: "
        f"{(negatives['required_skill_coverage'] >= 0.83).mean():.4%}"
    )


# ---------------------------------------------------------------------------
# 5. LABEL STRUCTURE
# ---------------------------------------------------------------------------

def label_structure(df: pd.DataFrame) -> None:

    print_header("5. POSITIVE LABEL STRUCTURE")

    if "candidate_id" not in df.columns:
        print("candidate_id not available.")
        return

    positives_per_candidate = (
        df.groupby("candidate_id")["label"]
        .sum()
    )

    total = len(positives_per_candidate)

    exactly_one = (
        positives_per_candidate == 1
    ).sum()

    zero = (
        positives_per_candidate == 0
    ).sum()

    multiple = (
        positives_per_candidate > 1
    ).sum()

    print(
        "Candidates with exactly one positive: "
        f"{exactly_one / total:.4%}"
    )

    print(
        f"Candidates with zero positives: {zero}"
    )

    print(
        f"Candidates with multiple positives: {multiple}"
    )


# ---------------------------------------------------------------------------
# 6. HARD NEGATIVE DIFFICULTY
# ---------------------------------------------------------------------------

def hard_negative_difficulty(
    df: pd.DataFrame,
) -> None:

    print_header("6. HARD-NEGATIVE DIFFICULTY")

    negatives = df[df["label"] == 0]

    features = [
        "semantic_similarity",
        "required_skill_coverage",
        "experience_match",
        "education_match",
        "seniority_match",
        "domain_match",
    ]

    stats = negatives[features].describe(
        percentiles=[0.25, 0.50, 0.75]
    ).T

    stats = stats[
        [
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "mean",
        ]
    ]

    print(
        stats.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ---------------------------------------------------------------------------
# 7. CONTROLLED MODEL EXPERIMENTS
# ---------------------------------------------------------------------------

def build_model() -> Pipeline:

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
    features: list[str],
) -> None:

    X_train = train_df[features].astype(float)
    y_train = train_df["label"].astype(int)

    X_test = test_df[features].astype(float)
    y_test = test_df["label"].astype(int)

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(
        f"{name:<55}"
        f"Accuracy={accuracy:.4f} "
        f"ROC-AUC={roc_auc:.4f}"
    )


def controlled_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print_header(
        "7. CONTROLLED MODEL EXPERIMENTS"
    )

    # A: all features
    all_features = FEATURE_NAMES.copy()

    # B: remove only duplicate skill_gap_ratio
    without_skill_gap = [
        feature
        for feature in FEATURE_NAMES
        if feature != "skill_gap_ratio"
    ]

    # C: remove all direct skill features
    direct_skill_features = {
        "required_skill_coverage",
        "preferred_skill_coverage",
        "matched_required_skill_count",
        "missing_required_skill_count",
        "skill_gap_ratio",
    }

    without_direct_skill = [
        feature
        for feature in FEATURE_NAMES
        if feature not in direct_skill_features
    ]

    # D: non-skill matching features
    semantic_profile = [
        "semantic_similarity",
        "experience_match",
        "education_match",
        "seniority_match",
        "domain_match",
    ]

    # E: semantic similarity alone
    semantic_only = [
        "semantic_similarity",
    ]

    # F: skill coverage alone
    skill_only = [
        "required_skill_coverage",
    ]

    run_experiment(
        train_df,
        test_df,
        "A - All 15 features",
        all_features,
    )

    run_experiment(
        train_df,
        test_df,
        "B - Remove skill_gap_ratio",
        without_skill_gap,
    )

    run_experiment(
        train_df,
        test_df,
        "C - Remove all direct skill features",
        without_direct_skill,
    )

    run_experiment(
        train_df,
        test_df,
        "D - Semantic + experience + education + seniority + domain",
        semantic_profile,
    )

    run_experiment(
        train_df,
        test_df,
        "E - Semantic similarity only",
        semantic_only,
    )

    run_experiment(
        train_df,
        test_df,
        "F - Required skill coverage only",
        skill_only,
    )


# ---------------------------------------------------------------------------
# 8. FEATURE / LABEL CORRELATION
# ---------------------------------------------------------------------------

def feature_label_correlation(
    df: pd.DataFrame,
) -> None:

    print_header(
        "8. FEATURE vs LABEL CORRELATION"
    )

    correlations = {}

    for feature in FEATURE_NAMES:

        correlation = df[feature].corr(
            df["label"]
        )

        correlations[feature] = correlation

    ordered = sorted(
        correlations.items(),
        key=lambda item: (
            -abs(item[1])
            if pd.notna(item[1])
            else float("inf")
        ),
    )

    for feature, correlation in ordered:

        if pd.isna(correlation):
            print(
                f"{feature:<36} NaN"
            )
        else:
            print(
                f"{feature:<36}"
                f"{correlation:.4f}"
            )


# ---------------------------------------------------------------------------
# 9. SOURCE BREAKDOWN
# ---------------------------------------------------------------------------

def source_breakdown(
    df: pd.DataFrame,
) -> None:

    print_header(
        "9. NEGATIVE SOURCE BREAKDOWN"
    )

    if "source" not in df.columns:
        print("source column not available.")
        return

    negatives = df[df["label"] == 0]

    counts = (
        negatives["source"]
        .value_counts()
    )

    total = len(negatives)

    for source, count in counts.items():

        percentage = (
            count / total
            if total
            else 0.0
        )

        print(
            f"{source:<45}"
            f"{count:>8,} "
            f"({percentage:.2%})"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    print("=" * 80)
    print("HARD-NEGATIVE DATASET DIAGNOSTIC")
    print("=" * 80)

    df = load_dataset(
        args.dataset
    )

    print()
    print(
        f"Dataset: "
        f"{Path(args.dataset).resolve()}"
    )

    print(
        f"Rows   : {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    print()
    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Test rows : {len(test_df):,}"
    )

    check_feature_redundancy(df)

    check_constant_features(df)

    positive_negative_distribution(df)

    skill_coverage_analysis(df)

    label_structure(df)

    hard_negative_difficulty(df)

    controlled_experiments(
        train_df,
        test_df,
    )

    feature_label_correlation(df)

    source_breakdown(df)

    print_header(
        "DIAGNOSTIC COMPLETE"
    )


if __name__ == "__main__":
    main()