"""
Diagnose a learned-ranker training dataset.

This script:
- Loads a candidate-job feature dataset.
- Supports a custom dataset path via --dataset.
- Checks feature redundancy.
- Checks constant features.
- Compares positive vs negative distributions.
- Checks positive skill coverage.
- Checks positive-label structure.
- Measures negative difficulty.
- Runs controlled LogisticRegression experiments.
- Measures feature/label correlation.

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
        description="Diagnose a learned-ranker training dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset_training_examples.csv",
        help="Path to the training-example CSV.",
    )

    return parser.parse_args()


def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n  {path.resolve()}"
        )

    df = pd.read_csv(path)

    missing_features = [
        feature for feature in FEATURE_NAMES
        if feature not in df.columns
    ]

    if "label" not in df.columns:
        raise ValueError("Dataset is missing required column: label")

    if missing_features:
        raise ValueError(
            "Dataset is missing feature columns:\n"
            + "\n".join(f"  - {feature}" for feature in missing_features)
        )

    return df


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def feature_redundancy_check(df: pd.DataFrame) -> None:
    print_header("1. FEATURE REDUNDANCY CHECK")

    X = df[FEATURE_NAMES].astype(float)

    found_high_correlation = False

    for i, feature_a in enumerate(FEATURE_NAMES):
        for feature_b in FEATURE_NAMES[i + 1:]:
            a = X[feature_a]
            b = X[feature_b]

            if np.array_equal(a.values, b.values):
                print(f"EXACT DUPLICATE: {feature_a} == {feature_b}")
                found_high_correlation = True
                continue

            if np.array_equal(a.values, -b.values):
                print(f"EXACT NEGATIVE DUPLICATE: {feature_a} == -{feature_b}")
                found_high_correlation = True
                continue

            correlation = a.corr(b)

            if pd.notna(correlation) and abs(correlation) >= 0.95:
                print(
                    f"{feature_a:<36}"
                    f"{feature_b:<36}"
                    f"r={correlation:.4f}"
                )
                found_high_correlation = True

    if not found_high_correlation:
        print("No highly correlated feature pairs found.")


def constant_feature_check(df: pd.DataFrame) -> None:
    print_header("2. CONSTANT FEATURES")

    found_constant = False

    for feature in FEATURE_NAMES:
        unique_values = df[feature].nunique(dropna=False)

        if unique_values <= 1:
            value = df[feature].iloc[0]
            print(
                f"  {feature:<36}"
                f"unique_values={unique_values} "
                f"value={value}"
            )
            found_constant = True

    if not found_constant:
        print("No constant features found.")


def positive_negative_distribution(df: pd.DataFrame) -> None:
    print_header("3. POSITIVE vs NEGATIVE DISTRIBUTIONS")

    means = df.groupby("label")[FEATURE_NAMES].mean().T

    output = pd.DataFrame(
        {
            "negative_mean": means.get(0, pd.Series(index=FEATURE_NAMES)),
            "positive_mean": means.get(1, pd.Series(index=FEATURE_NAMES)),
        }
    )

    print(output.to_string(float_format=lambda x: f"{x:.4f}"))


def positive_skill_coverage(df: pd.DataFrame) -> None:
    print_header("4. POSITIVE SKILL COVERAGE")

    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    positive_full = (
        (positives["required_skill_coverage"] == 1.0).mean()
        if len(positives)
        else 0.0
    )

    negative_full = (
        (negatives["required_skill_coverage"] == 1.0).mean()
        if len(negatives)
        else 0.0
    )

    print(
        "Positive examples with required_skill_coverage == 1.0: "
        f"{positive_full:.4%}"
    )

    print(
        "Negative examples with required_skill_coverage == 1.0: "
        f"{negative_full:.4%}"
    )

    print()
    print("Skill coverage quantiles:")

    quantiles = df.groupby("label")[
        "required_skill_coverage"
    ].quantile([0.00, 0.25, 0.50, 0.75, 1.00]).unstack()

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


def positive_label_structure(df: pd.DataFrame) -> None:
    print_header("5. POSITIVE LABEL STRUCTURE")

    if "candidate_id" not in df.columns:
        print("candidate_id column not available; skipping candidate analysis.")
        return

    positives_per_candidate = (
        df.groupby("candidate_id")["label"]
        .sum()
    )

    total_candidates = len(positives_per_candidate)

    exactly_one = (
        positives_per_candidate == 1
    ).sum()

    zero = (
        positives_per_candidate == 0
    ).sum()

    multiple = (
        positives_per_candidate > 1
    ).sum()

    exactly_one_pct = (
        exactly_one / total_candidates
        if total_candidates
        else 0.0
    )

    print(
        f"Candidates with exactly one positive: "
        f"{exactly_one_pct:.4%}"
    )
    print(f"Candidates with zero positives: {zero}")
    print(f"Candidates with multiple positives: {multiple}")


def negative_sampling_difficulty(df: pd.DataFrame) -> None:
    print_header("6. NEGATIVE SAMPLING DIFFICULTY")

    negatives = df[df["label"] == 0]

    difficulty_features = [
        "semantic_similarity",
        "required_skill_coverage",
        "experience_match",
        "education_match",
        "domain_match",
    ]

    stats = negatives[difficulty_features].describe(
        percentiles=[0.25, 0.50, 0.75]
    ).T

    stats = stats[
        ["min", "25%", "50%", "75%", "max", "mean"]
    ]

    print(
        stats.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


def make_model(X_train: pd.DataFrame) -> Pipeline:
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


def controlled_model_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
    features: list[str],
) -> None:
    X_train = train_df[features].astype(float)
    y_train = train_df["label"].astype(int)

    X_test = test_df[features].astype(float)
    y_test = test_df["label"].astype(int)

    model = make_model(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(
        f"{name:<55}"
        f"Accuracy={accuracy:.4f} "
        f"ROC-AUC={roc_auc:.4f}"
    )


def controlled_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    print_header("7. CONTROLLED MODEL EXPERIMENTS")

    all_features = FEATURE_NAMES.copy()

    without_skill_gap = [
        feature
        for feature in FEATURE_NAMES
        if feature != "skill_gap_ratio"
    ]

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

    semantic_and_profile = [
        "semantic_similarity",
        "experience_match",
        "education_match",
        "seniority_match",
        "domain_match",
    ]

    controlled_model_experiment(
        train_df,
        test_df,
        "A - All 15 features",
        all_features,
    )

    controlled_model_experiment(
        train_df,
        test_df,
        "B - Remove skill_gap_ratio",
        without_skill_gap,
    )

    controlled_model_experiment(
        train_df,
        test_df,
        "C - Remove all direct skill features",
        without_direct_skill,
    )

    controlled_model_experiment(
        train_df,
        test_df,
        "D - Semantic + experience + education + seniority + domain",
        semantic_and_profile,
    )


def feature_label_correlation(df: pd.DataFrame) -> None:
    print_header("8. FEATURE vs LABEL CORRELATION")

    correlations = {}

    for feature in FEATURE_NAMES:
        correlation = df[feature].corr(df["label"])
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
            print(f"{feature:<36} NaN")
        else:
            print(f"{feature:<36} {correlation:.4f}")


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("LEARNED RANKER DIAGNOSTIC")
    print("=" * 80)

    df = load_dataset(args.dataset)

    print()
    print(f"Dataset: {Path(args.dataset).resolve()}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        raise ValueError(
            "Dataset must contain a 'split' column "
            "with train/test values."
        )

    print()
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")

    feature_redundancy_check(df)
    constant_feature_check(df)
    positive_negative_distribution(df)
    positive_skill_coverage(df)
    positive_label_structure(df)
    negative_sampling_difficulty(df)
    controlled_experiments(train_df, test_df)
    feature_label_correlation(df)

    print_header("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()