from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.learned_ranker import (
    CandidateJobFeatures,
    LearnedRanker,
    LearnedRankerMetadata,
)

DATA_PATH = BACKEND_DIR / "data" / "processed" / "dataset_training_examples.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "recommender_dataset_v1.joblib"
METADATA_PATH = MODEL_DIR / "recommender_dataset_v1_metadata.json"


def main() -> None:
    print("\n" + "#" * 80)
    print("# DATASET -> LEARNED RANKER TRAINING")
    print("# OFFLINE ONLY - NO MONGODB - NO SCRAPING - NO LIVE CHANGES")
    print("#" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    features = CandidateJobFeatures.get_feature_names()
    df = pd.read_csv(DATA_PATH)

    required = {"candidate_id", "job_id", "label", "split", *features}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[features].isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_candidates = set(train_df["candidate_id"])
    test_candidates = set(test_df["candidate_id"])
    overlap = train_candidates & test_candidates
    if overlap:
        raise ValueError(f"Candidate leakage detected: {len(overlap)} candidates overlap.")

    X_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)

    print("\nDataset:")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Train rows       : {len(train_df):,}")
    print(f"  Test rows        : {len(test_df):,}")
    print(f"  Train candidates : {len(train_candidates):,}")
    print(f"  Test candidates  : {len(test_candidates):,}")
    print(f"  Leakage overlap  : {len(overlap):,}")

    print("\nTraining:")
    ranker = LearnedRanker()
    metadata = LearnedRankerMetadata(
        model_version="dataset_v1",
        training_timestamp=datetime.utcnow().isoformat(),
        n_training_examples=len(train_df),
        n_positive_examples=int(y_train.sum()),
        n_negative_examples=int(len(y_train) - y_train.sum()),
        feature_names=features,
        training_dataset_source="dataset_training_examples.csv",
        model_type="LogisticRegression",
    )

    train_metrics = ranker.train(X_train, y_train, metadata=metadata)
    print(json.dumps(train_metrics, indent=2))

    print("\nHeld-out test evaluation:")
    y_pred = ranker._model.predict(X_test)
    y_proba = ranker._model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    print(json.dumps(test_metrics, indent=2))

    # Ranking evaluation: one labelled positive and three sampled negatives
    # per candidate. Measure whether the positive ranks first.
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    ranking_candidates = 0

    for _, group in test_df.groupby("candidate_id"):
        if int(group["label"].sum()) != 1:
            continue

        X = group[features].to_numpy(dtype=np.float32)
        probs = ranker._model.predict_proba(X)[:, 1]
        order = np.argsort(-probs)
        labels = group.iloc[order]["label"].to_numpy()

        positive_positions = np.where(labels == 1)[0]
        if len(positive_positions) == 0:
            continue

        rank = int(positive_positions[0]) + 1
        ranking_candidates += 1
        reciprocal_ranks.append(1.0 / rank)

        if rank <= 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1

    ranking_metrics = {
        "ranking_candidates": ranking_candidates,
        "hit_rate_at_1": round(hits_at_1 / ranking_candidates, 4),
        "hit_rate_at_3": round(hits_at_3 / ranking_candidates, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
    }

    print("\nTest ranking metrics:")
    print(json.dumps(ranking_metrics, indent=2))

    ranker._metadata.evaluation_metrics = {
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"test_{k}": v for k, v in ranking_metrics.items()},
    }

    # IMPORTANT: separate filename so the live recommender model is untouched.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ranker.save_model(MODEL_PATH, METADATA_PATH)

    print("\nSaved:")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Metadata : {METADATA_PATH}")
    print("\nLIVE recommender was NOT changed.")


if __name__ == "__main__":
    main()
