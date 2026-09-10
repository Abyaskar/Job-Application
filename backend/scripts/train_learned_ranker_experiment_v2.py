
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.learned_ranker import CandidateJobFeatures

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

#DATA_PATH = BACKEND_DIR / "data" / "processed" / "dataset_training_examples.csv"
DATA_PATH = BACKEND_DIR / "data" / "processed" / "dataset_training_examples_contrastive_negatives.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "recommender_experiment_non_skill_v2.joblib"
META_PATH = MODEL_DIR / "recommender_experiment_non_skill_v2_metadata.json"

#FEATURE_NAMES = CandidateJobFeatures.get_feature_names()
FEATURE_NAMES = [
    "semantic_similarity",
    "experience_match",
    "education_match",
    "domain_match",
    "resume_experience_years",
    "required_experience_years",
]
# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------------------
# Use the EXISTING candidate-level split
# ------------------------------------------------------------

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"Train rows: {len(train_df):,}")
print(f"Test rows : {len(test_df):,}")

X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
y_train = train_df["label"].values.astype(int)

X_test = test_df[FEATURE_NAMES].values.astype(np.float32)
y_test = test_df["label"].values.astype(int)

# ------------------------------------------------------------
# Build pipeline
# ------------------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )),
])

print("\nTraining LogisticRegression...")
pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

print("\nEvaluating...")

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="binary",
)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

classifier = pipeline.named_steps["classifier"]

coef_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "coefficient": classifier.coef_[0],
    "abs_weight": np.abs(classifier.coef_[0]),
}).sort_values("abs_weight", ascending=False)

print("\nTop learned features:")
print(coef_df[["feature", "coefficient"]].to_string(index=False))

# ------------------------------------------------------------
# Candidate ranking evaluation
# ------------------------------------------------------------

print("\nRanking evaluation...")

test_df = test_df.copy()
test_df["probability"] = y_prob

hits_at_1 = 0
hits_at_3 = 0
mrr_total = 0

candidate_groups = test_df.groupby("candidate_id")

for _, group in candidate_groups:
    ranked = group.sort_values("probability", ascending=False)

    labels = ranked["label"].tolist()

    if labels[0] == 1:
        hits_at_1 += 1

    if 1 in labels[:3]:
        hits_at_3 += 1

    positive_rank = labels.index(1) + 1
    mrr_total += 1 / positive_rank

n_candidates = len(candidate_groups)

hit1 = hits_at_1 / n_candidates
hit3 = hits_at_3 / n_candidates
mrr = mrr_total / n_candidates

print(f"Hit@1: {hit1:.4f}")
print(f"Hit@3: {hit3:.4f}")
print(f"MRR  : {mrr:.4f}")

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

joblib.dump(pipeline, MODEL_PATH)

metadata = {
    "model_version": "dataset_v2",
    "training_timestamp": datetime.utcnow().isoformat(),
    "n_training_examples": int(len(train_df)),
    "n_positive_examples": int(y_train.sum()),
    "n_negative_examples": int(len(y_train) - y_train.sum()),
    "feature_names": FEATURE_NAMES,
    "evaluation_metrics": {
        "test_accuracy": round(float(accuracy), 4),
        "test_precision": round(float(precision), 4),
        "test_recall": round(float(recall), 4),
        "test_f1": round(float(f1), 4),
        "test_roc_auc": round(float(roc_auc), 4),
        "test_ranking_candidates": int(n_candidates),
        "test_hit_rate_at_1": round(float(hit1), 4),
        "test_hit_rate_at_3": round(float(hit3), 4),
        "test_mrr": round(float(mrr), 4),
    },
    "training_dataset_source": DATA_PATH.name,
    "model_type": "LogisticRegression",
}

import json

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 80)
print("MODEL SAVED")
print("=" * 80)
print(MODEL_PATH)
print(META_PATH)