from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from app.services.learned_ranker import CandidateJobFeatures


BACKEND_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BACKEND_DIR / "models" / "recommender_dataset_v1.joblib"
METADATA_PATH = BACKEND_DIR / "models" / "recommender_dataset_v1_metadata.json"


def main() -> None:
    print("\n" + "=" * 80)
    print("LEARNED RANKER - MODEL INSPECTION")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    feature_names = CandidateJobFeatures.get_feature_names()

    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]

    coefficients = classifier.coef_[0]

    print("\nModel:")
    print(f"  Type       : {classifier.__class__.__name__}")
    print(f"  Features   : {len(feature_names)}")
    print(f"  Samples    : {classifier.n_iter_[0]} solver iterations")

    print("\nFeature coefficients:")
    print("-" * 80)

    feature_weights = list(zip(feature_names, coefficients))

    # Largest positive influence first
    feature_weights.sort(key=lambda x: x[1], reverse=True)

    for feature, coefficient in feature_weights:
        print(f"{feature:35s} {coefficient:+.6f}")

    print("\nFeature importance by absolute coefficient:")
    print("-" * 80)

    absolute_weights = sorted(
        feature_weights,
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    for feature, coefficient in absolute_weights:
        print(f"{feature:35s} |coef|={abs(coefficient):.6f}")

    print("\nScaler:")
    print("-" * 80)

    for feature, mean, scale in zip(
        feature_names,
        scaler.mean_,
        scaler.scale_,
    ):
        print(f"{feature:35s} mean={mean:.6f} scale={scale:.6f}")

    if METADATA_PATH.exists():
        print("\nMetadata:")
        print("-" * 80)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(json.dumps(metadata, indent=2))

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
    