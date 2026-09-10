"""
Training pipeline for the learned ML ranker.

IMPORTANT:

This pipeline is completely separate from inference.

It trains ONLY from real supervised learning examples stored
in MongoDB.

Flow:

    learning_examples
            ↓
    validate feature schema
            ↓
    candidate-level train/test split
            ↓
    LogisticRegression
            ↓
    evaluation
            ↓
    model persistence

A resume upload alone is NOT a training example.

A training example is created when a REAL candidate interacts
with a job and produces an observable outcome such as:

    applied   -> label 1
    accepted  -> label 1
    rejected  -> label 0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.db.mongo import get_db
from app.repositories.repository import (
    TrainingExampleRepository,
)
from app.services.learned_ranker import (
    CandidateJobFeatures,
    LearnedRanker,
    LearnedRankerMetadata,
)


logger = get_logger(
    "services.training_pipeline"
)


# ============================================================
# TRAINING EXAMPLE
# ============================================================

@dataclass
class TrainingExample:
    """
    In-memory representation of one supervised example.
    """

    features: CandidateJobFeatures

    label: int

    candidate_id: str

    job_id: str

    source: str


# ============================================================
# RECONSTRUCT FEATURES
# ============================================================

def _features_from_document(
    document: dict,
) -> CandidateJobFeatures | None:
    """
    Convert the stored numeric feature array back into the
    CandidateJobFeatures dataclass.
    """

    expected_names = (
        CandidateJobFeatures
        .get_feature_names()
    )

    stored_names = document.get(
        "feature_names",
        [],
    )

    # --------------------------------------------------------
    # Protect against feature schema changes
    # --------------------------------------------------------

    if stored_names != expected_names:

        logger.warning(
            "training.example_skipped",
            example_id=document.get(
                "example_id"
            ),
            reason="feature_names_mismatch",
        )

        return None

    values = document.get(
        "features"
    )

    if not isinstance(
        values,
        list,
    ):

        return None

    if len(values) != len(
        expected_names
    ):

        logger.warning(
            "training.example_skipped",
            example_id=document.get(
                "example_id"
            ),
            reason="feature_length_mismatch",
        )

        return None

    try:

        feature_dict = {
            name: float(value)
            for name, value in zip(
                expected_names,
                values,
            )
        }

        return CandidateJobFeatures(
            **feature_dict
        )

    except Exception as exc:

        logger.warning(
            "training.example_invalid",
            example_id=document.get(
                "example_id"
            ),
            error=str(exc),
        )

        return None


# ============================================================
# LOAD REAL TRAINING DATA
# ============================================================

async def load_real_training_examples(
    training_repo: TrainingExampleRepository,
) -> list[TrainingExample]:

    documents = (
        await training_repo.list_examples()
    )

    examples: list[
        TrainingExample
    ] = []

    for document in documents:

        try:

            features = (
                _features_from_document(
                    document
                )
            )

            if features is None:
                continue

            label = int(
                document.get(
                    "label",
                    0,
                )
            )

            if label not in (0, 1):

                logger.warning(
                    "training.example_skipped",
                    example_id=document.get(
                        "example_id"
                    ),
                    reason="invalid_label",
                )

                continue

            candidate_id = document.get(
                "candidate_id"
            )

            job_id = document.get(
                "job_id"
            )

            if not candidate_id or not job_id:

                continue

            examples.append(
                TrainingExample(
                    features=features,
                    label=label,
                    candidate_id=candidate_id,
                    job_id=job_id,
                    source=document.get(
                        "source",
                        "feedback",
                    ),
                )
            )

        except Exception as exc:

            logger.warning(
                "training.example_invalid",
                example_id=document.get(
                    "example_id"
                ),
                error=str(exc),
            )

    logger.info(
        "training.real_examples_loaded",
        n_examples=len(examples),
    )

    return examples


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def _train_test_split_by_candidate(
    examples: list[TrainingExample],
    test_ratio: float = 0.30,
) -> tuple[
    list[TrainingExample],
    list[TrainingExample],
]:

    """
    Split by candidate rather than by individual rows.

    This prevents the same candidate appearing in both
    training and testing data.
    """

    if not examples:

        return [], []

    # --------------------------------------------------------
    # Group examples by candidate
    # --------------------------------------------------------

    by_candidate: dict[
        str,
        list[TrainingExample],
    ] = {}

    for example in examples:

        by_candidate.setdefault(
            example.candidate_id,
            [],
        ).append(example)

    candidate_ids = sorted(
        by_candidate.keys()
    )

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed=42
    )

    rng.shuffle(
        candidate_ids
    )

    # --------------------------------------------------------
    # If only one candidate exists,
    # don't create a fake candidate split.
    # --------------------------------------------------------

    if len(candidate_ids) < 2:

        logger.warning(
            "training.split_not_possible",
            reason="less_than_two_candidates",
        )

        return examples, []

    n_test = max(
        1,
        int(
            len(candidate_ids)
            * test_ratio
        ),
    )

    # Make sure at least one candidate
    # remains for training.
    n_test = min(
        n_test,
        len(candidate_ids) - 1,
    )

    test_candidates = set(
        candidate_ids[:n_test]
    )

    train_candidates = set(
        candidate_ids[n_test:]
    )

    train_examples = []
    test_examples = []

    for example in examples:

        if (
            example.candidate_id
            in test_candidates
        ):

            test_examples.append(
                example
            )

        else:

            train_examples.append(
                example
            )

    logger.info(
        "training.data_split",
        n_train=len(train_examples),
        n_test=len(test_examples),
        n_train_candidates=len(
            train_candidates
        ),
        n_test_candidates=len(
            test_candidates
        ),
    )

    return (
        train_examples,
        test_examples,
    )


# ============================================================
# EVALUATION
# ============================================================

def _evaluate_model(
    ranker: LearnedRanker,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:

    if len(y_test) == 0:

        return {}

    y_pred = (
        ranker._model.predict(
            X_test
        )
    )

    tp = int(
        np.sum(
            (y_pred == 1)
            & (y_test == 1)
        )
    )

    fp = int(
        np.sum(
            (y_pred == 1)
            & (y_test == 0)
        )
    )

    fn = int(
        np.sum(
            (y_pred == 0)
            & (y_test == 1)
        )
    )

    accuracy = float(
        np.mean(
            y_pred == y_test
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "test_accuracy": round(
            accuracy,
            4,
        ),
        "test_precision": round(
            precision,
            4,
        ),
        "test_recall": round(
            recall,
            4,
        ),
        "test_f1": round(
            f1,
            4,
        ),
        "test_n_examples": len(
            y_test
        ),
    }


# ============================================================
# TRAIN RANKER
# ============================================================

async def train_ranker(
    training_repo: TrainingExampleRepository,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    force_save: bool = True,
) -> dict[str, Any]:

    """
    Train the learned ranker from real feedback examples.
    """

    logger.info(
        "training_pipeline.started"
    )

    # --------------------------------------------------------
    # Load real learning examples
    # --------------------------------------------------------

    examples = (
        await load_real_training_examples(
            training_repo
        )
    )

    if not examples:

        raise ValueError(
            "No real training examples found. "
            "Users must interact with recommendations "
            "before the learned ranker can be trained."
        )

    # --------------------------------------------------------
    # Check class diversity
    # --------------------------------------------------------

    labels = [
        example.label
        for example in examples
    ]

    n_positive = sum(
        label == 1
        for label in labels
    )

    n_negative = sum(
        label == 0
        for label in labels
    )

    if n_positive == 0 or n_negative == 0:

        raise ValueError(
            "Training requires both positive and negative "
            "examples. Current data contains "
            f"{n_positive} positive and "
            f"{n_negative} negative examples."
        )

    # --------------------------------------------------------
    # Candidate-level split
    # --------------------------------------------------------

    (
        train_examples,
        test_examples,
    ) = _train_test_split_by_candidate(
        examples,
        test_ratio=0.30,
    )

    if not train_examples:

        raise ValueError(
            "No training examples available "
            "after candidate-level split."
        )

    # --------------------------------------------------------
    # Feature matrices
    # --------------------------------------------------------

    X_train = np.vstack(
        [
            example.features.to_array()
            for example in train_examples
        ]
    )

    y_train = np.array(
        [
            example.label
            for example in train_examples
        ],
        dtype=np.int64,
    )

    if test_examples:

        X_test = np.vstack(
            [
                example.features.to_array()
                for example in test_examples
            ]
        )

        y_test = np.array(
            [
                example.label
                for example in test_examples
            ],
            dtype=np.int64,
        )

    else:

        X_test = np.empty(
            (
                0,
                X_train.shape[1],
            )
        )

        y_test = np.array(
            [],
            dtype=np.int64,
        )

    logger.info(
        "training.feature_matrices_built",
        X_train_shape=X_train.shape,
        X_test_shape=X_test.shape,
        train_positive=int(
            np.sum(y_train == 1)
        ),
        train_negative=int(
            np.sum(y_train == 0)
        ),
    )

    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    metadata = LearnedRankerMetadata(
        model_version="v1",
        training_timestamp=(
            datetime.utcnow().isoformat()
        ),
        n_training_examples=len(
            train_examples
        ),
        n_positive_examples=int(
            np.sum(y_train == 1)
        ),
        n_negative_examples=int(
            np.sum(y_train == 0)
        ),
        feature_names=(
            CandidateJobFeatures
            .get_feature_names()
        ),
        training_dataset_source=(
            "real_user_feedback"
        ),
    )

    # --------------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------------

    ranker = LearnedRanker()

    train_metrics = ranker.train(
        X_train,
        y_train,
        metadata=metadata,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    test_metrics = (
        _evaluate_model(
            ranker,
            X_test,
            y_test,
        )
    )

    # --------------------------------------------------------
    # Combine metrics
    # --------------------------------------------------------

    all_metrics = {
        **train_metrics,
        **test_metrics,
    }

    if ranker._metadata is not None:

        ranker._metadata.evaluation_metrics = (
            all_metrics
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    if force_save or (
        test_metrics.get(
            "test_f1",
            0.0,
        ) >= 0.5
    ):

        ranker.save_model(
            model_path,
            metadata_path,
        )

        logger.info(
            "training.model_saved"
        )

    else:

        logger.warning(
            "training.model_not_saved",
            reason=(
                "test_f1_below_threshold"
            ),
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "status": "completed",

        "n_total_examples": len(
            examples
        ),

        "n_training_examples": len(
            train_examples
        ),

        "n_test_examples": len(
            test_examples
        ),

        "n_positive_examples": int(
            np.sum(y_train == 1)
        ),

        "n_negative_examples": int(
            np.sum(y_train == 0)
        ),

        "train_metrics": train_metrics,

        "test_metrics": test_metrics,

        "training_dataset_source": (
            "real_user_feedback"
        ),

        "model_path": str(
            model_path
            or LearnedRanker.DEFAULT_MODEL_PATH
        ),

        "metadata_path": str(
            metadata_path
            or LearnedRanker.DEFAULT_METADATA_PATH
        ),

        "feature_names": (
            CandidateJobFeatures
            .get_feature_names()
        ),
    }

    logger.info(
        "training_pipeline.completed",
        n_total_examples=len(
            examples
        ),
        n_training_examples=len(
            train_examples
        ),
        n_test_examples=len(
            test_examples
        ),
    )

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import asyncio
    import json

    async def main():

        db = get_db()

        training_repo = (
            TrainingExampleRepository(db)
        )

        result = await train_ranker(
            training_repo
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    asyncio.run(
        main()
    )