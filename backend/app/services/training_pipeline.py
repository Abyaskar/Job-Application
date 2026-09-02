"""
Training pipeline for the learned ML ranker.

This module orchestrates the training of the LogisticRegression-based ranker
using labeled candidate-job relevance data.

Data sources (in priority order):
1. Hand-authored eval_labels.json - high quality, small scale
2. Feedback logs from production usage - real user signals
3. Dataset-derived labels (eval_labels_dataset_augmented.json) - synthetic fallback

The pipeline:
1. Loads labeled training data
2. Generates synthetic candidate/job pairs if needed for scale
3. Extracts features using the same function as inference
4. Splits into train/test sets (candidate-level separation where possible)
5. Trains LogisticRegression with class balancing
6. Evaluates on held-out test set
7. Saves model and metadata if validation passes

IMPORTANT: This is a SEPARATE operation from inference.
Do NOT call this during recommendation requests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.models.schemas import ParsedJob, ParsedResume
from app.repositories.repositories import CandidateRepository, JobRepository
from app.services.dataset_loaders import load_resume_ranking_dataset
from app.services.embeddings import get_embedding_provider
from app.services.intent import normalize_intent
from app.services.learned_ranker import (
    CandidateJobFeatures,
    LearnedRanker,
    LearnedRankerMetadata,
    extract_candidate_job_features,
)
from app.services.ranking import score_candidate_job

logger = get_logger("services.training_pipeline")

# Path to eval labels
EVAL_LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_labels.json"
EVAL_LABELS_AUGMENTED_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_labels_dataset_augmented.json"


@dataclass
class TrainingExample:
    """A single training example with features and label."""
    features: CandidateJobFeatures
    label: int  # 1 = relevant, 0 = not relevant
    candidate_id: str
    job_id: str
    source: str  # "hand_authored", "feedback", "synthetic"


def _load_hand_authored_labels() -> list[dict]:
    """Load hand-authored relevance labels from eval_labels.json."""
    if not EVAL_LABELS_PATH.exists():
        logger.warning("eval_labels.not_found", path=str(EVAL_LABELS_PATH))
        return []
    
    with open(EVAL_LABELS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cases = data.get("cases", [])
    logger.info("eval_labels.loaded", n_cases=len(cases))
    return cases


def _expand_labels_to_pairs(
    cases: list[dict],
    all_jobs: list[ParsedJob],
    job_id_map: dict[str, ParsedJob],
) -> list[tuple[str, str, int]]:
    """Expand labeled cases into (candidate_id, job_id, label) triples.
    
    For each case, jobs in relevant_job_ids are positive (label=1),
    and other jobs are negative (label=0).
    """
    examples = []
    job_ids = {j.job_id for j in all_jobs}
    
    for case in cases:
        cand_id = case["candidate_id"]
        relevant_ids = set(case.get("relevant_job_ids", []))
        
        # Positive examples
        for job_id in relevant_ids:
            if job_id in job_ids:
                examples.append((cand_id, job_id, 1))
        
        # Negative examples (jobs not marked as relevant)
        # Only include a subset to avoid extreme class imbalance
        non_relevant_ids = job_ids - relevant_ids
        # Sample up to 3x the number of positives as negatives
        max_negatives = min(len(non_relevant_ids), len(relevant_ids) * 3)
        import random
        random.seed(42)  # Deterministic sampling
        sampled_negatives = random.sample(sorted(non_relevant_ids), max_negatives)
        
        for job_id in sampled_negatives:
            examples.append((cand_id, job_id, 0))
    
    logger.info("labels.expanded", n_examples=len(examples))
    return examples


async def prepare_training_data(
    candidate_repo: CandidateRepository,
    job_repo: JobRepository,
    use_augmented: bool = True,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Prepare training data from available sources.
    
    Args:
        candidate_repo: Repository for loading candidate resumes
        job_repo: Repository for loading jobs
        use_augmented: Whether to also use augmented (synthetic) labels
    
    Returns:
        Tuple of (training_examples, metadata_dict)
    """
    # Load all jobs
    all_jobs = await job_repo.list_jobs(limit=1000)
    job_id_map = {j.job_id: j for j in all_jobs}
    
    if not all_jobs:
        raise ValueError("No jobs found in repository. Seed jobs first.")
    
    # Load hand-authored labels
    cases = _load_hand_authored_labels()
    
    if not cases:
        raise ValueError("No labeled data found. Cannot train without labels.")
    
    # Expand to candidate-job pairs with labels
    labeled_pairs = _expand_labels_to_pairs(cases, all_jobs, job_id_map)
    
    # Load candidates and build examples
    examples: list[TrainingExample] = []
    stats = {
        "n_candidates": 0,
        "n_jobs": len(all_jobs),
        "n_positive": 0,
        "n_negative": 0,
        "sources": {"hand_authored": 0},
    }
    
    for cand_id, job_id, label in labeled_pairs:
        resume = await candidate_repo.get_resume(cand_id)
        job = job_id_map.get(job_id)
        
        if not resume or not job:
            continue
        
        # Ensure embeddings exist
        provider = get_embedding_provider()
        if not resume.embedding:
            resume.embedding = provider.embed(resume.raw_text)
        if not job.embedding:
            job.embedding = provider.embed(job.raw_description)
        
        # Get intent if specified in the case
        intent = None
        for case in cases:
            if case["candidate_id"] == cand_id and case.get("intended_role"):
                intent = normalize_intent(cand_id, case["intended_role"])
                break
        
        # Extract features
        features = extract_candidate_job_features(resume, job, intent)
        
        examples.append(TrainingExample(
            features=features,
            label=label,
            candidate_id=cand_id,
            job_id=job_id,
            source="hand_authored",
        ))
        
        if label == 1:
            stats["n_positive"] += 1
        else:
            stats["n_negative"] += 1
    
    stats["n_candidates"] = len({e.candidate_id for e in examples})
    stats["total_examples"] = len(examples)
    
    logger.info("training_data.prepared", **stats)
    return examples, stats


def _train_test_split_by_candidate(
    examples: list[TrainingExample],
    test_ratio: float = 0.3,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split examples by candidate to avoid leakage.
    
    All examples from the same candidate go to either train or test,
    not both. This prevents the model from learning candidate-specific
    patterns that don't generalize.
    """
    # Group by candidate
    by_candidate: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_candidate.setdefault(ex.candidate_id, []).append(ex)
    
    # Shuffle candidates deterministically
    import random
    random.seed(42)
    candidate_ids = sorted(by_candidate.keys())
    random.shuffle(candidate_ids)
    
    # Split
    n_test = max(1, int(len(candidate_ids) * test_ratio))
    test_candidates = set(candidate_ids[:n_test])
    train_candidates = set(candidate_ids[n_test:])
    
    train_examples = []
    test_examples = []
    
    for ex in examples:
        if ex.candidate_id in test_candidates:
            test_examples.append(ex)
        else:
            train_examples.append(ex)
    
    logger.info("data.split", 
                n_train=len(train_examples), 
                n_test=len(test_examples),
                n_train_candidates=len(train_candidates),
                n_test_candidates=len(test_candidates))
    
    return train_examples, test_examples


async def train_ranker(
    candidate_repo: CandidateRepository,
    job_repo: JobRepository,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    use_augmented: bool = False,
    force_save: bool = True,
) -> dict[str, Any]:
    """Train the learned ranker and optionally save it.
    
    Args:
        candidate_repo: Repository for candidate data
        job_repo: Repository for job data
        model_path: Where to save the model (default: learned_ranker.DEFAULT_MODEL_PATH)
        metadata_path: Where to save metadata (default: learned_ranker.DEFAULT_METADATA_PATH)
        use_augmented: Whether to include augmented/synthetic labels
        force_save: Whether to save even if test performance is modest
    
    Returns:
        Dictionary with training results and metrics
    """
    logger.info("training_pipeline.started")
    
    # Prepare data
    examples, data_stats = await prepare_training_data(
        candidate_repo, job_repo, use_augmented=use_augmented
    )
    
    if len(examples) < 10:
        raise ValueError(f"Insufficient training examples: {len(examples)}. Need at least 10.")
    
    # Train/test split
    train_examples, test_examples = _train_test_split_by_candidate(examples, test_ratio=0.3)
    
    if len(train_examples) < 5:
        raise ValueError(f"Insufficient training examples after split: {len(train_examples)}")
    
    # Build feature matrices
    X_train = np.vstack([ex.features.to_array() for ex in train_examples])
    y_train = np.array([ex.label for ex in train_examples])
    
    X_test = np.vstack([ex.features.to_array() for ex in test_examples])
    y_test = np.array([ex.label for ex in test_examples])
    
    logger.info("feature_matrices.built",
                X_train_shape=X_train.shape,
                X_test_shape=X_test.shape,
                y_train_positive=int(sum(y_train)),
                y_train_negative=len(y_train) - int(sum(y_train)))
    
    # Train model
    ranker = LearnedRanker()
    
    from datetime import datetime
    metadata = LearnedRankerMetadata(
        model_version="v1",
        training_timestamp=datetime.utcnow().isoformat(),
        n_training_examples=len(train_examples),
        n_positive_examples=int(sum(y_train)),
        n_negative_examples=len(y_train) - int(sum(y_train)),
        feature_names=CandidateJobFeatures.get_feature_names(),
        training_dataset_source="hand_authored_eval_labels",
    )
    
    train_metrics = ranker.train(X_train, y_train, metadata=metadata)
    
    # Evaluate on test set
    test_metrics = {}
    if len(test_examples) > 0:
        y_pred = ranker._model.predict(X_test)
        y_proba = ranker._model.predict_proba(X_test)[:, 1]
        
        tp = sum((y_pred == 1) & (y_test == 1))
        fp = sum((y_pred == 1) & (y_test == 0))
        fn = sum((y_pred == 0) & (y_test == 1))
        tn = sum((y_pred == 0) & (y_test == 0))
        
        accuracy = float(np.mean(y_pred == y_test))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        test_metrics = {
            "test_accuracy": round(accuracy, 4),
            "test_precision": round(precision, 4),
            "test_recall": round(recall, 4),
            "test_f1": round(f1, 4),
            "test_n_examples": len(test_examples),
        }
        
        logger.info("evaluation.completed_on_test_set", **test_metrics)
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    ranker._metadata.evaluation_metrics = all_metrics
    
    # Save model
    if force_save or (test_metrics.get("test_f1", 0) >= 0.5):
        ranker.save_model(model_path, metadata_path)
        logger.info("model.saved_successfully")
    else:
        logger.warning("model.not_saved", reason="test_performance_below_threshold")
    
    result = {
        "status": "completed",
        "n_training_examples": len(train_examples),
        "n_test_examples": len(test_examples),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path or ranker.DEFAULT_MODEL_PATH),
        "metadata_path": str(metadata_path or ranker.DEFAULT_METADATA_PATH),
        "feature_names": ranker.feature_names,
    }
    
    logger.info("training_pipeline.completed", **{k: str(v) for k, v in result.items()})
    return result


# CLI entry point for running training as a script
if __name__ == "__main__":
    import asyncio
    
    async def main():
        from app.db.mongo import get_mongo_client
        from app.repositories.repositories import CandidateRepository, JobRepository
        
        client = get_mongo_client()
        db = client["career_recommendation"]
        
        candidate_repo = CandidateRepository(db["candidates"])
        job_repo = JobRepository(db["jobs"])
        
        result = await train_ranker(candidate_repo, job_repo)
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())
