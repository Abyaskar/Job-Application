"""
Learned ML ranker using LogisticRegression.

This module implements a supervised machine learning approach to candidate-job
ranking, replacing the manually weighted hybrid score with a model trained on
labeled relevance data.

Architecture:
- Feature extraction from candidate-job pairs (identical for training and inference)
- LogisticRegression classifier predicting relevance probability
- Model persistence via joblib
- Fallback to deterministic hybrid ranker if no trained model exists

The model predicts P(relevant | features), which becomes the ML ranking score.
Eligibility constraints remain separate (see ranking.determine_eligibility).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.logging import get_logger
from app.models.schemas import (
    ExtractedRequirements,
    IntentProfile,
    ParsedJob,
    ParsedResume,
    ScoreBreakdown,
)
from app.services.embeddings import cosine_similarity
from app.services.intent import title_alignment_score
from app.services.ranking import compute_hard_skill_match, compute_education_match, compute_experience_match, compute_location_match

logger = get_logger("services.learned_ranker")

# Default model storage path
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "recommender_v1.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "recommender_v1_metadata.json"


@dataclass(frozen=True)
class CandidateJobFeatures:
    """Feature vector for a candidate-job pair.
    
    All features are numeric and normalized to roughly [0, 1] range where possible.
    This ensures consistent scaling for the LogisticRegression model.
    """
    semantic_similarity: float = 0.0
    intent_alignment: float = 0.0
    required_skill_coverage: float = 0.0
    preferred_skill_coverage: float = 0.0
    matched_required_skill_count: int = 0
    missing_required_skill_count: int = 0
    experience_match: float = 0.0
    education_match: float = 0.0
    location_match: float = 0.0
    seniority_match: float = 0.0
    domain_match: float = 0.0
    resume_experience_years: float = 0.0
    required_experience_years: float = 0.0
    skill_gap_ratio: float = 0.0
    intent_confidence: float = 0.0
    
    def to_array(self, feature_names: list[str] | None = None) -> np.ndarray:
        """Convert to numpy array in the expected order."""
        if feature_names is None:
            feature_names = self.get_feature_names()
        
        values = []
        for name in feature_names:
            val = getattr(self, name, 0.0)
            values.append(float(val))
        return np.array(values, dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> list[str]:
        """Return the canonical feature names in order."""
        return [
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


@dataclass
class LearnedRankerMetadata:
    """Metadata about a trained model."""
    model_version: str
    training_timestamp: str
    n_training_examples: int
    n_positive_examples: int
    n_negative_examples: int
    feature_names: list[str]
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    training_dataset_source: str = "unknown"
    model_type: str = "LogisticRegression"
    
    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "LearnedRankerMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class LearnedRanker:
    """Learned ML ranker using LogisticRegression.
    
    The ranker predicts P(relevant | candidate, job) based on extracted features.
    
    Usage:
        # Training (separate pipeline)
        ranker = LearnedRanker()
        ranker.train(X_train, y_train)
        ranker.save_model(model_path, metadata)
        
        # Inference (production)
        ranker = LearnedRanker.load_model(model_path)
        features = extract_features(resume, job, intent)
        relevance_prob = ranker.predict_proba(features)
    """
    
    def __init__(self):
        self._model: Pipeline | None = None
        self._metadata: LearnedRankerMetadata | None = None
        self._feature_names = CandidateJobFeatures.get_feature_names()
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained and ready for inference."""
        return self._model is not None
    
    @property
    def metadata(self) -> LearnedRankerMetadata | None:
        """Return model metadata if available."""
        return self._metadata
    
    @property
    def feature_names(self) -> list[str]:
        """Return the feature names used by this model."""
        return self._feature_names.copy()
    
    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline with preprocessing and classifier."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            )),
        ])
    
    def train(self, X: np.ndarray, y: np.ndarray, metadata: LearnedRankerMetadata | None = None) -> dict[str, float]:
        """Train the model on labeled examples.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels array of shape (n_samples,), where 1=relevant, 0=not_relevant
            metadata: Optional metadata to store with the model
        
        Returns:
            Dictionary with training metrics (accuracy, precision, recall, etc.)
        """
        if len(X) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        
        n_positive = int(sum(y))
        n_negative = len(y) - n_positive
        
        logger.info("learned_ranker.training_started", 
                    n_samples=len(y), n_positive=n_positive, n_negative=n_negative)
        
        self._model = self._build_pipeline()
        self._model.fit(X, y)
        
        # Compute training metrics
        y_pred = self._model.predict(X)
        y_proba = self._model.predict_proba(X)[:, 1]
        
        accuracy = float(np.mean(y_pred == y))
        
        # Precision/recall for positive class
        tp = sum((y_pred == 1) & (y == 1))
        fp = sum((y_pred == 1) & (y == 0))
        fn = sum((y_pred == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        
        if metadata is None:
            from datetime import datetime
            metadata = LearnedRankerMetadata(
                model_version="v1",
                training_timestamp=datetime.utcnow().isoformat(),
                n_training_examples=len(y),
                n_positive_examples=n_positive,
                n_negative_examples=n_negative,
                feature_names=self._feature_names,
                evaluation_metrics=metrics,
            )
        
        self._metadata = metadata
        self._metadata.evaluation_metrics = metrics
        
        logger.info("learned_ranker.training_completed", **metrics)
        return metrics
    
    def predict_proba(self, features: CandidateJobFeatures) -> float:
        """Predict relevance probability for a candidate-job pair.
        
        Args:
            features: Extracted features for the candidate-job pair
        
        Returns:
            Probability of relevance (0.0 to 1.0)
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        X = features.to_array(self._feature_names).reshape(1, -1)
        proba = self._model.predict_proba(X)[0, 1]
        return float(proba)
    
    def predict_proba_batch(self, features_list: list[CandidateJobFeatures]) -> list[float]:
        """Predict relevance probabilities for multiple candidate-job pairs.
        
        Args:
            features_list: List of feature objects
        
        Returns:
            List of relevance probabilities
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        if not features_list:
            return []
        
        X = np.vstack([f.to_array(self._feature_names) for f in features_list])
        probas = self._model.predict_proba(X)[:, 1]
        return probas.tolist()
    
    def predict(self, features: CandidateJobFeatures, threshold: float = 0.5) -> int:
        """Predict binary relevance label.
        
        Args:
            features: Extracted features for the candidate-job pair
            threshold: Decision threshold (default 0.5)
        
        Returns:
            1 if relevant, 0 if not relevant
        """
        proba = self.predict_proba(features)
        return 1 if proba >= threshold else 0
    
    def save_model(self, model_path: Path | None = None, metadata_path: Path | None = None) -> None:
        """Save the trained model and metadata to disk.
        
        Args:
            model_path: Path to save the model (default: DEFAULT_MODEL_PATH)
            metadata_path: Path to save metadata (default: DEFAULT_METADATA_PATH)
        """
        if self._model is None:
            raise RuntimeError("No model to save. Train first.")
        
        import joblib
        
        model_path = model_path or DEFAULT_MODEL_PATH
        metadata_path = metadata_path or DEFAULT_METADATA_PATH
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, model_path)
        logger.info("learned_ranker.model_saved", path=str(model_path))
        
        if self._metadata is not None:
            self._metadata.save(metadata_path)
            logger.info("learned_ranker.metadata_saved", path=str(metadata_path))
    
    @classmethod
    def load_model(cls, model_path: Path | None = None, metadata_path: Path | None = None) -> "LearnedRanker":
        """Load a trained model from disk.
        
        Args:
            model_path: Path to the model file (default: DEFAULT_MODEL_PATH)
            metadata_path: Path to metadata file (default: DEFAULT_METADATA_PATH)
        
        Returns:
            Loaded LearnedRanker instance
        """
        import joblib
        
        model_path = model_path or DEFAULT_MODEL_PATH
        metadata_path = metadata_path or DEFAULT_METADATA_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        ranker = cls()
        ranker._model = joblib.load(model_path)
        logger.info("learned_ranker.model_loaded", path=str(model_path))
        
        if metadata_path.exists():
            try:
                ranker._metadata = LearnedRankerMetadata.load(metadata_path)
                logger.info("learned_ranker.metadata_loaded", path=str(metadata_path))
            except Exception as e:
                logger.warning("learned_ranker.metadata_load_failed", error=str(e))
        
        return ranker


def extract_candidate_job_features(
    resume: ParsedResume,
    job: ParsedJob,
    intent: IntentProfile | None = None,
) -> CandidateJobFeatures:
    """Extract features from a candidate-job pair.
    
    This function is used for both training and inference, ensuring consistency.
    
    Args:
        resume: Parsed resume with skills, experience, embedding, etc.
        job: Parsed job with requirements, location, embedding, etc.
        intent: Optional career intent profile
    
    Returns:
        CandidateJobFeatures object with all extracted features
    """
    # Semantic similarity from embeddings
    semantic_sim = cosine_similarity(resume.embedding, job.embedding) if resume.embedding and job.embedding else 0.0
    semantic_sim = max(0.0, semantic_sim)  # Floor at 0
    
    # Intent alignment
    intent_align = title_alignment_score(intent, job.title, job.domain) if intent else 0.0
    intent_conf = float(intent.confidence) if intent else 0.0
    
    # Hard skill match
    skill_score, skill_gap = compute_hard_skill_match(resume.skills, job.requirements)
    req_cov = skill_gap.coverage_ratio
    
    # Preferred skill coverage
    candidate_set = set(resume.skills)
    preferred_set = set(job.requirements.preferred_skills)
    pref_cov = len(candidate_set & preferred_set) / len(preferred_set) if preferred_set else 1.0
    
    # Experience match
    exp_match = compute_experience_match(resume.total_experience_years, job.requirements.min_experience_years)
    
    # Education match
    edu_level = max([e.level for e in resume.education], default=0)
    edu_match = compute_education_match(edu_level, job.requirements.education_level_required)
    
    # Location match
    loc_match = compute_location_match(resume.preferred_locations, job.location)
    
    # Seniority match (heuristic based on experience gap)
    req_exp = job.requirements.min_experience_years
    cand_exp = resume.total_experience_years
    if req_exp <= 0:
        seniority_match = 1.0
    elif cand_exp >= req_exp:
        seniority_match = min(1.0, 0.9 + 0.02 * (cand_exp - req_exp))
    else:
        seniority_match = max(0.2, cand_exp / req_exp)
    
    # Domain match (simple keyword overlap in domain/titles)
    job_domain_lower = (job.domain or "").lower()
    job_title_lower = job.title.lower()
    resume_skills_lower = [s.lower() for s in resume.skills]
    
    domain_overlap = 0
    if job_domain_lower:
        domain_tokens = set(job_domain_lower.split())
        skill_tokens = set()
        for s in resume_skills_lower:
            skill_tokens.update(s.split())
        domain_overlap = len(domain_tokens & skill_tokens) / max(len(domain_tokens), 1)
    
    title_overlap = 0
    title_tokens = set(job_title_lower.split())
    for s in resume_skills_lower:
        if s in title_tokens or any(t in s for t in title_tokens):
            title_overlap += 1
    domain_match = min(1.0, (domain_overlap + title_overlap * 0.1) / 2)
    
    # Skill gap ratio
    skill_gap_ratio = float(skill_gap.coverage_ratio)
    
    return CandidateJobFeatures(
        semantic_similarity=round(semantic_sim, 4),
        intent_alignment=round(intent_align, 4),
        required_skill_coverage=round(req_cov, 4),
        preferred_skill_coverage=round(pref_cov, 4),
        matched_required_skill_count=len(skill_gap.matched_required),
        missing_required_skill_count=len(skill_gap.missing_required),
        experience_match=round(exp_match, 4),
        education_match=round(edu_match, 4),
        location_match=round(loc_match, 4),
        seniority_match=round(seniority_match, 4),
        domain_match=round(domain_match, 4),
        resume_experience_years=round(cand_exp, 2),
        required_experience_years=round(req_exp, 2),
        skill_gap_ratio=round(skill_gap_ratio, 4),
        intent_confidence=round(intent_conf, 4),
    )
