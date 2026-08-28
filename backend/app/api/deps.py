from __future__ import annotations

from app.core.config import Settings, get_settings
from app.db.mongo import get_db
from app.repositories.repositories import (
    ApplicationRepository,
    CandidateRepository,
    FeedbackRepository,
    IntentRepository,
    JobRepository,
    RecommendationRepository,
)


def get_candidate_repo() -> CandidateRepository:
    return CandidateRepository(get_db())


def get_job_repo() -> JobRepository:
    return JobRepository(get_db())


def get_recommendation_repo() -> RecommendationRepository:
    return RecommendationRepository(get_db())


def get_feedback_repo() -> FeedbackRepository:
    return FeedbackRepository(get_db())


def get_application_repo() -> ApplicationRepository:
    return ApplicationRepository(get_db())


def get_intent_repo() -> IntentRepository:
    return IntentRepository(get_db())


def get_app_settings() -> Settings:
    return get_settings()
