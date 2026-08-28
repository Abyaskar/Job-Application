from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_application_repo, get_feedback_repo
from app.core.logging import get_logger
from app.models.schemas import ApplicationRecord, FeedbackIn
from app.repositories.repositories import ApplicationRepository, FeedbackRepository

router = APIRouter(prefix="/feedback", tags=["feedback"])
logger = get_logger("api.feedback")


@router.post("", status_code=201)
async def submit_feedback(
    payload: FeedbackIn,
    repo: FeedbackRepository = Depends(get_feedback_repo),
    app_repo: ApplicationRepository = Depends(get_application_repo),
) -> dict:
    """Captures accept/reject signal on a recommendation.

    This is the training signal for closing the loop described in the
    roadmap ("recommender learns from accepted/rejected recommendations").
    In this project's scope that's implemented as (a) an acceptance-rate
    metric surfaced in /evaluation/summary, and (b) a documented path in
    the README for how these labels would feed a learned re-ranker
    (e.g. logistic regression over the same score components) in a
    follow-up iteration -- explicitly scoped out here to keep the ranking
    layer deterministic and evaluable rather than shipping an undertrained
    model on a handful of feedback events.
    """
    await repo.save(payload)
    if payload.applied:
        await app_repo.upsert(
            ApplicationRecord(candidate_id=payload.candidate_id, job_id=payload.job_id, status="applied")
        )
    logger.info("feedback.recorded", candidate_id=payload.candidate_id, job_id=payload.job_id, accepted=payload.accepted)
    return {"status": "recorded"}


@router.get("/{candidate_id}")
async def get_feedback_history(
    candidate_id: str, repo: FeedbackRepository = Depends(get_feedback_repo)
) -> list[dict]:
    return await repo.list_for_candidate(candidate_id)
