from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import (
    get_app_settings,
    get_candidate_repo,
    get_intent_repo,
    get_job_repo,
    get_recommendation_repo,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.models.schemas import RankRequest, Recommendation, SearchMode
from app.repositories.repositories import (
    CandidateRepository,
    IntentRepository,
    JobRepository,
    RecommendationRepository,
)
from app.services.recommender import rank_jobs_for_candidate

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
logger = get_logger("api.recommendations")


@router.post("/rank", response_model=list[Recommendation])
async def rank(
    payload: RankRequest,
    candidate_repo: CandidateRepository = Depends(get_candidate_repo),
    job_repo: JobRepository = Depends(get_job_repo),
    rec_repo: RecommendationRepository = Depends(get_recommendation_repo),
    intent_repo: IntentRepository = Depends(get_intent_repo),
    settings: Settings = Depends(get_app_settings),
) -> list[Recommendation]:
    resume = await candidate_repo.get_resume(payload.candidate_id)
    if not resume:
        raise HTTPException(404, "Candidate resume not found. Ingest a resume first.")

    intent = await intent_repo.get(payload.candidate_id) if payload.use_intent else None

    recommendations, meta = await rank_jobs_for_candidate(
        resume=resume,
        job_repo=job_repo,
        rec_repo=rec_repo,
        settings=settings,
        top_k=payload.top_k,
        mode=payload.mode,
        location_filter=payload.location_filter,
        domain_filter=payload.domain_filter,
        intent=intent,
    )
    logger.info(
        "recommendations.ranked",
        candidate_id=payload.candidate_id,
        role_family=intent.role_family if intent else None,
        **meta,
    )
    return recommendations


@router.get("/{candidate_id}", response_model=list[Recommendation])
async def get_recommendations(
    candidate_id: str, rec_repo: RecommendationRepository = Depends(get_recommendation_repo)
) -> list[Recommendation]:
    return await rec_repo.list_for_candidate(candidate_id)


@router.get("/{candidate_id}/{job_id}", response_model=Recommendation)
async def get_recommendation_detail(
    candidate_id: str,
    job_id: str,
    rec_repo: RecommendationRepository = Depends(get_recommendation_repo),
) -> Recommendation:
    rec = await rec_repo.get(candidate_id, job_id)
    if not rec:
        raise HTTPException(404, "No recommendation found for this candidate/job pair. Call /rank first.")
    return rec


@router.post("/compare")
async def compare_search_modes(
    payload: RankRequest,
    candidate_repo: CandidateRepository = Depends(get_candidate_repo),
    job_repo: JobRepository = Depends(get_job_repo),
    rec_repo: RecommendationRepository = Depends(get_recommendation_repo),
    intent_repo: IntentRepository = Depends(get_intent_repo),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """Runs the same candidate through keyword/vector/hybrid ranking so the
    UI (or evaluation script) can render a side-by-side comparison."""
    resume = await candidate_repo.get_resume(payload.candidate_id)
    if not resume:
        raise HTTPException(404, "Candidate resume not found.")

    intent = await intent_repo.get(payload.candidate_id) if payload.use_intent else None

    results = {}
    for mode in (SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID):
        recs, meta = await rank_jobs_for_candidate(
            resume=resume,
            job_repo=job_repo,
            rec_repo=rec_repo,
            settings=settings,
            top_k=payload.top_k,
            mode=mode,
            use_cache=False,
            intent=intent,
        )
        results[mode.value] = {
            "recommendations": [r.model_dump(mode="json") for r in recs],
            "latency_ms": meta["latency_ms"],
        }
    return results
