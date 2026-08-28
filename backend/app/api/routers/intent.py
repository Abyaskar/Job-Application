from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_intent_repo
from app.core.logging import get_logger
from app.models.schemas import IntentIn, IntentProfile
from app.repositories.repositories import IntentRepository
from app.services.intent import normalize_intent

router = APIRouter(prefix="/intent", tags=["intent"])
logger = get_logger("api.intent")


@router.post("", response_model=IntentProfile, status_code=201)
async def submit_intent(payload: IntentIn, repo: IntentRepository = Depends(get_intent_repo)) -> IntentProfile:
    """Normalize free-text career intent (e.g. 'GenAI Engineer') into a role
    family, related titles, skills, and keywords, and store it as the
    candidate's active intent for use in ranking (see RankRequest.use_intent).
    """
    profile = normalize_intent(payload.candidate_id, payload.free_text)
    await repo.upsert(profile)
    logger.info(
        "intent.submitted",
        candidate_id=payload.candidate_id,
        role_family=profile.role_family,
        method=profile.match_method.value,
    )
    return profile


@router.get("/preview", response_model=IntentProfile)
async def preview_intent(text: str = Query(..., min_length=2)) -> IntentProfile:
    """Normalize free text without persisting it — used by the frontend to
    show a live preview ('did you mean: GenAI Engineer?') before the
    candidate confirms their intent.
    """
    return normalize_intent(candidate_id="__preview__", free_text=text)


@router.get("/{candidate_id}", response_model=IntentProfile)
async def get_intent(candidate_id: str, repo: IntentRepository = Depends(get_intent_repo)) -> IntentProfile:
    intent = await repo.get(candidate_id)
    if not intent:
        raise HTTPException(404, "No career intent set for this candidate yet. Call POST /intent first.")
    return intent
