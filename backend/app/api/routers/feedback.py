from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from app.api.deps import (
    get_application_repo,
    get_candidate_repo,
    get_feedback_repo,
    get_intent_repo,
    get_job_repo,
    get_learning_candidate_repo,
    get_training_example_repo,
)

from app.core.logging import get_logger

from app.models.schemas import (
    ApplicationRecord,
    FeedbackIn,
)

from app.repositories.repository import (
    ApplicationRepository,
    CandidateRepository,
    FeedbackRepository,
    IntentRepository,
    JobRepository,
    LearningCandidateRepository,
    TrainingExampleRepository,
)

from app.services.learned_ranker import (
    CandidateJobFeatures,
    extract_candidate_job_features,
)


router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
)

logger = get_logger(
    "api.feedback"
)


@router.post(
    "",
    status_code=201,
)
async def submit_feedback(
    payload: FeedbackIn,

    repo: FeedbackRepository = Depends(
        get_feedback_repo
    ),

    app_repo: ApplicationRepository = Depends(
        get_application_repo
    ),

    candidate_repo: CandidateRepository = Depends(
        get_candidate_repo
    ),

    job_repo: JobRepository = Depends(
        get_job_repo
    ),

    intent_repo: IntentRepository = Depends(
        get_intent_repo
    ),

    learning_candidate_repo: LearningCandidateRepository = Depends(
        get_learning_candidate_repo
    ),

    training_repo: TrainingExampleRepository = Depends(
        get_training_example_repo
    ),
) -> dict:

    # ========================================================
    # 1. ALWAYS save normal feedback
    # ========================================================

    await repo.save(
        payload
    )

    # ========================================================
    # 2. Save application if applicable
    # ========================================================

    if payload.applied:

        await app_repo.upsert(
            ApplicationRecord(
                candidate_id=payload.candidate_id,
                job_id=payload.job_id,
                status="applied",
            )
        )

    # ========================================================
    # 3. Find candidate
    # ========================================================

    resume = await candidate_repo.get_resume(
        payload.candidate_id
    )

    if not resume:

        raise HTTPException(
            status_code=404,
            detail="Candidate resume not found.",
        )

    # ========================================================
    # 4. Find job
    # ========================================================

    job = await job_repo.get_job(
        payload.job_id
    )

    if not job:

        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    # ========================================================
    # 5. IMPORTANT:
    # Demo candidates NEVER enter ML training
    # ========================================================

    training_eligible = (
        await learning_candidate_repo
        .is_training_eligible(
            payload.candidate_id
        )
    )

    if not training_eligible:

        logger.info(
            "feedback.demo_candidate",
            candidate_id=payload.candidate_id,
            job_id=payload.job_id,
        )

        return {
            "status": "recorded",
            "training_example_created": False,
            "reason": "demo_or_non_learning_candidate",
        }

    # ========================================================
    # 6. Get candidate intent
    # ========================================================

    intent = await intent_repo.get(
        payload.candidate_id
    )

    # ========================================================
    # 7. Extract SAME features used during recommendation
    # ========================================================

    features = extract_candidate_job_features(
        resume,
        job,
        intent,
    )

    # ========================================================
    # 8. Convert behavior into supervised label
    # ========================================================

    if payload.accepted:

        label = 1
        outcome = "accepted"

    elif payload.applied:

        label = 1
        outcome = "applied"

    else:

        label = 0
        outcome = "rejected"

    # ========================================================
    # 9. Store training example
    # ========================================================

    example_id = await training_repo.save(
        candidate_id=payload.candidate_id,
        job_id=payload.job_id,
        label=label,
        features=features.to_array().tolist(),
        feature_names=(
            CandidateJobFeatures
            .get_feature_names()
        ),
        source="feedback",
        outcome=outcome,
    )

    logger.info(
        "learning.example_created",
        example_id=example_id,
        candidate_id=payload.candidate_id,
        job_id=payload.job_id,
        label=label,
        outcome=outcome,
    )

    return {
        "status": "recorded",
        "training_example_created": True,
        "training_example_id": example_id,
        "label": label,
        "outcome": outcome,
    }