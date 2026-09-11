from __future__ import annotations

from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from app.api.deps import (
    get_candidate_repo,
    get_learning_candidate_repo,
)

from app.core.logging import get_logger

from app.models.schemas import (
    DocumentValidationState,
    ParsedResume,
    ResumeIn,
)

from app.repositories.repository import (
    CandidateRepository,
    LearningCandidateRepository,
)

from app.services.document_intelligence import (
    process_uploaded_document,
)

from app.services.extraction import (
    extract_resume_profile,
)

from app.services.embeddings import (
    embed_text,
)


router = APIRouter(
    prefix="/candidates",
    tags=["candidates"],
)

logger = get_logger(
    "api.candidates"
)


# ============================================================
# RESUME TEXT INGESTION
# ============================================================

@router.post(
    "/resume",
    response_model=ParsedResume,
    status_code=201,
)
async def ingest_resume(
    payload: ResumeIn,
    repo: CandidateRepository = Depends(
        get_candidate_repo
    ),
    learning_repo: LearningCandidateRepository = Depends(
        get_learning_candidate_repo
    ),
) -> ParsedResume:

    """
    Ingest already-extracted resume text.

    This path is also considered a genuine candidate
    ingestion path, therefore the candidate is stored
    in learning_candidates.
    """

    if len(
        payload.raw_text.strip()
    ) < 20:

        raise HTTPException(
            status_code=422,
            detail=(
                "Resume text is too short to "
                "extract a meaningful profile."
            ),
        )

    # --------------------------------------------------------
    # NLP extraction
    # --------------------------------------------------------

    profile = extract_resume_profile(
        payload.raw_text
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    try:

        embedding = embed_text(
            payload.raw_text
        )

    except Exception as exc:

        logger.warning(
            "resume.embedding_failed",
            error=str(exc),
        )

        embedding = []

    # --------------------------------------------------------
    # Build ParsedResume
    # --------------------------------------------------------

    resume = ParsedResume(
        candidate_id=payload.candidate_id,
        raw_text=payload.raw_text,
        preferred_locations=payload.preferred_locations,
        preferred_domains=payload.preferred_domains,
        current_location=payload.current_location,
        preferred_countries=payload.preferred_countries,
        open_to_remote=payload.open_to_remote,
        open_to_relocation=payload.open_to_relocation,
        embedding=embedding,
        **profile,
    )

    # --------------------------------------------------------
    # Store normal candidate
    # --------------------------------------------------------

    await repo.upsert_resume(
        resume
    )

    # --------------------------------------------------------
    # Store as REAL learning candidate
    # --------------------------------------------------------

    await learning_repo.save_real_candidate(
        candidate_id=resume.candidate_id,
        resume=resume,
    )

    logger.info(
        "resume.ingested",
        candidate_id=resume.candidate_id,
        n_skills=len(resume.skills),
        learning_eligible=True,
    )

    return resume


# ============================================================
# FILE UPLOAD
# ============================================================

@router.post(
    "/resume/upload",
    response_model=ParsedResume,
    status_code=201,
)
async def upload_resume(
    file: UploadFile = File(...),

    repo: CandidateRepository = Depends(
        get_candidate_repo
    ),

    learning_repo: LearningCandidateRepository = Depends(
        get_learning_candidate_repo
    ),

) -> ParsedResume:

    """
    Upload a resume file.

    Flow:

        upload
          ↓
        document validation
          ↓
        text extraction
          ↓
        NLP
          ↓
        embedding
          ↓
        candidates
          ↓
        learning_candidates
    """

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_bytes) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="Resume must be smaller than 10 MB.",
        )

    # --------------------------------------------------------
    # Document Intelligence
    # --------------------------------------------------------

    try:

        validation_result = (
            await process_uploaded_document(
                file_bytes=file_bytes,
                filename=file.filename,
            )
        )

    except Exception as exc:

        logger.exception(
            "resume.processing_error",
            filename=file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while "
                "processing your resume."
            ),
        ) from exc

    # --------------------------------------------------------
    # Validation failures
    # --------------------------------------------------------

    if (
        validation_result.state
        == DocumentValidationState.CORRUPTED_FILE
    ):

        raise HTTPException(
            status_code=422,
            detail=validation_result.user_message,
        )

    if (
        validation_result.state
        == DocumentValidationState.UNSUPPORTED_DOCUMENT
    ):

        raise HTTPException(
            status_code=415,
            detail=validation_result.user_message,
        )

    if (
        validation_result.state
        == DocumentValidationState.RESUME_REQUIRES_OCR
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                f"{validation_result.user_message} "
                f"{validation_result.next_action}"
            ),
        )

    if (
        validation_result.state
        == DocumentValidationState.LOW_EXTRACTION_QUALITY
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                f"{validation_result.user_message} "
                f"{validation_result.next_action}"
            ),
        )

    if (
        validation_result.state
        == DocumentValidationState.NOT_A_RESUME
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                f"{validation_result.user_message} "
                f"{validation_result.next_action}"
            ),
        )

    if (
        validation_result.state
        == DocumentValidationState.INSUFFICIENT_INFORMATION
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                f"{validation_result.user_message} "
                f"{validation_result.next_action}"
            ),
        )

    if (
        validation_result.state
        == DocumentValidationState.EXTRACTION_FAILED
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "The resume file could not be processed. "
                "Please try another file."
            ),
        )

    # --------------------------------------------------------
    # Valid resume
    # --------------------------------------------------------

    raw_text = validation_result.raw_text

    candidate_id = (
        f"cand_upload_{uuid4().hex[:12]}"
    )

    # --------------------------------------------------------
    # NLP
    # --------------------------------------------------------

    profile = extract_resume_profile(
        raw_text
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    try:

        embedding = embed_text(
            raw_text
        )

    except Exception as exc:

        logger.warning(
            "resume.embedding_failed",
            error=str(exc),
        )

        embedding = []

    # --------------------------------------------------------
    # ParsedResume
    # --------------------------------------------------------

    resume = ParsedResume(
        candidate_id=candidate_id,
        raw_text=raw_text,
        preferred_locations=[],
        preferred_domains=[],

        validation_state=validation_result.state,

        document_type=(
            validation_result.document_type
        ),

        extraction_quality_score=(
            validation_result.extraction_quality_score
        ),

        detected_name=(
            validation_result.detected_name
        ),

        detected_email=(
            validation_result.detected_email
        ),

        detected_phone=(
            validation_result.detected_phone
        ),

        current_location=(
            validation_result.detected_location
        ),

        processing_metadata={
            "file_type_detected": (
                validation_result.file_type_detected
            ),

            "file_type_expected": (
                validation_result.file_type_expected
            ),

            "extraction_warnings": (
                validation_result
                .technical_details
                .get("warnings", [])
            ),
        },

        embedding=embedding,

        **profile,
    )

    # --------------------------------------------------------
    # Store normal candidate
    # --------------------------------------------------------

    await repo.upsert_resume(
        resume
    )

    # --------------------------------------------------------
    # Store REAL candidate in learning population
    # --------------------------------------------------------

    await learning_repo.save_real_candidate(
        candidate_id=candidate_id,
        resume=resume,
    )

    logger.info(
        "resume.uploaded",
        candidate_id=candidate_id,
        filename=file.filename,
        n_skills=len(resume.skills),
        validation_state=(
            validation_result.state.value
        ),
        learning_eligible=True,
    )

    return resume


# ============================================================
# GET RESUME
# ============================================================

@router.get(
    "/{candidate_id}/resume",
    response_model=ParsedResume,
)
async def get_resume(
    candidate_id: str,

    repo: CandidateRepository = Depends(
        get_candidate_repo
    ),

) -> ParsedResume:

    resume = await repo.get_resume(
        candidate_id
    )

    if not resume:

        raise HTTPException(
            status_code=404,
            detail="Candidate resume not found.",
        )

    return resume