from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_candidate_repo
from app.core.logging import get_logger
from app.models.schemas import (
    DocumentValidationState,
    ParsedResume,
    ResumeIn,
)
from app.repositories.repositories import CandidateRepository
from app.services.document_intelligence import process_uploaded_document
from app.services.extraction import extract_resume_profile
from app.services.embeddings import embed_text


router = APIRouter(prefix="/candidates", tags=["candidates"])
logger = get_logger("api.candidates")


@router.post("/resume", response_model=ParsedResume, status_code=201)
async def ingest_resume(
    payload: ResumeIn, repo: CandidateRepository = Depends(get_candidate_repo)
) -> ParsedResume:
    """Ingest resume text and run structured extraction."""
    if len(payload.raw_text.strip()) < 20:
        raise HTTPException(
            422,
            "Resume text is too short to extract a meaningful profile.",
        )

    profile = extract_resume_profile(payload.raw_text)

    # Generate embedding for vector search
    try:
        embedding = embed_text(payload.raw_text)
    except Exception as e:
        logger.warning("resume.embedding_failed", error=str(e))
        embedding = []

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

    await repo.upsert_resume(resume)

    logger.info(
        "resume.ingested",
        candidate_id=payload.candidate_id,
        n_skills=len(resume.skills),
    )

    return resume


@router.post("/resume/upload", response_model=ParsedResume, status_code=201)
async def upload_resume(
    file: UploadFile = File(...),
    repo: CandidateRepository = Depends(get_candidate_repo),
) -> ParsedResume:
    """Upload a resume (PDF, DOCX, TXT, RTF, ODT, HTML), validate it with the 
    Document Intelligence pipeline, extract its text, and store the parsed profile.
    
    V2 enhancements:
    - Detects actual file type from magic bytes, not just extension
    - Validates document type (resume vs invoice vs certificate etc.)
    - Checks extraction quality and provides human-readable error messages
    - Supports PDF, DOCX, TXT, RTF, ODT, HTML formats
    - Identifies image-based PDFs requiring OCR
    - Extracts personal info, education level, experience years
    """

    if not file.filename:
        raise HTTPException(400, "No file was provided.")

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(400, "The uploaded file is empty.")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Resume must be smaller than 10 MB.")

    # Process through Document Intelligence pipeline
    try:
        validation_result = await process_uploaded_document(
            file_bytes=file_bytes,
            filename=file.filename,
        )
    except Exception as exc:
        logger.exception("resume.processing_error", filename=file.filename)
        raise HTTPException(
            500,
            "An unexpected error occurred while processing your resume.",
        ) from exc

    # Check validation state - never silently continue on failure
    if validation_result.state == DocumentValidationState.CORRUPTED_FILE:
        raise HTTPException(422, validation_result.user_message)
    
    if validation_result.state == DocumentValidationState.UNSUPPORTED_DOCUMENT:
        raise HTTPException(415, validation_result.user_message)
    
    if validation_result.state == DocumentValidationState.RESUME_REQUIRES_OCR:
        raise HTTPException(
            422,
            f"{validation_result.user_message} {validation_result.next_action}",
        )
    
    if validation_result.state == DocumentValidationState.LOW_EXTRACTION_QUALITY:
        raise HTTPException(
            422,
            f"{validation_result.user_message} {validation_result.next_action}",
        )
    
    if validation_result.state == DocumentValidationState.NOT_A_RESUME:
        raise HTTPException(
            422,
            f"{validation_result.user_message} {validation_result.next_action}",
        )
    
    if validation_result.state == DocumentValidationState.INSUFFICIENT_INFORMATION:
        raise HTTPException(
            422,
            f"{validation_result.user_message} {validation_result.next_action}",
        )
    
    if validation_result.state == DocumentValidationState.EXTRACTION_FAILED:
        raise HTTPException(
            500,
            "The resume file could not be processed. Please try another file.",
        )

    # At this point we have a VALID_RESUME
    raw_text = validation_result.raw_text
    
    candidate_id = f"cand_upload_{uuid4().hex[:12]}"

    # Extract structured profile
    profile = extract_resume_profile(raw_text)

    # Generate embedding for vector search
    try:
        embedding = embed_text(raw_text)
    except Exception as e:
        logger.warning("resume.embedding_failed", error=str(e))
        embedding = []

    resume = ParsedResume(
        candidate_id=candidate_id,
        raw_text=raw_text,
        preferred_locations=[],
        preferred_domains=[],
        # V2 fields from document intelligence
        validation_state=validation_result.state,
        document_type=validation_result.document_type,
        extraction_quality_score=validation_result.extraction_quality_score,
        detected_name=validation_result.detected_name,
        detected_email=validation_result.detected_email,
        detected_phone=validation_result.detected_phone,
        current_location=validation_result.detected_location,
        total_experience_years=validation_result.detected_total_experience_years or 0.0,
        processing_metadata={
            "file_type_detected": validation_result.file_type_detected,
            "file_type_expected": validation_result.file_type_expected,
            "extraction_warnings": validation_result.technical_details.get("warnings", []),
        },
        embedding=embedding,
        **profile,
    )

    await repo.upsert_resume(resume)

    logger.info(
        "resume.uploaded",
        candidate_id=candidate_id,
        filename=file.filename,
        n_skills=len(resume.skills),
        validation_state=validation_result.state.value,
    )

    return resume


@router.get("/{candidate_id}/resume", response_model=ParsedResume)
async def get_resume(
    candidate_id: str,
    repo: CandidateRepository = Depends(get_candidate_repo),
) -> ParsedResume:
    resume = await repo.get_resume(candidate_id)

    if not resume:
        raise HTTPException(404, "Candidate resume not found.")

    return resume