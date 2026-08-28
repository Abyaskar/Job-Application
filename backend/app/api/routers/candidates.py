from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_candidate_repo
from app.core.logging import get_logger
from app.models.schemas import ParsedResume, ResumeIn
from app.repositories.repositories import CandidateRepository
from app.services.extraction import extract_resume_profile

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


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

    resume = ParsedResume(
        candidate_id=payload.candidate_id,
        raw_text=payload.raw_text,
        preferred_locations=payload.preferred_locations,
        preferred_domains=payload.preferred_domains,
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
    """Upload a PDF or TXT resume, extract its text, and store the parsed profile."""

    if not file.filename:
        raise HTTPException(400, "No file was provided.")

    filename = file.filename.lower()

    is_pdf = filename.endswith(".pdf")
    is_txt = filename.endswith(".txt")

    if not (is_pdf or is_txt):
        raise HTTPException(
            415,
            "Unsupported file type. Please upload a PDF or TXT resume.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(400, "The uploaded file is empty.")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(413, "Resume must be smaller than 10 MB.")

    try:
        if is_pdf:
            if PdfReader is None:
                raise HTTPException(
                    500,
                    "PDF support is not installed on the backend.",
                )

            import io

            reader = PdfReader(io.BytesIO(file_bytes))

            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)

            raw_text = "\n".join(pages).strip()

        else:
            raw_text = file_bytes.decode("utf-8", errors="ignore").strip()

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("resume.extraction_failed", filename=file.filename)
        raise HTTPException(
            422,
            "The resume file could not be read. Please try another PDF or TXT file.",
        ) from exc

    if len(raw_text) < 20:
        raise HTTPException(
            422,
            "Not enough readable text was found in the resume.",
        )

    candidate_id = f"cand_upload_{uuid4().hex[:12]}"

    profile = extract_resume_profile(raw_text)

    resume = ParsedResume(
        candidate_id=candidate_id,
        raw_text=raw_text,
        preferred_locations=[],
        preferred_domains=[],
        **profile,
    )

    await repo.upsert_resume(resume)

    logger.info(
        "resume.uploaded",
        candidate_id=candidate_id,
        filename=file.filename,
        n_skills=len(resume.skills),
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