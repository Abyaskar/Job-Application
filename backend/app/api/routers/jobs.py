from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_job_repo
from app.core.logging import get_logger
from app.models.schemas import JobIn, ParsedJob
from app.repositories.repository import JobRepository
from app.services.extraction import extract_job_requirements
from app.services.recommender import rebuild_job_index

router = APIRouter(prefix="/jobs", tags=["jobs"])
logger = get_logger("api.jobs")


@router.post("", response_model=ParsedJob, status_code=201)
async def ingest_job(payload: JobIn, repo: JobRepository = Depends(get_job_repo)) -> ParsedJob:
    requirements = extract_job_requirements(payload.raw_description)
    job = ParsedJob(
        job_id=payload.job_id,
        title=payload.title,
        company=payload.company,
        location=payload.location,
        domain=payload.domain,
        raw_description=payload.raw_description,
        requirements=requirements,
        external_url=payload.external_url,
    )
    await repo.upsert_job(job)
    logger.info("job.ingested", job_id=job.job_id, n_required_skills=len(requirements.required_skills))
    return job


@router.post("/bulk", status_code=201)
async def ingest_jobs_bulk(payload: list[JobIn], repo: JobRepository = Depends(get_job_repo)) -> dict:
    created = 0
    for item in payload:
        requirements = extract_job_requirements(item.raw_description)
        job = ParsedJob(
            job_id=item.job_id,
            title=item.title,
            company=item.company,
            location=item.location,
            domain=item.domain,
            raw_description=item.raw_description,
            requirements=requirements,
            external_url=item.external_url,
        )
        await repo.upsert_job(job)
        created += 1
    n_indexed = await rebuild_job_index(repo)
    logger.info("jobs.bulk_ingested", created=created, indexed=n_indexed)
    return {"created": created, "indexed": n_indexed}


@router.post("/reindex", status_code=200)
async def reindex(repo: JobRepository = Depends(get_job_repo)) -> dict:
    """Rebuild the embedding provider fit + vector index over all stored
    jobs. Call after bulk ingestion, or on a schedule in production as new
    jobs accumulate."""
    n = await rebuild_job_index(repo)
    return {"indexed": n}


@router.get("/{job_id}", response_model=ParsedJob)
async def get_job(job_id: str, repo: JobRepository = Depends(get_job_repo)) -> ParsedJob:
    job = await repo.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("", response_model=list[ParsedJob])
async def list_jobs(limit: int = 100, repo: JobRepository = Depends(get_job_repo)) -> list[ParsedJob]:
    return await repo.list_jobs(limit=limit)
