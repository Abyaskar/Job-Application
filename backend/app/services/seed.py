"""
Seeds the store with sample resumes + jobs and rebuilds the vector index.

Called automatically on startup in DEMO_MODE (see app.main lifespan) so the
API, evaluation endpoints, and frontend all have data to work with
immediately with zero manual setup. Also safe to call against a real
Mongo-backed deployment (idempotent upserts) to bootstrap a demo dataset.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.logging import get_logger
from app.models.schemas import ParsedResume, ResumeIn
from app.repositories.repository import CandidateRepository, JobRepository
from app.services.extraction import extract_resume_profile
from app.services.recommender import rebuild_job_index

logger = get_logger("seed")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


async def seed_demo_data(candidate_repo: CandidateRepository, job_repo: JobRepository) -> dict:
    jobs_path = DATA_DIR / "sample_jobs.json"
    resumes_path = DATA_DIR / "sample_resumes.json"

    n_jobs = 0
    if jobs_path.exists():
        from app.models.schemas import JobIn
        from app.services.extraction import extract_job_requirements
        from app.models.schemas import ParsedJob

        with open(jobs_path) as f:
            jobs = json.load(f)
        for item in jobs:
            job_in = JobIn(**item)
            requirements = extract_job_requirements(job_in.raw_description)
            job = ParsedJob(
                job_id=job_in.job_id,
                title=job_in.title,
                company=job_in.company,
                location=job_in.location,
                domain=job_in.domain,
                raw_description=job_in.raw_description,
                requirements=requirements,
            )
            await job_repo.upsert_job(job)
            n_jobs += 1

    n_candidates = 0
    if resumes_path.exists():
        with open(resumes_path) as f:
            resumes = json.load(f)
        for item in resumes:
            resume_in = ResumeIn(**item)
            profile = extract_resume_profile(resume_in.raw_text)
            resume = ParsedResume(
                candidate_id=resume_in.candidate_id,
                raw_text=resume_in.raw_text,
                preferred_locations=resume_in.preferred_locations,
                preferred_domains=resume_in.preferred_domains,
                **profile,
            )
            await candidate_repo.upsert_resume(resume)
            n_candidates += 1

    n_indexed = await rebuild_job_index(job_repo)
    logger.info("seed.complete", n_jobs=n_jobs, n_candidates=n_candidates, n_indexed=n_indexed)
    return {"jobs": n_jobs, "candidates": n_candidates, "indexed": n_indexed}
