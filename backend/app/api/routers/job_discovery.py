"""
Job Discovery Router — V2.

Provides endpoints for realistic job discovery, generating live job search
links based on candidate profile rather than presenting stale dataset jobs
as current opportunities.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_candidate_repo
from app.core.logging import get_logger
from app.models.schemas import ParsedResume
from app.repositories.repository import CandidateRepository
from app.services.job_discovery import (
    JobDiscoveryResult,
    discover_jobs_for_candidate,
    get_live_job_search_url,
)

router = APIRouter(prefix="/job-discovery", tags=["job-discovery"])
logger = get_logger("api.job_discovery")


@router.get("/candidate/{candidate_id}")
async def discover_jobs(
    candidate_id: str,
    repo: CandidateRepository = Depends(get_candidate_repo),
) -> JobDiscoveryResult:
    """Generate live job search destinations for a candidate.
    
    This endpoint:
    1. Retrieves the candidate's resume/profile
    2. Analyzes their role, skills, location preferences
    3. Generates search URLs for LinkedIn, Naukri, Indeed, Glassdoor
    4. Provides contextual search tips
    
    Returns job search destinations, NOT specific job listings.
    The user performs actual applications on external platforms.
    """
    resume = await repo.get_resume(candidate_id)
    
    if not resume:
        raise HTTPException(404, "Candidate resume not found.")
    
    try:
        result = discover_jobs_for_candidate(resume)
        
        logger.info(
            "job_discovery.generated",
            candidate_id=candidate_id,
            n_destinations=len(result.destinations),
            primary_role=result.primary_role,
        )
        
        return result
        
    except Exception as e:
        logger.exception("job_discovery.failed", candidate_id=candidate_id)
        raise HTTPException(
            500,
            "Failed to generate job search destinations.",
        ) from e


@router.get("/search-url")
async def get_search_url(
    role: str,
    location: str | None = None,
    platform: str = "linkedin",
    remote: bool = False,
) -> dict[str, str]:
    """Get a direct job search URL for immediate redirection.
    
    Query params:
    - role: Job role/title to search for (required)
    - location: Preferred location (optional)
    - platform: One of 'linkedin', 'naukri', 'indeed', 'glassdoor' (default: linkedin)
    - remote: Whether to filter for remote jobs (default: false)
    
    Returns a redirect URL to the platform's job search results.
    """
    valid_platforms = ["linkedin", "naukri", "indeed", "glassdoor"]
    if platform.lower() not in valid_platforms:
        raise HTTPException(
            400,
            f"Invalid platform. Choose from: {', '.join(valid_platforms)}",
        )
    
    try:
        search_url = get_live_job_search_url(
            role=role,
            location=location,
            platform=platform,
            remote=remote,
        )
        
        return {
            "search_url": search_url,
            "platform": platform,
            "role": role,
            "location": location,
            "remote": remote,
        }
        
    except Exception as e:
        logger.exception("search_url.failed", role=role, platform=platform)
        raise HTTPException(
            500,
            "Failed to generate search URL.",
        ) from e
