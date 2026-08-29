"""
Job Discovery Layer — V2.

This module provides realistic job discovery capabilities that go beyond
the static test dataset. It uses the candidate's recommended role, location,
experience level and preferences to generate relevant live job-search
destinations.

Key principles:
1. Keep existing job dataset for testing/evaluation/RAG knowledge
2. Do NOT present old sample jobs as current live opportunities
3. Generate job-search links based on candidate profile
4. Where authorized LinkedIn API access is available, integrate properly
5. Never scrape personal LinkedIn profiles or bypass restrictions
6. Show "View on LinkedIn" or equivalent button redirecting to external platform
7. User performs the actual application on the external platform

Design notes:
- Does not handle job applications directly
- Extracts location from resume but allows user override
- Supports Mumbai, other Indian cities, remote work, country-level preferences
- Tracks relocation preferences
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

from app.core.logging import get_logger
from app.models.schemas import CareerDomain, ParsedResume

logger = get_logger("services.job_discovery")


@dataclass
class JobSearchDestination:
    """A destination where the candidate can search for live jobs."""
    
    platform_name: str
    platform_url: str
    search_query: str
    search_url: str
    job_count_estimate: str | None = None
    is_primary: bool = False
    requires_auth: bool = False
    
    # Metadata for UI display
    icon: str | None = None  # Lucide icon name
    description: str | None = None
    
    # V2: Search strategy details
    primary_title: str | None = None
    alternative_titles: list[str] = None
    search_terms: list[str] = None
    location_filter: str | None = None
    posted_within_days: int | None = None
    boolean_query: str | None = None


def __post_init__(self):
    if self.alternative_titles is None:
        self.alternative_titles = []
    if self.search_terms is None:
        self.search_terms = []


@dataclass
class JobDiscoveryResult:
    """Complete job discovery result for a candidate."""
    
    candidate_id: str
    primary_role: str | None
    career_domains: list[CareerDomain]
    location_preference: str | None
    open_to_remote: bool
    open_to_relocation: bool
    
    # Search destinations
    destinations: list[JobSearchDestination]
    
    # Contextual advice
    search_tips: list[str]
    in_demand_skills: list[str]
    

# Platform-specific URL builders
def build_linkedin_job_search(
    keywords: str,
    location: str | None = None,
    remote: bool = False,
    posted_within_days: int | None = None,
    alternative_titles: list[str] | None = None,
) -> str:
    """Build LinkedIn job search URL with proper parameters.
    
    Uses LinkedIn's public job search URL format.
    Note: For production use with user auth, use LinkedIn API v2.
    
    Supports boolean queries with OR for alternative titles.
    """
    base_url = "https://www.linkedin.com/jobs/search"
    
    # Build boolean query if alternative titles provided
    if alternative_titles and len(alternative_titles) > 0:
        # Create OR query: ("title1" OR "title2" OR "title3")
        quoted_titles = [f'"{t}"' for t in alternative_titles]
        keywords_query = f'({" OR ".join(quoted_titles)})'
    else:
        keywords_query = keywords
    
    params = [f"keywords={quote_plus(keywords_query)}"]
    
    if location:
        # LinkedIn uses geographic URNs for precise locations
        # For simplicity, we use the location string directly
        # In production, you'd map to LinkedIn's geo URNs via API
        params.append(f"geoUrn={quote_plus(location)}")
    
    if remote:
        params.append("f_WT=2")  # Remote filter
    
    if posted_within_days:
        # LinkedIn time filters: f_TPR=r86400 (24h), r604800 (7d), r2592000 (30d)
        seconds = posted_within_days * 86400
        params.append(f"f_TPR=r{seconds}")
    
    return f"{base_url}?{'&'.join(params)}"


def build_naukri_search(
    keywords: str,
    location: str | None = None,
    experience_min: float | None = None,
) -> str:
    """Build Naukri.com job search URL (India-focused)."""
    base_url = "https://www.naukri.com/"
    query_parts = [keywords]
    
    if location:
        query_parts.append(location)
    
    query = "-".join(query_parts).replace(" ", "-")
    return f"{base_url}{query}-jobs"


def build_indeed_search(
    keywords: str,
    location: str | None = None,
    remote: bool = False,
) -> str:
    """Build Indeed job search URL."""
    base_url = "https://www.indeed.com/jobs"
    params = [f"q={quote_plus(keywords)}"]
    
    if location:
        params.append(f"l={quote_plus(location)}")
    
    if remote:
        params.append("remotejob=1")
    
    return f"{base_url}?{'&'.join(params)}"


def build_glassdoor_search(
    keywords: str,
    location: str | None = None,
) -> str:
    """Build Glassdoor job search URL."""
    base_url = "https://www.glassdoor.co.in/Job/jobs.htm"
    params = [f"sc.keyword={quote_plus(keywords)}"]
    
    if location:
        params.append(f"locT={quote_plus(location)}")
    
    return f"{base_url}?{'&'.join(params)}"


def discover_jobs_for_candidate(
    resume: ParsedResume,
) -> JobDiscoveryResult:
    """Generate job search destinations based on candidate profile.
    
    This function:
    1. Extracts primary role/title from resume
    2. Identifies career domains
    3. Determines location preferences
    4. Builds search URLs for multiple platforms
    5. Provides contextual search tips
    """
    # Determine primary role/title
    primary_role = None
    if resume.experience and len(resume.experience) > 0:
        # Use most recent/senior title
        primary_role = resume.experience[-1].title if resume.experience else None
    
    # If no experience, try to infer from skills or education
    if not primary_role and resume.skills:
        skill_based_roles = {
            "python": "Python Developer",
            "java": "Java Developer",
            "react": "Frontend Developer",
            "data_analysis": "Data Analyst",
            "financial_reporting": "Financial Analyst",
            "accounting": "Accountant",
            "marketing": "Marketing Specialist",
            "hr_management": "HR Coordinator",
        }
        for skill, role in skill_based_roles.items():
            if any(skill in s.lower() for s in resume.skills):
                primary_role = role
                break
    
    # Location preference
    location_pref = resume.current_location
    if resume.preferred_countries:
        location_pref = resume.preferred_countries[0]
    
    # Build search queries based on role and skills
    search_keywords = primary_role or "entry level"
    if resume.skills and len(resume.skills) > 0:
        # Add top skills to search
        skill_keywords = resume.skills[:3]
        search_keywords = f"{primary_role or ''} {' '.join(skill_keywords)}".strip()
    
    # Build destinations
    destinations: list[JobSearchDestination] = []
    
    # LinkedIn (primary)
    linkedin_url = build_linkedin_job_search(
        keywords=search_keywords,
        location=location_pref if not resume.open_to_remote else None,
        remote=resume.open_to_remote,
    )
    destinations.append(JobSearchDestination(
        platform_name="LinkedIn",
        platform_url="https://www.linkedin.com/jobs",
        search_query=search_keywords,
        search_url=linkedin_url,
        is_primary=True,
        requires_auth=True,
        icon="Linkedin",
        description=f"Search {search_keywords} roles" + (f" in {location_pref}" if location_pref else ""),
    ))
    
    # Naukri (for India-based candidates)
    if location_pref and any(city in location_pref.lower() for city in 
                            ["mumbai", "delhi", "bangalore", "bengaluru", "chennai", 
                             "hyderabad", "pune", "ahmedabad", "kolkata", "india"]):
        naukri_url = build_naukri_search(
            keywords=search_keywords,
            location=location_pref,
            experience_min=resume.total_experience_years,
        )
        destinations.append(JobSearchDestination(
            platform_name="Naukri.com",
            platform_url="https://www.naukri.com",
            search_query=search_keywords,
            search_url=naukri_url,
            is_primary=False,
            icon="Briefcase",
            description="India's largest job portal",
        ))
    
    # Indeed (global)
    indeed_url = build_indeed_search(
        keywords=search_keywords,
        location=location_pref if not resume.open_to_remote else "Remote",
        remote=resume.open_to_remote,
    )
    destinations.append(JobSearchDestination(
        platform_name="Indeed",
        platform_url="https://www.indeed.com",
        search_query=search_keywords,
        search_url=indeed_url,
        is_primary=False,
        icon="Search",
        description="Global job aggregator",
    ))
    
    # Glassdoor (for company research)
    glassdoor_url = build_glassdoor_search(
        keywords=search_keywords,
        location=location_pref,
    )
    destinations.append(JobSearchDestination(
        platform_name="Glassdoor",
        platform_url="https://www.glassdoor.co.in",
        search_query=search_keywords,
        search_url=glassdoor_url,
        is_primary=False,
        icon="Building",
        description="Jobs + company reviews + salary insights",
    ))
    
    # Generate search tips
    search_tips = []
    
    if resume.total_experience_years < 2:
        search_tips.append(
            "With less than 2 years of experience, also search for 'junior', "
            "'associate', or 'trainee' positions."
        )
    
    if resume.open_to_remote:
        search_tips.append(
            "Remote roles are competitive. Filter by 'Posted in last 7 days' "
            "to find fresh openings."
        )
    
    if resume.skills and len(resume.skills) > 0:
        top_skill = resume.skills[0]
        search_tips.append(
            f"Add '{top_skill}' plus related tools/frameworks to your search "
            "for more targeted results."
        )
    
    if resume.open_to_relocation:
        search_tips.append(
            "Since you're open to relocation, try searching major hubs like "
            "Bangalore, Mumbai, Hyderabad, or Pune for more opportunities."
        )
    
    # Identify in-demand skills (from resume skills)
    in_demand_skills = resume.skills[:5] if resume.skills else []
    
    return JobDiscoveryResult(
        candidate_id=resume.candidate_id,
        primary_role=primary_role,
        career_domains=resume.career_domains,
        location_preference=location_pref,
        open_to_remote=resume.open_to_remote,
        open_to_relocation=resume.open_to_relocation,
        destinations=destinations,
        search_tips=search_tips,
        in_demand_skills=in_demand_skills,
    )


def get_live_job_search_url(
    role: str,
    location: str | None = None,
    platform: str = "linkedin",
    remote: bool = False,
    alternative_titles: list[str] | None = None,
    posted_within_days: int | None = 7,
) -> str:
    """Get a live job search URL for immediate redirection.
    
    Args:
        role: Job role/title to search for
        location: Preferred location (city, country)
        platform: One of 'linkedin', 'naukri', 'indeed', 'glassdoor'
        remote: Whether to include remote-only filter
        alternative_titles: List of alternative job titles for boolean OR query
        posted_within_days: Filter jobs posted within this many days (None = no filter)
    
    Returns:
        Direct URL to job search results page
    """
    platform_builders = {
        "linkedin": lambda: build_linkedin_job_search(
            keywords=role, 
            location=location, 
            remote=remote,
            posted_within_days=posted_within_days,
            alternative_titles=alternative_titles,
        ),
        "naukri": lambda: build_naukri_search(role, location),
        "indeed": lambda: build_indeed_search(role, location, remote),
        "glassdoor": lambda: build_glassdoor_search(role, location),
    }
    
    builder = platform_builders.get(platform.lower())
    if builder:
        return builder()
    
    # Default to LinkedIn
    return build_linkedin_job_search(role, location, remote)


def generate_search_strategy(
    primary_role: str,
    location: str | None = None,
    skills: list[str] | None = None,
    experience_years: float = 0.0,
    open_to_remote: bool = True,
    career_intent: str | None = None,
) -> JobSearchDestination:
    """Generate an intelligent job search strategy with alternative titles and boolean queries.
    
    This creates optimized search queries that go beyond exact title matching.
    
    Args:
        primary_role: The main role/title (e.g., "Business Analyst")
        location: Preferred location
        skills: List of candidate skills
        experience_years: Years of experience
        open_to_remote: Whether open to remote work
        career_intent: Stated career intent (may differ from current role)
    
    Returns:
        JobSearchDestination with full search strategy
    """
    # Define alternative titles for common roles
    role_alternatives = {
        "business analyst": ["Business Analyst", "MIS Analyst", "Reporting Analyst", "Junior Business Analyst", "Data Analyst"],
        "data analyst": ["Data Analyst", "Business Analyst", "Reporting Analyst", "BI Analyst", "Analytics Associate"],
        "accountant": ["Accountant", "Junior Accountant", "Accounts Executive", "Tax Assistant", "Bookkeeper"],
        "software engineer": ["Software Engineer", "Software Developer", "Programmer", "Application Developer", "SDE"],
        "ml engineer": ["ML Engineer", "Machine Learning Engineer", "AI Engineer", "Data Scientist - ML", "Deep Learning Engineer"],
        "frontend developer": ["Frontend Developer", "Front-end Developer", "UI Developer", "React Developer", "Web Developer"],
        "backend developer": ["Backend Developer", "Back-end Developer", "Server-side Developer", "API Developer", "Python Developer"],
        "full stack developer": ["Full Stack Developer", "Fullstack Developer", "Web Developer", "Software Engineer - Full Stack"],
        "marketing specialist": ["Marketing Specialist", "Digital Marketing Executive", "Marketing Coordinator", "Brand Executive"],
        "hr coordinator": ["HR Coordinator", "HR Executive", "Recruitment Executive", "Talent Acquisition Associate"],
        "financial analyst": ["Financial Analyst", "Finance Executive", "Investment Analyst", "Credit Analyst"],
    }
    
    # Determine primary role to use
    search_role = career_intent if career_intent else primary_role
    search_role_lower = search_role.lower() if search_role else ""
    
    # Get alternative titles
    alternatives = []
    for key, titles in role_alternatives.items():
        if key in search_role_lower or search_role_lower in key:
            alternatives = titles
            break
    
    if not alternatives and search_role:
        # Generate basic variations
        alternatives = [
            search_role,
            f"Junior {search_role}" if experience_years < 2 else search_role,
            f"Senior {search_role}" if experience_years >= 5 else None,
        ]
        alternatives = [a for a in alternatives if a]
    
    # Build boolean query
    if alternatives:
        quoted_titles = [f'"{t}"' for t in alternatives[:5]]  # Limit to 5 for URL length
        boolean_query = f'({" OR ".join(quoted_titles)})'
    else:
        boolean_query = f'"{search_role}"' if search_role else '"entry level"'
    
    # Add skill keywords if available
    skill_keywords = ""
    if skills and len(skills) > 0:
        top_skills = skills[:3]
        skill_keywords = f" {' '.join(top_skills)}"
    
    # Determine time filter based on experience level
    posted_within = 7 if experience_years < 3 else 14  # Junior roles move faster
    
    # Build LinkedIn URL
    linkedin_url = build_linkedin_job_search(
        keywords=search_role or "entry level",
        location=location if not open_to_remote else None,
        remote=open_to_remote,
        posted_within_days=posted_within,
        alternative_titles=alternatives if alternatives else None,
    )
    
    description_parts = []
    if alternatives and len(alternatives) > 1:
        description_parts.append(f"Searching: {alternatives[0]}")
        if len(alternatives) > 1:
            description_parts.append(f"(also: {', '.join(alternatives[1:3])})")
    if location:
        description_parts.append(f"in {location}")
    if open_to_remote:
        description_parts.append("+ Remote")
    
    return JobSearchDestination(
        platform_name="LinkedIn",
        platform_url="https://www.linkedin.com/jobs",
        search_query=search_role or "entry level",
        search_url=linkedin_url,
        is_primary=True,
        requires_auth=True,
        icon="Linkedin",
        description=" ".join(description_parts),
        primary_title=search_role,
        alternative_titles=alternatives,
        search_terms=(skills or [])[:3],
        location_filter=location,
        posted_within_days=posted_within,
        boolean_query=boolean_query,
    )
