"""
Structured requirement / profile extraction.

This is intentionally rule-based (regex + taxonomy lookups) rather than an
LLM call: extraction runs on every ingested resume/JD (high volume, latency
sensitive, must be deterministic for evaluation), whereas the LLM budget in
this system is reserved for the RAG explanation layer where natural
language generation is actually needed. This mirrors a real production
trade-off -- cheap deterministic NLP for structured fields, GenAI only
where free-text reasoning adds value the rules can't.
"""
from __future__ import annotations

import re

from app.models.schemas import (
    EducationEntry,
    ExperienceEntry,
    ExtractedRequirements,
)
from app.services.taxonomy import extract_skills

EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience", re.I),
    re.compile(r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*-\s*\d+\s*(?:years?|yrs?)", re.I),
]

EDUCATION_LEVELS = {
    3: [r"\bphd\b", r"\bdoctorate\b"],
    2: [r"\bmaster'?s?\b", r"\bmsc\b", r"\bm\.sc\b", r"\bmba\b", r"\bm\.eng\b"],
    1: [r"\bbachelor'?s?\b", r"\bbsc\b", r"\bb\.sc\b", r"\bb\.eng\b", r"\bundergraduate\b"],
}

LOCATION_HINTS = re.compile(
    r"\b(remote|hybrid|on-?site|london|new york|mumbai|bangalore|bengaluru|"
    r"berlin|san francisco|singapore|dublin|toronto)\b",
    re.I,
)


def _max_education_level(text: str) -> int:
    text_lower = text.lower()
    for level in (3, 2, 1):
        for pattern in EDUCATION_LEVELS[level]:
            if re.search(pattern, text_lower):
                return level
    return 0


def _extract_years_experience(text: str) -> float:
    matches = []
    for pattern in EXPERIENCE_PATTERNS:
        matches += [float(m) for m in pattern.findall(text)]
    return max(matches) if matches else 0.0


def _extract_location(text: str) -> str:
    match = LOCATION_HINTS.search(text)
    return match.group(1).title() if match else "Remote"


def extract_job_requirements(raw_description: str) -> ExtractedRequirements:
    skills = extract_skills(raw_description)

    # Heuristic split: skills mentioned near "required"/"must have" vs
    # "nice to have"/"preferred" sections. Falls back to treating all
    # extracted skills as required if no preference section is detected.
    preferred_section = re.search(
        r"(?:nice to have|preferred|bonus)[\s\S]{0,600}", raw_description, re.I
    )
    preferred_skills = extract_skills(preferred_section.group(0)) if preferred_section else []
    required_skills = [s for s in skills if s not in preferred_skills]

    return ExtractedRequirements(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience_years=_extract_years_experience(raw_description),
        education_level_required=_max_education_level(raw_description) or 1,
        location=_extract_location(raw_description),
        seniority=_infer_seniority(raw_description),
    )


def _infer_seniority(text: str) -> str | None:
    text_lower = text.lower()
    if re.search(r"\bintern(ship)?\b", text_lower):
        return "internship"
    if re.search(r"\bsenior\b|\blead\b|\bstaff\b", text_lower):
        return "senior"
    if re.search(r"\bjunior\b|\bentry.level\b|\bgraduate\b", text_lower):
        return "junior"
    return "mid"


def extract_resume_profile(raw_text: str) -> dict:
    """Returns a dict compatible with ParsedResume's extra fields."""
    skills = extract_skills(raw_text)
    years = _extract_years_experience(raw_text)

    education = []
    edu_level = _max_education_level(raw_text)
    if edu_level:
        label = {1: "Bachelor's", 2: "Master's", 3: "PhD"}[edu_level]
        education.append(EducationEntry(degree=label, level=edu_level))

    # Very light-weight "experience entries" — split on common resume
    # section breaks and keep short lines that look like role/employer
    # headers. This is a heuristic, not a full resume parser.
    experience = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if 5 < len(line) < 80 and re.search(r"\b(engineer|scientist|analyst|developer|intern|manager)\b", line, re.I):
            experience.append(ExperienceEntry(title=line, years=0.0))

    return {
        "skills": skills,
        "education": education,
        "experience": experience[:10],
        "total_experience_years": years,
    }
