"""
Structured requirement / profile extraction.

Rule-based extraction shared by resume and job ingestion.
V2 additions: career-domain detection, seniority inference, and
experience-year extraction.
"""
from __future__ import annotations

import re

from app.models.schemas import (
    CareerDomain,
    EducationEntry,
    ExperienceEntry,
    ExtractedRequirements,
)
from app.services.taxonomy import extract_skills, load_role_taxonomy


EXPERIENCE_PATTERNS = [
    re.compile(
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience",
        re.I,
    ),
    re.compile(
        r"minimum\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        re.I,
    ),
    re.compile(
        r"(\d+(?:\.\d+)?)\s*-\s*\d+\s*(?:years?|yrs?)",
        re.I,
    ),
]

EDUCATION_LEVELS = {
    3: [r"\bphd\b", r"\bdoctorate\b"],
    2: [
        r"\bmaster'?s?\b",
        r"\bmsc\b",
        r"\bm\.sc\b",
        r"\bmba\b",
        r"\bm\.eng\b",
    ],
    1: [
        r"\bbachelor'?s?\b",
        r"\bbsc\b",
        r"\bb\.sc\b",
        r"\bb\.eng\b",
        r"\bundergraduate\b",
    ],
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
    matches: list[float] = []

    for pattern in EXPERIENCE_PATTERNS:
        matches += [float(m) for m in pattern.findall(text)]

    return max(matches) if matches else 0.0


def _extract_location(text: str) -> str:
    match = LOCATION_HINTS.search(text)
    return match.group(1).title() if match else "Remote"


def extract_job_requirements(raw_description: str) -> ExtractedRequirements:
    """Extract structured requirements from a job description."""
    skills = extract_skills(raw_description)

    # Skills near a "nice to have"/"preferred"/"bonus" section are
    # treated as preferred. Everything else is required.
    preferred_section = re.search(
        r"(?:nice to have|preferred|bonus)[\s\S]{0,600}",
        raw_description,
        re.I,
    )

    preferred_skills = (
        extract_skills(preferred_section.group(0))
        if preferred_section
        else []
    )

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
    """Return fields compatible with ParsedResume's extra fields.

    Full document validation/parsing remains in the document-intelligence
    layer. This function supplies the structured profile used downstream.
    """
    skills = extract_skills(raw_text)
    years = _extract_years_experience(raw_text)

    # Reuse the already extracted skills so career-domain detection
    # does not rescan the same resume for every taxonomy role.
    career_domains = detect_career_domain(
        raw_text,
        extracted_skills=skills,
    )
    primary_domain = career_domains[0].domain_id if career_domains else None

    education: list[EducationEntry] = []
    edu_level = _max_education_level(raw_text)

    if edu_level:
        label = {
            1: "Bachelor's",
            2: "Master's",
            3: "PhD",
        }[edu_level]

        education.append(
            EducationEntry(
                degree=label,
                level=edu_level,
            )
        )

    # Lightweight experience-entry heuristic. The document-intelligence
    # layer remains responsible for richer experience extraction.
    experience: list[ExperienceEntry] = []

    for line in raw_text.split("\n"):
        line = line.strip()

        if (
            5 < len(line) < 80
            and re.search(
                r"\b(engineer|scientist|analyst|developer|intern|manager|"
                r"accountant|executive|consultant)\b",
                line,
                re.I,
            )
        ):
            experience.append(
                ExperienceEntry(
                    title=line,
                    years=0.0,
                )
            )

    return {
        "skills": skills,
        "education": education,
        "experience": experience[:10],
        "total_experience_years": years,
        "career_domains": [d.model_dump() for d in career_domains],
        "primary_domain": primary_domain,
        "seniority_level": _infer_seniority(raw_text),
    }


def detect_career_domain(
    text: str,
    extracted_skills: list[str] | None = None,
) -> list[CareerDomain]:
    """Detect professional domains using the shared role taxonomy."""
    role_taxonomy = load_role_taxonomy()
    text_lower = text.lower()

    # If skills were already extracted by extract_resume_profile(),
    # reuse them. Existing callers can still call this function with
    # only text, in which case skills are extracted here as before.
    if extracted_skills is None:
        extracted_skills = extract_skills(text)

    extracted_set = set(extracted_skills)
    detected_domains: list[CareerDomain] = []

    for role_id, role_data in role_taxonomy.items():
        keyword_matches = sum(
            1
            for kw in role_data.get("keywords", [])
            if kw.lower() in text_lower
        )

        core_skills = set(role_data.get("core_skills", []))
        skill_matches = len(extracted_set & core_skills)

        title_matches = sum(
            1
            for alias in role_data.get("aliases", [])
            if alias.lower() in text_lower
        )

        total_signals = keyword_matches + skill_matches + title_matches * 2

        if total_signals > 0:
            confidence = min(1.0, total_signals / 10.0)

            if confidence >= 0.2:
                related = list(
                    role_data.get("related_titles", [])[:3]
                )

                detected_domains.append(
                    CareerDomain(
                        domain_id=role_id,
                        domain_name=role_data.get(
                            "canonical_title",
                            role_id.replace("_", " ").title(),
                        ),
                        confidence=round(confidence, 2),
                        related_titles=related,
                    )
                )

    detected_domains.sort(
        key=lambda d: d.confidence,
        reverse=True,
    )

    return detected_domains[:5]
