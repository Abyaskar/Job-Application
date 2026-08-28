from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Candidate / Resume
# ---------------------------------------------------------------------------

class EducationEntry(BaseModel):
    degree: str
    field: str | None = None
    institution: str | None = None
    level: int = Field(0, description="0=none,1=bachelor,2=master,3=phd")


class ExperienceEntry(BaseModel):
    title: str
    company: str | None = None
    years: float = 0.0
    description: str = ""


class ResumeIn(BaseModel):
    candidate_id: str
    raw_text: str
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    min_salary: int | None = None


class ParsedResume(BaseModel):
    candidate_id: str
    raw_text: str
    skills: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    total_experience_years: float = 0.0
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Career Intent Layer
# ---------------------------------------------------------------------------

class IntentIn(BaseModel):
    candidate_id: str
    free_text: str = Field(..., description="e.g. 'GenAI Engineer', 'I want to move into data analytics'")


class IntentMatchMethod(str, Enum):
    EXACT_ALIAS = "exact_alias"
    FUZZY_ALIAS = "fuzzy_alias"
    KEYWORD_OVERLAP = "keyword_overlap"
    EMBEDDING = "embedding"
    UNRESOLVED = "unresolved"


class IntentProfile(BaseModel):
    """Normalized career intent — the output of the Career Intent Layer.

    This is what lets ranking distinguish "resume happens to be textually
    similar to this JD" from "candidate actually wants this kind of role."
    """
    candidate_id: str
    raw_text: str
    role_family: str | None = None
    canonical_title: str | None = None
    related_titles: list[str] = Field(default_factory=list)
    intent_skills: list[str] = Field(default_factory=list)
    intent_keywords: list[str] = Field(default_factory=list)
    match_method: IntentMatchMethod = IntentMatchMethod.UNRESOLVED
    confidence: float = 0.0
    updated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Job Description
# ---------------------------------------------------------------------------

class JobIn(BaseModel):
    job_id: str
    title: str
    company: str
    location: str = "Remote"
    domain: str | None = None
    raw_description: str
    seniority: str | None = None


class ExtractedRequirements(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: float = 0.0
    education_level_required: int = 1  # 0 none,1 bachelor,2 master,3 phd
    location: str = "Remote"
    domain: str | None = None
    seniority: str | None = None


class ParsedJob(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    domain: str | None
    raw_description: str
    requirements: ExtractedRequirements
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class RecommendedAction(str, Enum):
    APPLY_NOW = "apply_now"
    TAILOR_RESUME_FIRST = "tailor_resume_first"
    BUILD_MISSING_EVIDENCE = "build_missing_evidence"
    LOW_PRIORITY = "low_priority"


class ScoreBreakdown(BaseModel):
    intent_alignment: float = 1.0
    semantic_similarity: float
    hard_skill_match: float
    experience_match: float
    education_match: float
    location_match: float
    final_score: float
    weights: dict[str, float]
    intent_gated: bool = Field(
        False, description="True if this job was hard-penalized for conflicting with stated career intent"
    )


class SkillGap(BaseModel):
    missing_required: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    matched_required: list[str] = Field(default_factory=list)
    coverage_ratio: float = 0.0


class EvidenceSnippet(BaseModel):
    source: str  # "resume" | "job_description" | "skill_taxonomy"
    text: str
    relevance: float


class Explanation(BaseModel):
    summary: str
    reasons: list[str]
    why_apply: list[str] = Field(default_factory=list)
    why_not_apply: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSnippet]
    grounded: bool = True
    confidence: float = 0.0


class Recommendation(BaseModel):
    candidate_id: str
    job_id: str
    job_title: str
    company: str
    job_location: str = "Remote"
    job_domain: str | None = None
    score: ScoreBreakdown
    skill_gap: SkillGap
    action: RecommendedAction
    explanation: Explanation
    uncertainty: float = Field(
        0.0, description="0=confident,1=high uncertainty (e.g. thin resume evidence)"
    )
    generated_at: datetime = Field(default_factory=utcnow)


class FeedbackIn(BaseModel):
    candidate_id: str
    job_id: str
    accepted: bool
    applied: bool = False
    comment: str | None = None


class ApplicationRecord(BaseModel):
    candidate_id: str
    job_id: str
    status: str = "recommended"  # recommended -> applied -> interview -> offer/rejected
    updated_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Search / Evaluation
# ---------------------------------------------------------------------------

class SearchMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


class RankRequest(BaseModel):
    candidate_id: str
    top_k: int = 10
    mode: SearchMode = SearchMode.HYBRID
    location_filter: str | None = None
    domain_filter: str | None = None
    use_intent: bool = Field(
        True, description="Apply the candidate's stored career intent (see POST /intent) to gate/boost ranking."
    )


class EvalRunResult(BaseModel):
    mode: SearchMode
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    avg_latency_ms: float
    n_queries: int
