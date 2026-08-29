from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Document Intelligence / Validation
# ---------------------------------------------------------------------------

class DocumentValidationState(str, Enum):
    """Explicit validation states for document processing."""
    VALID_RESUME = "valid_resume"
    RESUME_REQUIRES_OCR = "resume_requires_ocr"
    LOW_EXTRACTION_QUALITY = "low_extraction_quality"
    NOT_A_RESUME = "not_a_resume"
    CORRUPTED_FILE = "corrupted_file"
    UNSUPPORTED_DOCUMENT = "unsupported_document"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    EXTRACTION_FAILED = "extraction_failed"


class DocumentType(str, Enum):
    """Detected document types."""
    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    JOB_DESCRIPTION = "job_description"
    CERTIFICATE = "certificate"
    MARKSHEET = "marksheet"
    INVOICE = "invoice"
    ID_DOCUMENT = "id_document"
    PORTFOLIO = "portfolio"
    UNKNOWN = "unknown"


class DocumentValidationResult(BaseModel):
    """Complete validation result with human-readable messages."""
    state: DocumentValidationState
    document_type: DocumentType
    raw_text: str
    file_type_detected: str
    file_type_expected: str | None = None
    extraction_quality_score: float
    
    # Human-readable messages
    user_message: str
    next_action: str
    
    # Technical details for debugging
    technical_details: dict[str, Any] = Field(default_factory=dict)
    
    # Extracted metadata (only populated for valid resumes)
    detected_name: str | None = None
    detected_email: str | None = None
    detected_phone: str | None = None
    detected_location: str | None = None
    detected_total_experience_years: float | None = None
    detected_education_level: int | None = None
    detected_skills_count: int = 0
    detected_experience_entries: int = 0


class ProcessingProgressStep(str, Enum):
    """Steps in the document processing pipeline for UI visualization."""
    FILE_UPLOADED = "file_uploaded"
    TYPE_DETECTED = "type_detected"
    CONTENT_EXTRACTED = "content_extracted"
    RESUME_DETECTED = "resume_detected"
    INFORMATION_IDENTIFIED = "information_identified"
    SKILLS_IDENTIFIED = "skills_identified"
    EXPERIENCE_IDENTIFIED = "experience_identified"
    CAREER_DOMAIN_IDENTIFIED = "career_domain_identified"
    PROCESSING_COMPLETE = "processing_complete"


class ProcessingProgress(BaseModel):
    """Real-time processing progress for UI display."""
    current_step: ProcessingProgressStep
    completed_steps: list[ProcessingProgressStep] = Field(default_factory=list)
    is_complete: bool = False
    has_error: bool = False
    error_message: str | None = None
    progress_percentage: float = 0.0


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


class CareerDomain(BaseModel):
    """Detected professional career domain."""
    domain_id: str
    domain_name: str
    confidence: float
    related_domains: list[str] = Field(default_factory=list)


class ParsedResumeV2(BaseModel):
    """Enhanced resume schema with V2 intelligence fields."""
    candidate_id: str
    raw_text: str
    
    # Original fields (backward compatible)
    skills: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    total_experience_years: float = 0.0
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)
    
    # V2 enhancements - Document validation
    validation_state: DocumentValidationState = DocumentValidationState.VALID_RESUME
    document_type: DocumentType = DocumentType.RESUME
    extraction_quality_score: float = 1.0
    
    # V2 enhancements - Personal info extraction
    detected_name: str | None = None
    detected_email: str | None = None
    detected_phone: str | None = None
    
    # V2 enhancements - Career intelligence
    career_domains: list[CareerDomain] = Field(default_factory=list)
    primary_domain: str | None = None
    seniority_level: str | None = None  # entry, mid, senior, executive
    
    # V2 enhancements - Location preferences (more flexible)
    current_location: str | None = None
    preferred_countries: list[str] = Field(default_factory=list)
    open_to_remote: bool = True
    open_to_relocation: bool = False
    
    # V2 enhancements - Metadata
    processing_metadata: dict[str, Any] = Field(default_factory=dict)
    model_version: str = "v2.0.0"


class ResumeIn(BaseModel):
    candidate_id: str
    raw_text: str
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)
    min_salary: int | None = None
    # V2 additions
    current_location: str | None = None
    preferred_countries: list[str] = Field(default_factory=list)
    open_to_remote: bool = True
    open_to_relocation: bool = False


# Alias for backward compatibility - existing code uses ParsedResume
ParsedResume = ParsedResumeV2


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


class EligibilityState(str, Enum):
    """Eligibility determination for a job application."""
    ELIGIBLE = "eligible"  # Candidate has sufficient evidence to apply
    PARTIALLY_ELIGIBLE = "partially_eligible"  # Some gaps but may still apply with tailoring
    NOT_ELIGIBLE = "not_eligible"  # Missing critical required skills


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
    # V2 eligibility fields - separate from ranking score
    eligibility_state: EligibilityState = EligibilityState.ELIGIBLE
    skill_coverage_ratio: float = 0.0
    missing_critical_skills: list[str] = Field(default_factory=list)


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
