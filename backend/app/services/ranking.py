"""
Hybrid ranking model — the Strategy Ranking Layer.

Ranking formula
----------------
final_score = w_intent * intent_alignment
            + w_sem    * semantic_similarity
            + w_skill  * hard_skill_match
            + w_exp    * experience_match
            + w_edu    * education_match
            + w_loc    * location_match

Each component is normalized to [0, 1] before weighting so the weights are
directly interpretable as "percentage of the final score this signal can
contribute." Default weights (see Settings) give career-intent alignment
and hard skill match together just under half the score (40%), semantic
similarity 25%, and experience/education/location the rest as qualifying
and tie-breaking signals — nothing here is a binary filter *except* intent,
which is a deliberate exception (see below).

Why intent gets special treatment, not just another weighted component:
without it, a resume that happens to share vocabulary with an unrelated
job (a backend engineer's resume and a data-engineer JD both say "Python,"
"distributed systems," "cloud") can still rank respectably on semantic
similarity + skill overlap alone, even though the candidate explicitly said
they want to move into GenAI engineering. `intent_alignment` (see
`services/intent.title_alignment_score`) is a deterministic, inspectable
function of the job's *title/domain* against the candidate's resolved role
family — not another fuzzy similarity score layered on top of
`semantic_similarity`. When intent is present and a job's alignment score
is very low (<0.2), the job is *gated*: capped at a low final score
regardless of how well the resume text otherwise matches, and flagged via
`ScoreBreakdown.intent_gated=True` so the UI can show why. This is the
concrete mechanism behind "don't recommend unrelated roles just because of
generic resume similarity."

Why not pure semantic similarity? Embedding similarity captures topical
closeness ("this resume and this JD talk about similar things") but not
*qualification* -- a resume that discusses ML broadly can score high
against an ML job description even if it's missing every specific required
tool. Hard skill match against the extracted taxonomy corrects for that.

Why not pure keyword/skill match? It misses transferable experience and
adjacent-skill relevance (e.g. "distributed systems" experience is
relevant to a "microservices" requirement even without an exact keyword
hit), which semantic similarity captures.

Hybrid = keyword-match precision + embedding-similarity recall + intent
gating for role relevance, combined — this is evaluated explicitly against
each pure strategy in `evaluation.py` / `evaluation/run_eval.py`, including
an intent-aware vs. intent-naive comparison (see README "Evaluation").
"""
from __future__ import annotations

from app.core.config import Settings
from app.models.schemas import (
    ExtractedRequirements,
    IntentProfile,
    ParsedJob,
    ParsedResume,
    RecommendedAction,
    ScoreBreakdown,
    SkillGap,
)
from app.services.embeddings import cosine_similarity
from app.services.intent import title_alignment_score

INTENT_GATE_THRESHOLD = 0.2  # below this alignment, cap the final score regardless of other signals
INTENT_GATE_CAP = 0.35  # the final-score ceiling applied to gated (off-intent) jobs

# Eligibility thresholds - separate from ranking scores
ELIGIBILITY_MIN_SKILL_COVERAGE = 0.5  # Minimum required skill coverage to be eligible (50%)
ELIGIBILITY_SOFT_THRESHOLD = 0.3  # Below this, definitely not eligible unless other factors compensate

def _normalize_skill(skill: str) -> str:
    """Normalize skill names so equivalent formats can match."""
    import re

    text = str(skill).strip().lower()

    # Convert common separators to underscores.
    text = re.sub(r"[\s./\\-]+", "_", text)

    # Keep letters, numbers, #, + and underscores.
    text = re.sub(r"[^a-z0-9_+#]", "", text)

    # Collapse repeated underscores.
    text = re.sub(r"_+", "_", text).strip("_")

    return text

def compute_hard_skill_match(
    candidate_skills: list[str], requirements: ExtractedRequirements
) -> tuple[float, SkillGap]:

    # Normalize both sides before comparison.
    candidate_normalized = {
        _normalize_skill(skill)
        for skill in candidate_skills
        if str(skill).strip()
    }

    required_normalized = {
        _normalize_skill(skill)
        for skill in requirements.required_skills
        if str(skill).strip()
    }

    preferred_normalized = {
        _normalize_skill(skill)
        for skill in requirements.preferred_skills
        if str(skill).strip()
    }

    matched_required = sorted(candidate_normalized & required_normalized)
    missing_required = sorted(required_normalized - candidate_normalized)
    missing_preferred = sorted(preferred_normalized - candidate_normalized)

    if required_normalized:
        required_coverage = len(matched_required) / len(required_normalized)
    else:
        required_coverage = 1.0

    preferred_bonus = 0.0

    if preferred_normalized:
        preferred_bonus = (
            0.15
            * (
                len(candidate_normalized & preferred_normalized)
                / len(preferred_normalized)
            )
        )

    score = min(1.0, required_coverage + preferred_bonus)

    gap = SkillGap(
        missing_required=missing_required,
        missing_preferred=missing_preferred,
        matched_required=matched_required,
        coverage_ratio=round(required_coverage, 4),
    )

    return round(score, 4), gap

def determine_eligibility(
    skill_gap: SkillGap,
    required_skills_count: int,
) -> tuple["EligibilityState", list[str]]:
    """Determine if a candidate is eligible for a job based on required skill coverage.
    
    This is an EXPLICIT GATE that operates before ranking.
    A high semantic score cannot override missing critical skills.
    
    Returns:
        tuple of (eligibility_state, missing_critical_skills)
    """
    # If no required skills specified, assume eligible (let ranking decide)
    if required_skills_count == 0:
        from app.models.schemas import EligibilityState
        return EligibilityState.ELIGIBLE, []
    
    coverage = skill_gap.coverage_ratio
    missing_required = skill_gap.missing_required
    
    # NOT ELIGIBLE: Less than 30% of required skills
    if coverage < ELIGIBILITY_SOFT_THRESHOLD:
        from app.models.schemas import EligibilityState
        return EligibilityState.NOT_ELIGIBLE, missing_required
    
    # PARTIALLY ELIGIBLE: 30-50% coverage - may apply with significant tailoring
    if coverage < ELIGIBILITY_MIN_SKILL_COVERAGE:
        from app.models.schemas import EligibilityState
        # Critical missing skills are the first 2-3 most important missing ones
        critical_missing = missing_required[:3] if len(missing_required) > 2 else missing_required
        return EligibilityState.PARTIALLY_ELIGIBLE, critical_missing
    
    # ELIGIBLE: 50%+ coverage
    from app.models.schemas import EligibilityState
    return EligibilityState.ELIGIBLE, []


def compute_experience_match(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 1.0
    if candidate_years >= required_years:
        # small bonus for meeting/exceeding, capped — extra years beyond
        # +3 don't keep increasing the score (diminishing returns, avoids
        # over-penalizing early-career candidates relative to very senior ones)
        excess = min(candidate_years - required_years, 3)
        return round(min(1.0, 0.9 + 0.033 * excess), 4)
    # linear falloff for being under the bar, floor at 0.15 so a candidate
    # isn't zeroed out for being e.g. 1 year under a 5-year requirement
    ratio = candidate_years / required_years
    return round(max(0.15, ratio), 4)


def compute_education_match(candidate_level: int, required_level: int) -> float:
    if required_level <= 0:
        return 1.0
    if candidate_level >= required_level:
        return 1.0
    gap = required_level - candidate_level
    return round(max(0.2, 1 - 0.35 * gap), 4)


def compute_location_match(candidate_prefs: list[str], job_location: str) -> float:
    if not candidate_prefs:
        return 0.7  # neutral-ish when candidate expressed no preference
    job_location_lower = job_location.lower()
    if job_location_lower == "remote":
        return 1.0
    if any(p.lower() in job_location_lower or job_location_lower in p.lower() for p in candidate_prefs):
        return 1.0
    return 0.3


def score_candidate_job(
    resume: ParsedResume,
    job: ParsedJob,
    settings: Settings,
    mode: str = "hybrid",
    intent: IntentProfile | None = None,
) -> tuple[ScoreBreakdown, SkillGap]:
    semantic = cosine_similarity(resume.embedding, job.embedding) if mode != "keyword" else 0.0
    semantic = max(0.0, semantic)  # cosine can be negative in this vector space; floor for interpretability

    skill_score, gap = compute_hard_skill_match(resume.skills, job.requirements)
    exp_score = compute_experience_match(resume.total_experience_years, job.requirements.min_experience_years)
    edu_score = compute_education_match(
        max([e.level for e in resume.education], default=0), job.requirements.education_level_required
    )
    loc_score = compute_location_match(resume.preferred_locations, job.location)
    intent_score = title_alignment_score(intent, job.title, job.domain)

    if mode == "keyword":
        weights = {"intent": 0.0, "semantic": 0.0, "skill": 0.6, "experience": 0.2, "education": 0.1, "location": 0.1}
    elif mode == "vector":
        weights = {"intent": 0.0, "semantic": 0.75, "skill": 0.0, "experience": 0.1, "education": 0.05, "location": 0.1}
    else:  # hybrid
        weights = {
            "intent": settings.W_INTENT,
            "semantic": settings.W_SEMANTIC,
            "skill": settings.W_SKILL,
            "experience": settings.W_EXPERIENCE,
            "education": settings.W_EDUCATION,
            "location": settings.W_LOCATION,
        }

    final = (
        weights["intent"] * intent_score
        + weights["semantic"] * semantic
        + weights["skill"] * skill_score
        + weights["experience"] * exp_score
        + weights["education"] * edu_score
        + weights["location"] * loc_score
    )

    # Intent gating: a stated career intent that's clearly unrelated to this
    # job's title/domain caps the final score, independent of weighting —
    # see module docstring for why this is a deliberate exception to
    # "everything is a soft weighted signal."
    gated = intent is not None and intent.role_family is not None and intent_score <= INTENT_GATE_THRESHOLD
    if gated:
        final = min(final, INTENT_GATE_CAP)

    # V2 Eligibility Gate: Determine eligibility BEFORE applying ranking score
    # This ensures a high semantic score cannot override missing critical skills
    required_skills_count = len(job.requirements.required_skills)
    eligibility_state, missing_critical = determine_eligibility(gap, required_skills_count)
    
    breakdown = ScoreBreakdown(
        intent_alignment=round(intent_score, 4),
        semantic_similarity=round(semantic, 4),
        hard_skill_match=skill_score,
        experience_match=exp_score,
        education_match=edu_score,
        location_match=loc_score,
        final_score=round(final, 4),
        weights=weights,
        intent_gated=gated,
        # V2 eligibility fields
        eligibility_state=eligibility_state,
        skill_coverage_ratio=gap.coverage_ratio,
        missing_critical_skills=missing_critical,
    )
    return breakdown, gap


def recommend_action(score: ScoreBreakdown, gap: SkillGap) -> RecommendedAction:
    """Decision thresholds — tuned qualitatively, see README "Failure modes"
    for why these are intentionally conservative and validated against
    feedback in `evaluation.py`.
    
    V2 UPDATE: Eligibility gate now takes precedence over ranking score.
    A candidate who is NOT_ELIGIBLE cannot receive APPLY_NOW regardless of score.
    """
    # V2: Eligibility gate takes absolute precedence
    if score.eligibility_state == "not_eligible":
        return RecommendedAction.BUILD_MISSING_EVIDENCE
    
    if score.intent_gated:
        return RecommendedAction.LOW_PRIORITY
    
    # ELIGIBLE candidates with strong scores
    if score.eligibility_state == "eligible":
        if score.final_score >= 0.75 and gap.coverage_ratio >= 0.8:
            return RecommendedAction.APPLY_NOW
        if score.final_score >= 0.55 and gap.coverage_ratio < 0.8 and len(gap.missing_required) <= 2:
            return RecommendedAction.TAILOR_RESUME_FIRST
    
    # PARTIALLY_ELIGIBLE candidates need to build skills or tailor heavily
    if score.eligibility_state == "partially_eligible":
        if len(gap.missing_required) > 2:
            return RecommendedAction.BUILD_MISSING_EVIDENCE
        if score.final_score >= 0.55:
            return RecommendedAction.TAILOR_RESUME_FIRST
    
    # Fallback for low scores or many missing skills
    if score.final_score >= 0.4 and len(gap.missing_required) > 2:
        return RecommendedAction.BUILD_MISSING_EVIDENCE
    
    return RecommendedAction.LOW_PRIORITY


def compute_uncertainty(resume: ParsedResume, gap: SkillGap) -> float:
    """Heuristic confidence signal: thin resumes (few extracted skills /
    short text) and jobs with sparse extracted requirements produce
    noisier scores, so we surface that as uncertainty rather than
    presenting every score with false precision.
    """
    resume_thinness = max(0.0, 1 - len(resume.skills) / 8)
    text_thinness = max(0.0, 1 - len(resume.raw_text) / 1200)
    return round(min(1.0, 0.5 * resume_thinness + 0.5 * text_thinness), 4)
