from app.models.schemas import (
    EducationEntry,
    ExtractedRequirements,
    IntentMatchMethod,
    ParsedJob,
    ParsedResume,
)
from app.core.config import Settings
from app.services.intent import normalize_intent, title_alignment_score
from app.services.ranking import score_candidate_job


def test_normalize_intent_exact_alias_match():
    profile = normalize_intent("c1", "genai engineer")
    assert profile.role_family == "genai_engineer"
    assert profile.match_method == IntentMatchMethod.EXACT_ALIAS
    assert profile.confidence == 1.0
    assert "llm" in profile.intent_skills


def test_normalize_intent_case_and_whitespace_insensitive():
    profile = normalize_intent("c1", "  GenAI Engineer  ")
    assert profile.role_family == "genai_engineer"


def test_normalize_intent_fuzzy_typo():
    profile = normalize_intent("c1", "Data Analist")  # typo
    assert profile.role_family == "data_analyst"
    assert profile.match_method in (IntentMatchMethod.FUZZY_ALIAS, IntentMatchMethod.EXACT_ALIAS)


def test_normalize_intent_keyword_overlap_free_text():
    profile = normalize_intent(
        "c1", "I want to work with large language models and retrieval augmented generation systems"
    )
    assert profile.role_family == "genai_engineer"


def test_normalize_intent_business_analyst_vs_data_analyst_distinct():
    ba = normalize_intent("c1", "business analyst")
    da = normalize_intent("c1", "data analyst")
    assert ba.role_family == "business_analyst"
    assert da.role_family == "data_analyst"
    assert ba.role_family != da.role_family


def test_title_alignment_exact_title_match():
    profile = normalize_intent("c1", "GenAI Engineer")
    score = title_alignment_score(profile, "AI Engineer - Generative AI Platform", "ai_platform")
    assert score == 1.0


def test_title_alignment_unrelated_title_scores_low():
    profile = normalize_intent("c1", "GenAI Engineer")
    score = title_alignment_score(profile, "Data Analyst", "healthtech")
    assert score < 0.5


def test_title_alignment_no_intent_returns_neutral():
    score = title_alignment_score(None, "Any Job Title", "any_domain")
    assert score == 1.0


def _sample_resume(skills, years=2.0) -> ParsedResume:
    return ParsedResume(
        candidate_id="c1",
        raw_text="Sample resume text with enough length to avoid thinness penalties in tests here.",
        skills=skills,
        education=[EducationEntry(degree="Bachelor's", level=1)],
        total_experience_years=years,
        embedding=[0.1, 0.2, 0.3],
    )


def _sample_job(job_id, title, required_skills, domain="ai_platform") -> ParsedJob:
    return ParsedJob(
        job_id=job_id,
        title=title,
        company="TestCo",
        location="Remote",
        domain=domain,
        raw_description="Sample job description text.",
        requirements=ExtractedRequirements(required_skills=required_skills, min_experience_years=1.0),
        embedding=[0.1, 0.2, 0.3],
    )


def test_intent_gating_caps_score_for_off_intent_job():
    settings = Settings(DEMO_MODE=True)
    resume = _sample_resume(["python", "sql", "excel"])
    intent = normalize_intent("c1", "GenAI Engineer")

    off_intent_job = _sample_job("j1", "Data Analyst", ["python", "sql", "excel"], domain="healthtech")
    breakdown, _ = score_candidate_job(resume, off_intent_job, settings, mode="hybrid", intent=intent)

    assert breakdown.intent_gated is True
    assert breakdown.final_score <= 0.35


def test_intent_alignment_does_not_gate_on_intent_matching_job():
    settings = Settings(DEMO_MODE=True)
    resume = _sample_resume(["python", "llm", "rag"])
    intent = normalize_intent("c1", "GenAI Engineer")

    on_intent_job = _sample_job("j2", "GenAI Engineer", ["python", "llm", "rag"], domain="ai_platform")
    breakdown, _ = score_candidate_job(resume, on_intent_job, settings, mode="hybrid", intent=intent)

    assert breakdown.intent_gated is False
    assert breakdown.intent_alignment == 1.0


def test_no_intent_does_not_gate():
    settings = Settings(DEMO_MODE=True)
    resume = _sample_resume(["python", "sql"])
    job = _sample_job("j3", "Data Analyst", ["python", "sql"], domain="healthtech")
    breakdown, _ = score_candidate_job(resume, job, settings, mode="hybrid", intent=None)
    assert breakdown.intent_gated is False
    assert breakdown.intent_alignment == 1.0
