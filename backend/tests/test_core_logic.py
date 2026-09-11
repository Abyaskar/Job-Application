from app.models.schemas import ExtractedRequirements, ParsedResume
from app.services.evaluation import RankedResult, ndcg_at_k, precision_at_k, recall_at_k
from app.services.extraction import extract_job_requirements, extract_resume_profile
from app.services.job_discovery import get_candidate_search_location, resolve_job_external_url
from app.services.ranking import (
    compute_education_match,
    compute_experience_match,
    compute_hard_skill_match,
    compute_location_match,
)
from app.services.taxonomy import extract_skills


def test_extract_skills_finds_known_aliases():
    text = "Experience with Python, FastAPI, MongoDB and vector search / embeddings."
    skills = extract_skills(text)
    assert "python" in skills
    assert "fastapi" in skills
    assert "mongodb" in skills
    assert "vector_search" in skills


def test_extract_skills_avoids_substring_false_positive():
    # "r" should not match inside "framework" style words for single-letter aliases
    skills = extract_skills("We use a modern framework for our backend.")
    assert "r" not in skills


def test_extract_job_requirements_splits_required_vs_preferred():
    jd = (
        "Required: 3+ years experience, strong Python and FastAPI skills. "
        "Nice to have: experience with Docker and GCP."
    )
    reqs = extract_job_requirements(jd)
    assert "python" in reqs.required_skills
    assert "fastapi" in reqs.required_skills
    assert "docker" in reqs.preferred_skills
    assert "docker" not in reqs.required_skills
    assert reqs.min_experience_years == 3.0


def test_extract_resume_profile_years_and_education():
    text = "MSc Data Science graduate with 2 years of experience in Python and machine learning."
    profile = extract_resume_profile(text)
    assert profile["total_experience_years"] == 2.0
    assert any(e.level == 2 for e in profile["education"])
    assert "python" in profile["skills"]


def test_hard_skill_match_full_coverage_scores_high():
    reqs = ExtractedRequirements(required_skills=["python", "fastapi"], preferred_skills=["docker"])
    score, gap = compute_hard_skill_match(["python", "fastapi", "docker"], reqs)
    assert score == 1.0
    assert gap.coverage_ratio == 1.0
    assert gap.missing_required == []


def test_hard_skill_match_partial_coverage_scores_lower():
    reqs = ExtractedRequirements(required_skills=["python", "fastapi", "mongodb"], preferred_skills=[])
    score, gap = compute_hard_skill_match(["python"], reqs)
    assert 0 < score < 1
    assert gap.missing_required == ["fastapi", "mongodb"]
    assert gap.coverage_ratio == round(1 / 3, 4)


def test_experience_match_under_requirement_has_floor():
    score = compute_experience_match(candidate_years=1, required_years=5)
    assert 0.15 <= score < 0.5


def test_experience_match_meets_requirement_scores_high():
    score = compute_experience_match(candidate_years=5, required_years=5)
    assert score >= 0.9


def test_education_match_exceeding_requirement_is_perfect():
    assert compute_education_match(candidate_level=2, required_level=1) == 1.0


def test_education_match_below_requirement_is_penalized():
    score = compute_education_match(candidate_level=1, required_level=3)
    assert score < 1.0


def test_location_match_remote_job_always_matches():
    assert compute_location_match(["London"], "Remote") == 1.0


def test_location_match_mismatch_scores_low():
    assert compute_location_match(["London"], "Singapore") < 0.5


def test_precision_recall_ndcg_perfect_ranking():
    ranked = [RankedResult("a", 0.9), RankedResult("b", 0.8), RankedResult("c", 0.7)]
    relevant = {"a", "b"}
    assert precision_at_k(ranked, relevant, 2) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert ndcg_at_k(ranked, relevant, 2) == 1.0


def test_ndcg_penalizes_relevant_item_ranked_lower():
    ranked_good = [RankedResult("a", 0.9), RankedResult("b", 0.1)]
    ranked_bad = [RankedResult("b", 0.9), RankedResult("a", 0.1)]
    relevant = {"a"}
    assert ndcg_at_k(ranked_good, relevant, 2) == 1.0
    assert ndcg_at_k(ranked_bad, relevant, 2) < 1.0


def _sample_resume(**overrides) -> ParsedResume:
    data = {
        "candidate_id": "test_cand",
        "raw_text": "Data analyst based in Mumbai with SQL and Python skills.",
        "preferred_locations": ["Mumbai"],
    }
    data.update(overrides)
    return ParsedResume(**data)


def test_get_candidate_search_location_prefers_current_location():
    resume = _sample_resume(current_location="Delhi, India", preferred_locations=["Mumbai"])
    assert get_candidate_search_location(resume) == "Delhi, India"


def test_resolve_job_external_url_preserves_exact_source_url():
    exact = "https://example.com/real-job/123"
    resume = _sample_resume(preferred_locations=["Mumbai"])
    assert (
        resolve_job_external_url(
            title="Data Analyst",
            external_url=exact,
            resume=resume,
        )
        == exact
    )


def test_resolve_job_external_url_uses_role_and_candidate_location():
    resume = _sample_resume(preferred_locations=["Mumbai"])
    url = resolve_job_external_url(
        title="Data Analyst",
        external_url=None,
        resume=resume,
        posted_within_days=7,
    )
    assert url.startswith("https://www.linkedin.com/jobs/search?")
    assert "Data+Analyst" in url
    assert "Aurora" not in url
    assert "London" not in url
    assert "location=Mumbai" in url
    assert "f_TPR=r604800" in url


def test_resolve_job_external_url_uses_delhi_not_job_location():
    resume = _sample_resume(current_location="Delhi, India", preferred_locations=["Mumbai"])
    url = resolve_job_external_url(
        title="Data Scientist",
        external_url=None,
        resume=resume,
        posted_within_days=1,
    )
    assert "Data+Scientist" in url
    assert "Delhi" in url
    assert "Mumbai" not in url
    assert "London" not in url
    assert "Aurora" not in url
    assert "f_TPR=r86400" in url


def test_resolve_job_external_url_global_omits_location():
    resume = _sample_resume(preferred_locations=["Mumbai"])
    url = resolve_job_external_url(
        title="Data Analyst",
        external_url=None,
        resume=resume,
        posted_within_days=30,
        discovery_location="global",
    )
    assert "Data+Analyst" in url
    assert "location=" not in url
    assert "Mumbai" not in url
    assert "London" not in url
    assert "f_TPR=r2592000" in url


def test_resolve_job_external_url_handles_remote_candidate_location():
    resume = _sample_resume(preferred_locations=["Remote"], open_to_remote=True)
    url = resolve_job_external_url(
        title="Backend Engineer",
        external_url=None,
        resume=resume,
        posted_within_days=7,
    )
    assert "f_WT=2" in url
    assert "Backend+Engineer" in url
    assert "location=" not in url
    assert "f_TPR=r604800" in url


def test_fresher_resume_without_work_experience_is_sufficient():
    from app.services.document_intelligence import (
        extract_personal_info,
        validate_resume_sufficiency,
    )

    text = """
    Viraj Sharma
    viraj.sharma@gmail.com | Mumbai, Maharashtra
    LinkedIn: linkedin.com/in/viraj

    EDUCATION
    B.Tech Artificial Intelligence and Machine Learning
    University of Mumbai, 2021-2025

    PROJECTS
    1. AI Chatbot using Python, FastAPI and RAG with vector search
    2. Image classification with TensorFlow and PyTorch
    3. Data analysis dashboard with SQL, pandas and Power BI

    TECHNICAL SKILLS
    Python, SQL, Machine Learning, Deep Learning, TensorFlow, PyTorch,
    FastAPI, MongoDB, Docker, Git, Excel, Tableau

    CERTIFICATIONS
    Google Data Analytics Certificate
    AWS Cloud Practitioner

    EXTRACURRICULAR
    Hackathon winner, AI Club lead
    """
    info = extract_personal_info(text)
    ok, missing = validate_resume_sufficiency(text, info)
    assert ok is True
    assert "work experience" not in missing
    assert info["email"] == "viraj.sharma@gmail.com"
    assert info["location"] and "Mumbai" in info["location"]


def test_spaced_pdf_email_is_extracted():
    from app.services.document_intelligence import extract_personal_info

    text = "Viraj\nviraj.sharma @ gmail.com\nMumbai, Maharashtra\nSkills: Python SQL"
    info = extract_personal_info(text)
    assert info["email"] == "viraj.sharma@gmail.com"


def test_skills_detected_via_taxonomy_without_skills_heading():
    from app.services.document_intelligence import (
        extract_personal_info,
        validate_resume_sufficiency,
    )

    text = """
    Viraj Sharma
    contact: viraj@example.com
    Mumbai

    Education
    Bachelor of Technology in Computer Science, Mumbai University

    Projects
    Built an ML pipeline with Python, TensorFlow, Docker and MongoDB
    Implemented FastAPI services and Redis caching for inference

    Certifications
    Coursera Machine Learning
    """ + (" detail" * 40)
    info = extract_personal_info(text)
    ok, missing = validate_resume_sufficiency(text, info)
    assert ok is True
    assert "skills" not in missing
    assert "work experience" not in missing
