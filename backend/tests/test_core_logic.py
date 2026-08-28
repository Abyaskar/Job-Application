from app.models.schemas import ExtractedRequirements
from app.services.evaluation import RankedResult, ndcg_at_k, precision_at_k, recall_at_k
from app.services.extraction import extract_job_requirements, extract_resume_profile
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
