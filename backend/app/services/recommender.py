"""
Top-level recommendation orchestration.

Flow for `rank_jobs_for_candidate` (see README "Data Flow" for the sequence
diagram):
  1. Check Redis/in-memory cache for (candidate_version, filters, mode).
  2. If miss: vector-search the job index for candidate embedding ->
     candidate shortlist (retrieval).
  3. Score every shortlisted job with the hybrid ranker (ranking).
  4. Sort, take top_k, decide recommended action per job.
  5. Generate a grounded RAG explanation lazily -- only for jobs actually
     returned to the user (not the whole shortlist), since explanation
     generation is the most expensive step per job.
  6. Cache the result set.
"""
from __future__ import annotations

import hashlib
import json

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.cache import cache_get_json, cache_set_json
from app.models.schemas import (
    Explanation,
    IntentProfile,
    ParsedJob,
    ParsedResume,
    Recommendation,
    SearchMode,
)
from app.repositories.repositories import JobRepository, RecommendationRepository
from app.services.embeddings import get_embedding_provider
from app.services.evaluation import LatencyTimer
from app.services.intent import title_alignment_score
from app.services.rag import build_explanation
from app.services.ranking import compute_uncertainty, recommend_action, score_candidate_job
from app.services.vector_index import get_job_vector_index
from app.services.learned_ranker import LearnedRanker, extract_candidate_job_features

logger = get_logger("services.recommender")


def _cache_key(
    candidate_id: str, mode: SearchMode, top_k: int, location: str | None, domain: str | None,
    intent_signature: str,
) -> str:
    raw = f"rank::{candidate_id}::{mode.value}::{top_k}::{location}::{domain}::{intent_signature}"
    return "rec:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


async def rebuild_job_index(job_repo: JobRepository) -> int:
    """Rebuilds the embedding provider fit + vector index from all jobs
    currently in the store. Called on startup and after bulk job ingestion.
    """
    jobs = await job_repo.list_jobs(limit=10000)
    provider = get_embedding_provider()
    corpus = [j.raw_description for j in jobs]
    provider.fit(corpus)

    index = get_job_vector_index()
    items = []
    for job in jobs:
        job.embedding = provider.embed(job.raw_description)
        await job_repo.upsert_job(job)
        items.append((job.job_id, job.embedding))
    index.build(items)
    logger.info("index.rebuilt", n_jobs=len(jobs))
    return len(jobs)


async def _expand_shortlist_for_intent(
    all_jobs: list[ParsedJob], intent: IntentProfile, job_repo: JobRepository, max_extra: int = 20
) -> list[ParsedJob]:
    """Vector retrieval alone is content-similarity-based: a candidate whose
    resume doesn't yet contain GenAI vocabulary but who has stated intent to
    move into GenAI engineering could have every relevant job silently
    excluded before ranking ever sees them. This backfills the shortlist
    with jobs that title-match the stated intent (even if the resume text
    doesn't strongly resemble them), so the ranking layer -- not the
    retrieval layer -- is what ultimately decides whether they surface.
    """
    existing_ids = {j.job_id for j in all_jobs}
    all_available = await job_repo.list_jobs(limit=1000)
    added = 0
    for job in all_available:
        if job.job_id in existing_ids or added >= max_extra:
            continue
        if title_alignment_score(intent, job.title, job.domain) >= 0.5:
            all_jobs.append(job)
            added += 1
    return all_jobs


async def rank_jobs_for_candidate(
    resume: ParsedResume,
    job_repo: JobRepository,
    rec_repo: RecommendationRepository,
    settings: Settings,
    top_k: int = 10,
    mode: SearchMode = SearchMode.HYBRID,
    location_filter: str | None = None,
    domain_filter: str | None = None,
    use_cache: bool = True,
    intent: IntentProfile | None = None,
    use_learned_ranker: bool = False,  # NEW: Use ML ranker if available
) -> tuple[list[Recommendation], dict]:
    meta = {"cache_hit": False, "latency_ms": 0.0, "candidates_scored": 0, "ranking_method": "hybrid_baseline"}
    intent_signature = f"{intent.role_family}:{intent.confidence}" if intent else "none"
    key = _cache_key(resume.candidate_id, mode, top_k, location_filter, domain_filter, intent_signature)

    if use_cache:
        cached = await cache_get_json(key)
        if cached:
            meta["cache_hit"] = True
            return [Recommendation(**r) for r in cached], meta

    with LatencyTimer() as timer:
        index = get_job_vector_index()
        provider = get_embedding_provider()
        if not resume.embedding:
            resume.embedding = provider.embed(resume.raw_text)

        # Retrieval: for keyword mode we still need a shortlist, so we pull
        # a broad candidate set from vector search regardless of mode and
        # let the ranking weights (not retrieval) determine the "keyword"
        # vs "vector" vs "hybrid" comparison. In a production system with a
        # true inverted index, keyword mode would retrieve via that index
        # instead -- documented in README as a scale follow-up.
        shortlist_ids = [jid for jid, _ in index.search(resume.embedding, top_k=max(top_k * 5, 30))]
        if not shortlist_ids:
            all_jobs = await job_repo.list_jobs(limit=500)
        else:
            all_jobs = [await job_repo.get_job(jid) for jid in shortlist_ids]
            all_jobs = [j for j in all_jobs if j]

        if intent is not None and intent.role_family is not None:
            all_jobs = await _expand_shortlist_for_intent(all_jobs, intent, job_repo)

        if location_filter:
            all_jobs = [j for j in all_jobs if location_filter.lower() in j.location.lower()]
        if domain_filter:
            all_jobs = [j for j in all_jobs if j.domain and domain_filter.lower() in j.domain.lower()]

        meta["candidates_scored"] = len(all_jobs)

        # Try to load learned ranker if requested
        learned_ranker: LearnedRanker | None = None
        if use_learned_ranker:
            try:
                learned_ranker = LearnedRanker.load_model()
                if learned_ranker.is_trained:
                    meta["ranking_method"] = "learned_ml"
                    logger.info("learned_ranker.loaded_for_inference")
            except FileNotFoundError:
                logger.warning("learned_ranker.not_found", fallback="hybrid_baseline")
                meta["ranking_method"] = "hybrid_baseline"
            except Exception as e:
                logger.warning("learned_ranker.load_failed", error=str(e), fallback="hybrid_baseline")
                meta["ranking_method"] = "hybrid_baseline"

        scored: list[tuple[ParsedJob, object, object]] = []
        for job in all_jobs:
            if learned_ranker is not None and learned_ranker.is_trained:
                # Use ML ranker
                features = extract_candidate_job_features(resume, job, intent)
                ml_proba = learned_ranker.predict_proba(features)
                
                # Create a ScoreBreakdown with ML probability as final_score
                breakdown, gap = score_candidate_job(resume, job, settings, mode=mode.value, intent=intent)
                breakdown.final_score = round(ml_proba, 4)  # Override with ML probability
            else:
                # Use deterministic hybrid ranker
                breakdown, gap = score_candidate_job(resume, job, settings, mode=mode.value, intent=intent)
            
            scored.append((job, breakdown, gap))

        # V2: Separate eligible and ineligible jobs BEFORE ranking
        # Eligible jobs are ranked normally; ineligible jobs are still tracked
        # but separated to avoid presenting them as normal recommendations
        eligible_jobs = [(j, b, g) for j, b, g in scored if b.eligibility_state == "eligible"]
        partially_eligible_jobs = [(j, b, g) for j, b, g in scored if b.eligibility_state == "partially_eligible"]
        not_eligible_jobs = [(j, b, g) for j, b, g in scored if b.eligibility_state == "not_eligible"]
        
        # Sort eligible jobs by final score for main recommendations
        eligible_jobs.sort(key=lambda t: t[1].final_score, reverse=True)
        
        # For top_k results, prioritize eligible jobs first
        # If not enough eligible jobs, fill with partially_eligible
        top_eligible = eligible_jobs[:top_k]
        remaining_slots = max(0, top_k - len(top_eligible))
        top_partially = partially_eligible_jobs[:remaining_slots]
        
        top = top_eligible + top_partially

        recommendations: list[Recommendation] = []
        for job, breakdown, gap in top:
            action = recommend_action(breakdown, gap)
            explanation: Explanation = build_explanation(resume, job, breakdown, gap, action, settings, intent)
            uncertainty = compute_uncertainty(resume, gap)
            rec = Recommendation(
                candidate_id=resume.candidate_id,
                job_id=job.job_id,
                job_title=job.title,
                company=job.company,
                job_location=job.location,
                job_domain=job.domain,
                score=breakdown,
                skill_gap=gap,
                action=action,
                explanation=explanation,
                uncertainty=uncertainty,
            )
            recommendations.append(rec)
            await rec_repo.save(rec)

    meta["latency_ms"] = round(timer.elapsed_ms, 2)

    if use_cache:
        await cache_set_json(key, [r.model_dump(mode="json") for r in recommendations])

    return recommendations, meta
