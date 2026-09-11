from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings, get_candidate_repo, get_feedback_repo, get_job_repo, get_recommendation_repo
from app.core.config import Settings
from app.db.cache import get_cache
from app.models.schemas import SearchMode
from app.repositories.repository import CandidateRepository, FeedbackRepository, JobRepository, RecommendationRepository
from app.services.evaluation import RankedResult, aggregate_metrics
from app.services.intent import normalize_intent
from app.services.recommender import rank_jobs_for_candidate

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

EVAL_LABELS_PATH = Path(__file__).resolve().parents[3] / "data" / "eval_labels.json"


def _load_eval_set() -> dict | None:
    if not EVAL_LABELS_PATH.exists():
        return None
    with open(EVAL_LABELS_PATH) as f:
        return json.load(f)


@router.get("/cache-stats")
async def cache_stats() -> dict:
    return get_cache().stats()


@router.get("/run")
async def run_evaluation(
    top_k: int = 5,
    use_intent: bool = True,
    candidate_repo: CandidateRepository = Depends(get_candidate_repo),
    job_repo: JobRepository = Depends(get_job_repo),
    rec_repo: RecommendationRepository = Depends(get_recommendation_repo),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """Runs the labeled eval set through keyword/vector/hybrid ranking and
    returns Precision@K / Recall@K / NDCG@K / latency for each, so the
    frontend evaluation dashboard and README numbers come from the same
    live computation rather than being hand-typed.

    When `use_intent=true` (default) and a case specifies `intended_role`,
    the candidate's free-text intent is normalized via the Career Intent
    Layer and applied during ranking for every mode -- set `use_intent=false`
    to see keyword/vector/hybrid performance without intent-gating at all.
    Use `/evaluation/intent-impact` to isolate the effect of intent on the
    hybrid mode specifically.
    """
    eval_set = _load_eval_set()
    if eval_set is None:
        return {"error": "eval_labels.json not found. Run scripts/seed_demo_data.py first."}

    ks = [1, 3, 5, top_k] if top_k not in (1, 3, 5) else [1, 3, 5]
    ks = sorted(set(ks))

    results = {}
    for mode in (SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID):
        per_query_ranked = []
        per_query_relevant = []
        latencies = []
        for case in eval_set["cases"]:
            resume = await candidate_repo.get_resume(case["candidate_id"])
            if not resume:
                continue
            intent = None
            if use_intent and case.get("intended_role"):
                intent = normalize_intent(case["candidate_id"], case["intended_role"])
            recs, meta = await rank_jobs_for_candidate(
                resume=resume,
                job_repo=job_repo,
                rec_repo=rec_repo,
                settings=settings,
                top_k=max(ks),
                mode=mode,
                use_cache=False,
                intent=intent,
            )
            per_query_ranked.append([RankedResult(r.job_id, r.score.final_score) for r in recs])
            per_query_relevant.append(set(case["relevant_job_ids"]))
            latencies.append(meta["latency_ms"])

        metrics = aggregate_metrics(per_query_ranked, per_query_relevant, ks)
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        metrics["n_queries"] = len(per_query_ranked)
        results[mode.value] = metrics

    return results


@router.get("/intent-impact")
async def intent_impact(
    top_k: int = 5,
    candidate_repo: CandidateRepository = Depends(get_candidate_repo),
    job_repo: JobRepository = Depends(get_job_repo),
    rec_repo: RecommendationRepository = Depends(get_recommendation_repo),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """Isolates the effect of the Career Intent Layer: runs every eval case
    with `intended_role` set through hybrid ranking twice -- once with the
    normalized intent applied, once with intent disabled entirely -- and
    reports Precision/Recall/NDCG for both, plus which specific
    relevant-but-off-intent jobs got demoted by gating (evidence, not just
    a headline number).
    """
    eval_set = _load_eval_set()
    if eval_set is None:
        return {"error": "eval_labels.json not found."}

    ks = [1, 3, 5]
    cases_with_intent = [c for c in eval_set["cases"] if c.get("intended_role")]

    variants: dict[str, dict] = {}
    per_case_detail = []

    for variant_name, apply_intent in (("without_intent", False), ("with_intent", True)):
        per_query_ranked, per_query_relevant, latencies = [], [], []
        for case in cases_with_intent:
            resume = await candidate_repo.get_resume(case["candidate_id"])
            if not resume:
                continue
            intent = normalize_intent(case["candidate_id"], case["intended_role"]) if apply_intent else None
            recs, meta = await rank_jobs_for_candidate(
                resume=resume,
                job_repo=job_repo,
                rec_repo=rec_repo,
                settings=settings,
                top_k=max(ks),
                mode=SearchMode.HYBRID,
                use_cache=False,
                intent=intent,
            )
            per_query_ranked.append([RankedResult(r.job_id, r.score.final_score) for r in recs])
            per_query_relevant.append(set(case["relevant_job_ids"]))
            latencies.append(meta["latency_ms"])

            if variant_name == "with_intent":
                per_case_detail.append(
                    {
                        "candidate_id": case["candidate_id"],
                        "intended_role": case["intended_role"],
                        "top_job_with_intent": recs[0].job_id if recs else None,
                        "top_job_title_with_intent": recs[0].job_title if recs else None,
                        "gated_jobs": [r.job_id for r in recs if r.score.intent_gated],
                        "relevant_job_ids": case["relevant_job_ids"],
                    }
                )

        metrics = aggregate_metrics(per_query_ranked, per_query_relevant, ks)
        metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        metrics["n_queries"] = len(per_query_ranked)
        variants[variant_name] = metrics

    return {"variants": variants, "cases": per_case_detail}


@router.get("/summary")
async def evaluation_summary(feedback_repo: FeedbackRepository = Depends(get_feedback_repo)) -> dict:
    return {
        "cache": get_cache().stats(),
        "recommendation_acceptance_rate": await feedback_repo.acceptance_rate(),
    }
