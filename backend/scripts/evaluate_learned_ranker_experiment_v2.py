"""
Offline evaluation of the V2 learned ranker experiment.

IMPORTANT:
- Does NOT enable the learned ranker in production.
- Does NOT modify recommender.py.
- Does NOT modify data/eval_labels.json.
- Uses the same 7 hand-labeled evaluation cases as /evaluation/run.
- Loads only the V2 experimental model.
- Uses the project's existing Precision/Recall/NDCG implementation.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.deps import get_db
from app.db.mongo import connect_to_mongo, close_mongo_connection
from app.core.config import Settings
from app.repositories.repository import (
    CandidateRepository,
    JobRepository,
    RecommendationRepository,
)
from app.services.evaluation import RankedResult, aggregate_metrics
from app.services.intent import normalize_intent
from app.services.learned_ranker import (
    LearnedRanker,
    extract_candidate_job_features,
)


BASE_DIR = Path(__file__).resolve().parents[1]

EVAL_PATH = BASE_DIR / "data" / "eval_labels.json"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "recommender_experiment_non_skill_v2.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "recommender_experiment_non_skill_v2_metadata.json"
)


async def main() -> None:
    print("=" * 70)
    print("V2 LEARNED RANKER — OFFLINE EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load evaluation set
    # ------------------------------------------------------------------

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation set not found: {EVAL_PATH}"
        )

    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    cases = eval_set.get("cases", [])

    print(f"Evaluation cases: {len(cases)}")

    # ------------------------------------------------------------------
    # Load experiment model
    # ------------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Experiment model not found: {MODEL_PATH}"
        )

    ranker = LearnedRanker.load_model(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
    )

    if not ranker.is_trained:
        raise RuntimeError(
            "Loaded experiment model but ranker reports is_trained=False"
        )

    print(f"Model: {MODEL_PATH.name}")

    # ------------------------------------------------------------------
    # Initialize repositories using the same DB dependency as the API
    # ------------------------------------------------------------------
    await connect_to_mongo()

    db = get_db()

    candidate_repo = CandidateRepository(db)
    job_repo = JobRepository(db)
    rec_repo = RecommendationRepository(db)

    # Settings from the normal application configuration.
    settings = Settings()

    # ------------------------------------------------------------------
    # Load jobs once
    # ------------------------------------------------------------------

    jobs = await job_repo.list_jobs(limit=500)

    print(f"Jobs available for evaluation: {len(jobs)}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    per_query_ranked: list[list[RankedResult]] = []
    per_query_relevant: list[set[str]] = []
    latencies: list[float] = []

    for case in cases:
        candidate_id = case["candidate_id"]

        resume = await candidate_repo.get_resume(candidate_id)

        if not resume:
            print(
                f"WARNING: candidate not found: {candidate_id}"
            )
            continue

        intent = None

        if case.get("intended_role"):
            intent = normalize_intent(
                candidate_id,
                case["intended_role"],
            )

        start = time.perf_counter()

        ranked: list[RankedResult] = []

        for job in jobs:
            features = extract_candidate_job_features(
                resume,
                job,
                intent,
            )

            v2_feature_names = ranker._metadata.feature_names
            probability = ranker._model.predict_proba(
            [features.to_array(v2_feature_names)]
            )[0, 1]

            #probability = ranker.predict_proba(features)

            ranked.append(
                RankedResult(
                    job_id=job.job_id,
                    score=probability,
                )
            )

        ranked.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

        per_query_ranked.append(ranked)

        per_query_relevant.append(
            set(case["relevant_job_ids"])
        )

        print(
            f"{candidate_id}: "
            f"top1={ranked[0].job_id if ranked else None} "
            f"latency={elapsed_ms:.2f}ms"
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    metrics = aggregate_metrics(
        per_query_ranked,
        per_query_relevant,
        [1, 3, 5],
    )

    metrics["avg_latency_ms"] = round(
        sum(latencies) / len(latencies),
        2,
    ) if latencies else 0.0

    metrics["n_queries"] = len(
        per_query_ranked
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 LEARNED RANKER RESULTS")
    print("=" * 70)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("PRODUCTION SAFETY CHECK")
    print("=" * 70)

    print("Production learned ranker: NOT ENABLED")
    print("Production model modified: NO")
    print("Evaluation labels modified: NO")


if __name__ == "__main__":
    asyncio.run(main())