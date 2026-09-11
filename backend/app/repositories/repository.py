"""
Repository layer.

All repositories work with either the real MongoDB adapter or the
in-memory demo adapter because both expose the same small async API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.models.schemas import (
    ApplicationRecord,
    FeedbackIn,
    IntentProfile,
    ParsedJob,
    ParsedResume,
    Recommendation,
)


# ============================================================
# CANDIDATES
# ============================================================

class CandidateRepository:
    def __init__(self, db):
        self._col = db["candidates"]

    async def upsert_resume(self, resume: ParsedResume) -> None:
        await self._col.update_one(
            {"candidate_id": resume.candidate_id},
            {
                "$set": resume.model_dump(mode="json"),
            },
            upsert=True,
        )

    async def get_resume(
        self,
        candidate_id: str,
    ) -> ParsedResume | None:

        doc = await self._col.find_one(
            {"candidate_id": candidate_id}
        )

        return ParsedResume(**doc) if doc else None


# ============================================================
# JOBS
# ============================================================

class JobRepository:
    def __init__(self, db):
        self._col = db["jobs"]

    async def upsert_job(self, job: ParsedJob) -> None:
        await self._col.update_one(
            {"job_id": job.job_id},
            {
                "$set": job.model_dump(mode="json"),
            },
            upsert=True,
        )

    async def get_job(
        self,
        job_id: str,
    ) -> ParsedJob | None:

        doc = await self._col.find_one(
            {"job_id": job_id}
        )

        return ParsedJob(**doc) if doc else None

    async def list_jobs(
        self,
        limit: int = 500,
    ) -> list[ParsedJob]:

        docs = (
            await self._col
            .find({})
            .limit(limit)
            .to_list(limit)
        )

        return [
            ParsedJob(**doc)
            for doc in docs
        ]

    async def count(self) -> int:
        return await self._col.count_documents({})


# ============================================================
# RECOMMENDATIONS
# ============================================================

class RecommendationRepository:
    def __init__(self, db):
        self._col = db["recommendations"]

    async def save(
        self,
        rec: Recommendation,
    ) -> None:

        await self._col.update_one(
            {
                "candidate_id": rec.candidate_id,
                "job_id": rec.job_id,
            },
            {
                "$set": rec.model_dump(mode="json"),
            },
            upsert=True,
        )

    async def list_for_candidate(
        self,
        candidate_id: str,
        limit: int = 50,
    ) -> list[Recommendation]:

        docs = (
            await self._col
            .find({"candidate_id": candidate_id})
            .limit(limit)
            .to_list(limit)
        )

        return [
            Recommendation(**doc)
            for doc in docs
        ]

    async def get(
        self,
        candidate_id: str,
        job_id: str,
    ) -> Recommendation | None:

        doc = await self._col.find_one(
            {
                "candidate_id": candidate_id,
                "job_id": job_id,
            }
        )

        return Recommendation(**doc) if doc else None


# ============================================================
# FEEDBACK
# ============================================================

class FeedbackRepository:
    def __init__(self, db):
        self._col = db["feedback"]

    async def save(
        self,
        feedback: FeedbackIn,
    ) -> None:

        await self._col.insert_one(
            feedback.model_dump(mode="json")
        )

    async def list_for_candidate(
        self,
        candidate_id: str,
    ) -> list[dict]:

        return await (
            self._col
            .find({"candidate_id": candidate_id})
            .to_list(200)
        )

    async def acceptance_rate(self) -> float:

        total = await self._col.count_documents({})

        if total == 0:
            return 0.0

        accepted = await self._col.count_documents(
            {
                "accepted": {
                    "$eq": True
                }
            }
        )

        return round(
            accepted / total,
            4,
        )


# ============================================================
# APPLICATIONS
# ============================================================

class ApplicationRepository:
    def __init__(self, db):
        self._col = db["applications"]

    async def upsert(
        self,
        record: ApplicationRecord,
    ) -> None:

        await self._col.update_one(
            {
                "candidate_id": record.candidate_id,
                "job_id": record.job_id,
            },
            {
                "$set": record.model_dump(mode="json"),
            },
            upsert=True,
        )

    async def list_for_candidate(
        self,
        candidate_id: str,
    ) -> list[dict]:

        return await (
            self._col
            .find({"candidate_id": candidate_id})
            .to_list(200)
        )


# ============================================================
# CAREER INTENT
# ============================================================

class IntentRepository:
    """
    Stores the candidate's active career intent separately
    from the resume.
    """

    def __init__(self, db):
        self._col = db["intents"]

    async def upsert(
        self,
        intent: IntentProfile,
    ) -> None:

        await self._col.update_one(
            {
                "candidate_id": intent.candidate_id
            },
            {
                "$set": intent.model_dump(mode="json")
            },
            upsert=True,
        )

    async def get(
        self,
        candidate_id: str,
    ) -> IntentProfile | None:

        doc = await self._col.find_one(
            {
                "candidate_id": candidate_id
            }
        )

        return IntentProfile(**doc) if doc else None


# ============================================================
# LEARNING CANDIDATES
# ============================================================

class LearningCandidateRepository:
    """
    Stores candidates originating from genuine user uploads.

    Demo/seed candidates are NOT stored here.

    A resume itself is not a supervised ML example.
    It becomes part of the learning population.
    """

    def __init__(self, db):
        self._col = db["learning_candidates"]

    async def save_real_candidate(
        self,
        candidate_id: str,
        resume: ParsedResume,
    ) -> None:

        now = datetime.now(
            timezone.utc
        ).isoformat()

        await self._col.update_one(
            {
                "candidate_id": candidate_id
            },
            {
                "$set": {
                    "candidate_id": candidate_id,
                    "source": "real_user_upload",
                    "training_eligible": True,
                    "updated_at": now,
                    "resume": resume.model_dump(
                        mode="json"
                    ),
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )

    async def is_training_eligible(
        self,
        candidate_id: str,
    ) -> bool:

        doc = await self._col.find_one(
            {
                "candidate_id": candidate_id,
                "source": "real_user_upload",
                "training_eligible": True,
            }
        )

        return doc is not None

    async def count_real_candidates(self) -> int:

        return await self._col.count_documents(
            {
                "source": "real_user_upload",
                "training_eligible": True,
            }
        )


# ============================================================
# TRAINING EXAMPLES
# ============================================================

class TrainingExampleRepository:
    """
    Stores supervised ML examples.

    One example represents:

        real candidate
        +
        job
        +
        observed outcome
        +
        feature vector
        +
        label
    """

    def __init__(self, db):
        self._col = db["learning_examples"]

    async def save(
        self,
        candidate_id: str,
        job_id: str,
        label: int,
        features: list[float],
        feature_names: list[str],
        source: str = "feedback",
        outcome: str | None = None,
    ) -> str:

        now = datetime.now(
            timezone.utc
        ).isoformat()

        example_id = (
            f"train_{uuid4().hex}"
        )

        document = {
            "example_id": example_id,
            "candidate_id": candidate_id,
            "job_id": job_id,
            "label": int(label),
            "features": features,
            "feature_names": feature_names,
            "source": source,
            "outcome": outcome,
            "created_at": now,
        }

        await self._col.insert_one(
            document
        )

        return example_id

    async def list_examples(
        self,
        limit: int = 100000,
    ) -> list[dict]:

        return await (
            self._col
            .find({})
            .sort("created_at", 1)
            .limit(limit)
            .to_list(limit)
        )

    async def count(self) -> int:

        return await self._col.count_documents({})