"""
Repository layer.

Every repository takes the Mongo-or-InMemory database object from
app.db.mongo.get_db() and uses only the shared subset of the Motor API
that InMemoryCollection also implements (see app/db/memory_store.py). No
business logic lives here -- just persistence concerns -- so services stay
testable against either backend interchangeably.
"""
from __future__ import annotations

from app.models.schemas import (
    ApplicationRecord,
    FeedbackIn,
    IntentProfile,
    ParsedJob,
    ParsedResume,
    Recommendation,
)


class CandidateRepository:
    def __init__(self, db):
        self._col = db["candidates"]

    async def upsert_resume(self, resume: ParsedResume) -> None:
        await self._col.update_one(
            {"candidate_id": resume.candidate_id},
            {"$set": resume.model_dump(mode="json")},
            upsert=True,
        )

    async def get_resume(self, candidate_id: str) -> ParsedResume | None:
        doc = await self._col.find_one({"candidate_id": candidate_id})
        return ParsedResume(**doc) if doc else None


class JobRepository:
    def __init__(self, db):
        self._col = db["jobs"]

    async def upsert_job(self, job: ParsedJob) -> None:
        await self._col.update_one(
            {"job_id": job.job_id}, {"$set": job.model_dump(mode="json")}, upsert=True
        )

    async def get_job(self, job_id: str) -> ParsedJob | None:
        doc = await self._col.find_one({"job_id": job_id})
        return ParsedJob(**doc) if doc else None

    async def list_jobs(self, limit: int = 500) -> list[ParsedJob]:
        docs = await self._col.find({}).limit(limit).to_list(limit)
        return [ParsedJob(**d) for d in docs]

    async def count(self) -> int:
        return await self._col.count_documents({})


class RecommendationRepository:
    def __init__(self, db):
        self._col = db["recommendations"]

    async def save(self, rec: Recommendation) -> None:
        await self._col.update_one(
            {"candidate_id": rec.candidate_id, "job_id": rec.job_id},
            {"$set": rec.model_dump(mode="json")},
            upsert=True,
        )

    async def list_for_candidate(self, candidate_id: str, limit: int = 50) -> list[Recommendation]:
        docs = await self._col.find({"candidate_id": candidate_id}).limit(limit).to_list(limit)
        return [Recommendation(**d) for d in docs]

    async def get(self, candidate_id: str, job_id: str) -> Recommendation | None:
        doc = await self._col.find_one({"candidate_id": candidate_id, "job_id": job_id})
        return Recommendation(**doc) if doc else None


class FeedbackRepository:
    def __init__(self, db):
        self._col = db["feedback"]

    async def save(self, feedback: FeedbackIn) -> None:
        await self._col.insert_one(feedback.model_dump(mode="json"))

    async def list_for_candidate(self, candidate_id: str) -> list[dict]:
        return await self._col.find({"candidate_id": candidate_id}).to_list(200)

    async def acceptance_rate(self) -> float:
        total = await self._col.count_documents({})
        if total == 0:
            return 0.0
        accepted = await self._col.count_documents({"accepted": {"$eq": True}})
        return round(accepted / total, 4)


class ApplicationRepository:
    def __init__(self, db):
        self._col = db["applications"]

    async def upsert(self, record: ApplicationRecord) -> None:
        await self._col.update_one(
            {"candidate_id": record.candidate_id, "job_id": record.job_id},
            {"$set": record.model_dump(mode="json")},
            upsert=True,
        )

    async def list_for_candidate(self, candidate_id: str) -> list[dict]:
        return await self._col.find({"candidate_id": candidate_id}).to_list(200)


class IntentRepository:
    """Stores the candidate's currently active career intent.

    Kept as its own collection (rather than a field on ParsedResume) since
    a candidate may explore multiple career directions against the same
    resume over time (e.g. "GenAI Engineer" this week, "ML Engineer" next
    week) without re-uploading a resume — the intent is a separate,
    independently-updatable signal.
    """

    def __init__(self, db):
        self._col = db["intents"]

    async def upsert(self, intent: IntentProfile) -> None:
        await self._col.update_one(
            {"candidate_id": intent.candidate_id},
            {"$set": intent.model_dump(mode="json")},
            upsert=True,
        )

    async def get(self, candidate_id: str) -> IntentProfile | None:
        doc = await self._col.find_one({"candidate_id": candidate_id})
        return IntentProfile(**doc) if doc else None
