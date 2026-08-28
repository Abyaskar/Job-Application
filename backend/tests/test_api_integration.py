import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_and_full_pipeline():
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health")
            assert health.status_code == 200

            resume_payload = {
                "candidate_id": "test_cand_1",
                "raw_text": (
                    "Python developer with 2 years experience in FastAPI, MongoDB, "
                    "and Redis. Bachelor's degree in Computer Science."
                ),
                "preferred_locations": ["Remote"],
            }
            resume_resp = await client.post("/api/v1/candidates/resume", json=resume_payload)
            assert resume_resp.status_code == 201
            assert "python" in resume_resp.json()["skills"]

            job_payload = {
                "job_id": "test_job_1",
                "title": "Backend Engineer",
                "company": "TestCo",
                "location": "Remote",
                "raw_description": (
                    "Required: 1+ years experience, Python, FastAPI, MongoDB. "
                    "Bachelor's degree required."
                ),
            }
            job_resp = await client.post("/api/v1/jobs", json=job_payload)
            assert job_resp.status_code == 201

            await client.post("/api/v1/jobs/reindex")

            rank_resp = await client.post(
                "/api/v1/recommendations/rank",
                json={"candidate_id": "test_cand_1", "top_k": 5, "mode": "hybrid"},
            )
            assert rank_resp.status_code == 200
            recs = rank_resp.json()
            assert len(recs) >= 1
            assert recs[0]["explanation"]["grounded"] is True
            assert "action" in recs[0]

            feedback_resp = await client.post(
                "/api/v1/feedback",
                json={"candidate_id": "test_cand_1", "job_id": "test_job_1", "accepted": True},
            )
            assert feedback_resp.status_code == 201
