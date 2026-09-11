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
            assert recs[0]["external_url"]
            assert recs[0]["external_url"].startswith("https://www.linkedin.com/jobs/search")
            assert "TestCo" not in recs[0]["external_url"]

            mumbai_resp = await client.post(
                "/api/v1/candidates/resume",
                json={
                    "candidate_id": "test_cand_mumbai",
                    "raw_text": "Data analyst with SQL and Tableau experience in Mumbai.",
                    "preferred_locations": ["Mumbai"],
                },
            )
            assert mumbai_resp.status_code == 201

            london_job_resp = await client.post(
                "/api/v1/jobs",
                json={
                    "job_id": "test_job_london",
                    "title": "Data Analyst",
                    "company": "Aurora Health Analytics",
                    "location": "London",
                    "domain": "healthtech",
                    "raw_description": (
                        "Required: SQL, Excel, Python, Tableau. Bachelor's degree required."
                    ),
                },
            )
            assert london_job_resp.status_code == 201
            await client.post("/api/v1/jobs/reindex")

            mumbai_rank_resp = await client.post(
                "/api/v1/recommendations/rank",
                json={"candidate_id": "test_cand_mumbai", "top_k": 10, "mode": "hybrid"},
            )
            assert mumbai_rank_resp.status_code == 200
            mumbai_rec = next(
                r for r in mumbai_rank_resp.json() if r["job_id"] == "test_job_london"
            )
            assert mumbai_rec["external_url"]
            assert "Data+Analyst" in mumbai_rec["external_url"]
            assert "Aurora" not in mumbai_rec["external_url"]
            assert "London" not in mumbai_rec["external_url"]
            assert "Mumbai" in mumbai_rec["external_url"]

            exact_url = "https://example.com/jobs/backend-engineer-123"
            job_with_url_resp = await client.post(
                "/api/v1/jobs",
                json={
                    **job_payload,
                    "job_id": "test_job_exact_url",
                    "external_url": exact_url,
                },
            )
            assert job_with_url_resp.status_code == 201
            await client.post("/api/v1/jobs/reindex")

            rank_exact_resp = await client.post(
                "/api/v1/recommendations/rank",
                json={"candidate_id": "test_cand_1", "top_k": 10, "mode": "hybrid"},
            )
            assert rank_exact_resp.status_code == 200
            exact_rec = next(
                r for r in rank_exact_resp.json() if r["job_id"] == "test_job_exact_url"
            )
            assert exact_rec["external_url"] == exact_url

            feedback_resp = await client.post(
                "/api/v1/feedback",
                json={"candidate_id": "test_cand_1", "job_id": "test_job_1", "accepted": True},
            )
            assert feedback_resp.status_code == 201
