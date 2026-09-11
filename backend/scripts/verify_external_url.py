"""One-off verification script for external_url discovery behavior."""
from __future__ import annotations

import asyncio

from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.main import app


async def main() -> None:
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/candidates/resume",
                json={
                    "candidate_id": "live_mumbai_cand",
                    "raw_text": "Data analyst with SQL, Tableau, Python. Based in Mumbai, India.",
                    "preferred_locations": ["Mumbai"],
                },
            )
            resp = await client.post(
                "/api/v1/recommendations/rank",
                json={"candidate_id": "live_mumbai_cand", "top_k": 15, "mode": "hybrid"},
            )
            recs = resp.json()
            job010 = next((r for r in recs if r["job_id"] == "job_010"), None)
            if not job010:
                raise SystemExit("job_010 missing from rank results")

            url = job010["external_url"]
            print("job_010 external_url:", url)
            print(
                "checks:",
                {
                    "non_null": bool(url),
                    "has_data_analyst": "Data+Analyst" in url,
                    "has_mumbai": "Mumbai" in url,
                    "no_aurora": "Aurora" not in url,
                    "no_london": "London" not in url,
                },
            )
            for r in recs:
                if not r.get("external_url"):
                    raise SystemExit(f"missing external_url for {r['job_id']}")
            print("all recommendations have external_url: OK")


if __name__ == "__main__":
    asyncio.run(main())
