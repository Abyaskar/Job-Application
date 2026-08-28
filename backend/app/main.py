from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import candidates, evaluation, feedback, health, intent, jobs, recommendations
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.cache import connect_to_redis
from app.db.mongo import close_mongo_connection, connect_to_mongo
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware
from app.db.mongo import get_db
from app.repositories.repositories import CandidateRepository, JobRepository
from app.services.seed import seed_demo_data

settings = get_settings()
configure_logging(debug=settings.DEBUG)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup.begin", env=settings.ENV, demo_mode=settings.DEMO_MODE)
    await connect_to_mongo()
    await connect_to_redis()
    if settings.SEED_DEMO_DATA:
        db = get_db()
        summary = await seed_demo_data(CandidateRepository(db), JobRepository(db))
        logger.info("startup.seeded", **summary)
    logger.info("startup.complete")
    yield
    await close_mongo_connection()
    logger.info("shutdown.complete")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Ranks job descriptions against a candidate's resume using hybrid "
        "semantic + rule-based scoring, with grounded RAG explanations, "
        "skill-gap analysis, and recommended next actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning("validation.error", errors=exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("unhandled.exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(health.router)
app.include_router(candidates.router, prefix=settings.API_PREFIX)
app.include_router(intent.router, prefix=settings.API_PREFIX)
app.include_router(jobs.router, prefix=settings.API_PREFIX)
app.include_router(recommendations.router, prefix=settings.API_PREFIX)
app.include_router(feedback.router, prefix=settings.API_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "status": "running",
        "docs": "/docs",
        "demo_mode": settings.DEMO_MODE,
    }
