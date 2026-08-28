"""
Central application configuration.

All configuration is environment-driven (12-factor app) so the exact same
container image can run in local dev, docker-compose, and GCP Cloud Run
without rebuilding. Defaults are safe for a zero-dependency local demo
(DEMO_MODE=true uses in-memory Mongo/Redis substitutes), which is what lets
a reviewer clone this repo and run it in under a minute without spinning up
infrastructure first.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "Job Application Strategy AI"
    ENV: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # --- Demo / infra toggles ---
    # DEMO_MODE=true swaps MongoDB/Redis for in-process implementations of the
    # same repository/cache interfaces. This keeps the service boundary
    # honest (same interface, different adapter) while letting the project
    # be reviewed without docker. Set to false in docker-compose / GCP.
    DEMO_MODE: bool = True
    SEED_DEMO_DATA: bool = True

    # --- MongoDB ---
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "job_strategy_ai"

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 60 * 30  # 30 min for recommendation results
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # --- AI providers (pluggable) ---
    # "local" providers require no network/API key and are used by default so
    # the project is reviewable offline. "vertex_ai" / "openai" / "anthropic"
    # are implemented behind the same interface for a one-line prod swap.
    EMBEDDING_PROVIDER: str = "local_tfidf"
    LLM_PROVIDER: str = "local_template"
    EMBEDDING_DIM: int = 128
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GCP_PROJECT_ID: str | None = None

    # --- Ranking weights (see README "Ranking Formula") ---
    W_INTENT: float = 0.20
    W_SEMANTIC: float = 0.25
    W_SKILL: float = 0.25
    W_EXPERIENCE: float = 0.12
    W_EDUCATION: float = 0.08
    W_LOCATION: float = 0.10

    # --- Security ---
    JWT_SECRET: str = "dev-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24

    # --- CORS ---
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
