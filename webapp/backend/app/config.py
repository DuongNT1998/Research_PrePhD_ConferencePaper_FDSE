from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rl_chat"

    jwt_secret: str = "change-me"
    jwt_expire_min: int = 1440
    jwt_algorithm: str = "HS256"

    cors_origins: str = "*"

    research_project_path: str = ""
    agent_enabled: bool = False

    # Passed through to the research project's os.getenv() at agent build time
    # (so a single backend/.env controls Neo4j + Anthropic as well).
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "chauduong"
    neo4j_database: str = "neo4j"
    anthropic_api_key: str = "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"


@lru_cache
def get_settings() -> Settings:
    return Settings()