"""
Core configuration module using Pydantic Settings.

All environment variables are loaded and validated here.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Uses Pydantic Settings for validation and type conversion.
    Environment variables are loaded from .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    # Application
    APP_NAME: str = "Ibra Agent"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = True

    # Database (PostgreSQL REQUIRED)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://meta_user:meta_password@localhost:5432/meta_genai",
        description="PostgreSQL async connection URL (required)",
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DB_ECHO_SQL: bool = False

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # Meta API (optional for development)
    META_APP_ID: str = Field(default="", description="Meta App ID")
    META_APP_SECRET: str = Field(default="", description="Meta App Secret")
    INSTAGRAM_APP_SECRET: str = Field(default="", description="Instagram API App Secret (if using separate IG app)")
    META_VERIFY_TOKEN: str = Field(default="dev_verify_token", description="Webhook Verify Token")
    META_GRAPH_API_VERSION: str = "v21.0"
    META_API_BASE_URL: str = "https://graph.facebook.com"
    WEBHOOK_BASE_URL: str = Field(default="", description="Public base URL for webhook callbacks (e.g. ngrok URL)")

    # AI Configuration
    LLM_PROVIDER: str = "openai"
    GROQ_API_KEY: str = Field(default="", description="Groq API key (required when LLM_PROVIDER='groq')")
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MODEL_OPENAI: str = "gpt-4o-mini"
    LLM_MODEL_GROQ: str = "llama-3.3-70b-versatile"
    LLM_MODEL_PATH: str = "./models/qwen2.5-7b-instruct.gguf"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048  # Max response tokens (must be < LLM_CONTEXT_LENGTH)
    LLM_TOP_P: float = 0.9
    LLM_CONTEXT_LENGTH: int = 8192  # Total context window (Llama 3.x 70B supports 8192)

    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE: str = "cpu"

    # Vector Database
    CHROMADB_PERSIST_DIR: str = "./data/chromadb"
    CHROMA_PERSIST_DIRECTORY: str | None = None
    CHROMA_COLLECTION_PREFIX: str = "store_"

    # RAG Configuration
    RAG_TOP_K: int = 3
    RAG_SIMILARITY_THRESHOLD: float = 0.35  # cosine similarity lower bound (tuned for OpenAI text-embedding-3-small)

    # Security
    SECRET_KEY: str = Field(
        default="dev-secret-key-changeinproduction1234567890abcdefghijklmnopqrstuvwxyz",
        min_length=64,
        description="Secret key for JWT tokens (min 64 chars in production)",
    )
    ADMIN_USERNAME: str = Field(
        default="admin",
        description="Bootstrap admin username for token issuance",
    )
    ADMIN_PASSWORD: str = Field(
        default="admin123",
        description="Bootstrap admin password for token issuance (change in production!)",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 3

    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins",
    )

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Validate SECRET_KEY is strong enough for production."""
        env = info.data.get("ENVIRONMENT", "development")
        is_prod = env.lower() == "production"

        # Reject dangerous defaults in production
        if is_prod and v.startswith("dev-secret-key"):
            raise ValueError(
                "Default SECRET_KEY detected in production. "
                "Generate a strong key: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
            )

        if is_prod and len(v) < 64:
            raise ValueError("SECRET_KEY must be at least 64 characters in production")

        return v

    @field_validator("ADMIN_PASSWORD")
    @classmethod
    def validate_admin_password(cls, v: str, info) -> str:
        """Validate ADMIN_PASSWORD is strong enough for production."""
        env = info.data.get("ENVIRONMENT", "development")
        is_prod = env.lower() == "production"

        if is_prod:
            if v == "admin123" or v == "admin" or v == "password":
                raise ValueError(
                    "Weak ADMIN_PASSWORD detected in production. "
                    "Set a strong password (min 12 chars) via ADMIN_PASSWORD env var."
                )
            if len(v) < 12:
                raise ValueError("ADMIN_PASSWORD must be at least 12 characters in production")

        return v

    @field_validator("META_APP_ID")
    @classmethod
    def validate_meta_app_id(cls, v: str, info) -> str:
        """META_APP_ID must not be empty in production."""
        env = info.data.get("ENVIRONMENT", "development")
        if env.lower() == "production" and not v:
            raise ValueError(
                "META_APP_ID must be set in production. "
                "Provide your Meta App ID via the META_APP_ID env var."
            )
        return v

    @field_validator("META_APP_SECRET")
    @classmethod
    def validate_meta_app_secret(cls, v: str, info) -> str:
        """META_APP_SECRET must not be empty in production."""
        env = info.data.get("ENVIRONMENT", "development")
        if env.lower() == "production" and not v:
            raise ValueError(
                "META_APP_SECRET must be set in production. "
                "Provide your Meta App Secret via the META_APP_SECRET env var."
            )
        return v

    @field_validator("GROQ_API_KEY")
    @classmethod
    def validate_groq_api_key(cls, v: str, info) -> str:
        """GROQ_API_KEY must not be empty when LLM_PROVIDER is 'groq'."""
        provider = info.data.get("LLM_PROVIDER", "openai")
        if provider.lower() == "groq" and not v:
            raise ValueError(
                "GROQ_API_KEY must be set when LLM_PROVIDER='groq'. "
                "Get a key at https://console.groq.com and set GROQ_API_KEY env var."
            )
        return v

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: str, info) -> str:
        """Validate CORS_ORIGINS doesn't include localhost in production."""
        env = info.data.get("ENVIRONMENT", "development")
        is_prod = env.lower() == "production"

        if is_prod and ("localhost" in v.lower() or "127.0.0.1" in v):
            raise ValueError(
                "CORS_ORIGINS cannot contain localhost or 127.0.0.1 in production. "
                "Set the correct frontend domain via CORS_ORIGINS env var."
            )

        return v

    CORS_ALLOW_CREDENTIALS: bool = True

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Email / SMTP
    SMTP_HOST: str = Field(default="smtp.resend.com", description="SMTP server hostname")
    SMTP_PORT: int = Field(default=587, description="SMTP server port")
    SMTP_USER: str = Field(default="resend", description="SMTP username (use 'resend' for Resend.com)")
    SMTP_PASSWORD: str = Field(default="", description="SMTP password / Resend API key")
    SMTP_FROM_EMAIL: str = Field(default="onboarding@resend.dev", description="From email address")
    SMTP_FROM_NAME: str = Field(default="Ibra Agent", description="From name")
    SMTP_USE_TLS: bool = Field(default=True, description="Use TLS for SMTP")
    FRONTEND_URL: str = Field(default="http://localhost:3000", description="Frontend base URL for links in emails")

    # Monitoring
    SENTRY_DSN: str | None = None
    PROMETHEUS_ENABLED: bool = True

    # Push Notifications (OneSignal)
    ONESIGNAL_APP_ID: str = ""
    ONESIGNAL_API_KEY: str = ""

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 1000

    # Meta Webhook Settings
    WEBHOOK_TIMEOUT_SECONDS: int = 3
    MESSAGE_24H_WINDOW_HOURS: int = 24

    # Background Tasks
    CATALOG_SYNC_INTERVAL_HOURS: int = 6
    CONVERSATION_CLEANUP_DAYS: int = 30

    # Exchange Rates
    USD_TO_EGP: float = 50.0
    USD_TO_SAR: float = 3.75
    USD_TO_AED: float = 3.67
    USD_TO_IQD: float = 1309.0

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v) -> str:
        """Ensure CORS origins is a string."""
        if isinstance(v, list):
            return ",".join(v)
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL (for Alembic)."""
        return self.DATABASE_URL.replace("+asyncpg", "")

    @property
    def chroma_persist_directory(self) -> str:
        """Canonical Chroma path with backward-compatible env key support."""
        return self.CHROMA_PERSIST_DIRECTORY or self.CHROMADB_PERSIST_DIR


# Create global settings instance
settings = Settings()
