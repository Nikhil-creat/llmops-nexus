import os


def _e(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


class Settings:
    """All runtime configuration, read from environment variables."""

    def __init__(self) -> None:
        self.upstream_base = _e("UPSTREAM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.upstream_key = _e("OPENAI_API_KEY", "")
        self.mock = _e("MOCK_UPSTREAM", "0") == "1"
        self.mock_delay = float(_e("MOCK_DELAY_S", "0.9"))

        emb = _e("EMBEDDER", "auto")
        if emb == "auto":
            emb = "openai" if (self.upstream_key and not self.mock) else "local"
        self.embedder = emb
        self.embed_model = _e("EMBED_MODEL", "text-embedding-3-small")
        self.embed_dim = int(_e("EMBED_DIM", "1536" if emb == "openai" else "384"))

        self.store = _e("VECTOR_STORE", "memory")
        self.redis_url = _e("REDIS_URL", "redis://localhost:6379")
        self.qdrant_url = _e("QDRANT_URL", "http://localhost:6333")

        self.threshold = float(_e("SIM_THRESHOLD", "0.92" if emb == "openai" else "0.80"))
        self.ttl = int(_e("CACHE_TTL_S", "3600"))
        self.max_temperature = float(_e("MAX_TEMPERATURE", "1.0"))
        self.context_turns = int(_e("CONTEXT_TURNS", "0"))
        self.l1_size = int(_e("L1_SIZE", "2048"))
        self.admin_token = _e("ADMIN_TOKEN", "")
