from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Postgres / SQLAlchemy
    database_url: str = "postgresql+psycopg://teammember:teammember_password_change_me@postgres:5432/teammember"

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_distance: str = "cosine"
    qdrant_vector_size: int = 1536
    qdrant_knowledge_collection: str = "tm_knowledge"
    qdrant_memory_collection: str = "tm_user_memory"

    # DashScope (Qwen)
    dashscope_api_key: str = ""
    dashscope_vision_model: str = "qwen-vl-plus"
    dashscope_embedding_model: str = "text-embedding-v2"

    # Ark (Volcengine)
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_chat_model: str = "deepseek-v3-2-251201"
    ark_temperature: float = 0.2

    # Auth
    auth_enabled: bool = True
    jwt_secret: str = "change_me_in_production"
    jwt_exp_minutes: int = 60 * 24 * 7

    # RAG
    rag_policy: str = "auto"  # auto | force_on | force_off
    rag_top_k: int = 10
    rag_max_context: int = 12
    rag_rerank_min_docs: int = 8000
    rag_teaching_score_boost: float = 0.05
    rag_teaching_candidates: int = 40

    # Topic guard (optional scope control)
    topic_guard_enabled: bool = False
    topic_allowed_topics: str = "21V Microsoft 365（世纪互联）及相关的 Power Platform / Dynamics 365"

    # Web search (Bing)
    bing_search_api_key: str = ""
    bing_search_endpoint: str = "https://api.bing.microsoft.com/v7.0/search"
    bing_search_market: str = "zh-CN"
    bing_search_safe_search: str = "Moderate"  # Off | Moderate | Strict
    web_search_enabled: bool = False
    web_search_top_k: int = 5
    web_search_max_queries: int = 2

    # Sub-agent decomposition
    agent_decompose_policy: str = "auto"  # auto | force_on | force_off
    agent_decompose_bias: int = 30  # 0..100 (higher => decompose more often)
    agent_max_subtasks: int = 5

    # Memory
    memory_enabled: bool = True
    memory_top_k: int = 5

    # Proactive nudge (optional)
    proactive_enabled: bool = False
    proactive_min_user_msgs: int = 500
    proactive_weekday_only: bool = True
    proactive_work_start: str = "09:00"  # HH:MM
    proactive_work_end: str = "18:00"  # HH:MM
    proactive_timezone: str = "Asia/Shanghai"

    # Semantic thread state (open loops / topic entropy)
    thread_state_enabled: bool = True
    thread_state_window_msgs: int = 60
    thread_state_cooldown_seconds: int = 90

    # Decision profile (how the user decides/verifies/assumes)
    decision_profile_enabled: bool = True
    decision_profile_refresh_hours: int = 24 * 5

    # Privacy / disclosure controls
    # Whether the assistant is allowed to disclose internal Persona/DecisionProfile summaries to users.
    # Recommended: keep false in production.
    persona_disclosure_enabled: bool = False

    # AI trace (routing/tools/sub-agents) for debugging & transparency
    ai_trace_enabled: bool = True
    ai_trace_retention_days: int = 30


settings = Settings()
