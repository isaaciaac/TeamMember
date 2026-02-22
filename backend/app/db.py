from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential

from .settings import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # Phone-first auth (supports email for legacy / optional use).
    phone: Mapped[str | None] = mapped_column(String(30), unique=True, nullable=True, index=True)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    password_hash: Mapped[str] = mapped_column(String(300), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    allow_any_topic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    profile: Mapped[str] = mapped_column(Text, default="", nullable=False)
    last_proactive_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    proactive_pending_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    proactive_last_ask_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    proactive_pending_thread_id: Mapped[str] = mapped_column(String(36), default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class Thread(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_user_id: Mapped[str] = mapped_column(String(36), default="", nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    canvas_md: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(String(36), ForeignKey("threads.id"), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(36), default="", nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class AiTraceRun(Base):
    """
    Per-request trace for routing/tools/sub-agents.

    This is for transparency/debugging (not security). Retention is enforced by a daily cleanup job.
    """

    __tablename__ = "ai_trace_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(String(36), ForeignKey("threads.id"), index=True, nullable=False)
    actor_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)

    user_message_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    assistant_message_id: Mapped[str] = mapped_column(String(36), index=True, default="", nullable=False)

    router_json: Mapped[str] = mapped_column(Text, default="", nullable=False)
    web_search_json: Mapped[str] = mapped_column(Text, default="", nullable=False)
    decompose_json: Mapped[str] = mapped_column(Text, default="", nullable=False)
    subagent_json: Mapped[str] = mapped_column(Text, default="", nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class ThreadShare(Base):
    __tablename__ = "thread_shares"
    __table_args__ = (UniqueConstraint("thread_id", "shared_with_user_id", name="uq_thread_share"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id: Mapped[str] = mapped_column(String(36), ForeignKey("threads.id"), index=True, nullable=False)
    shared_with_user_id: Mapped[str] = mapped_column(String(36), default="", nullable=False, index=True)
    permission: Mapped[str] = mapped_column(String(10), default="read", nullable=False)  # read | write
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class ThreadState(Base):
    """
    Thread-level semantic state, maintained asynchronously by LLM analyzers.
    Used for lightweight routing decisions (open loops / topic entropy).
    """

    __tablename__ = "thread_state"

    thread_id: Mapped[str] = mapped_column(String(36), ForeignKey("threads.id"), primary_key=True)
    open_issues_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    entropy_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    last_analyzed_message_id: Mapped[str] = mapped_column(String(36), default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class AppConfig(Base):
    __tablename__ = "app_config"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class AuditLog(Base):
    """
    Append-only audit log for admin actions.

    NOTE:
    This is tamper-resistant for in-app admins (no API to modify; DB trigger blocks UPDATE/DELETE).
    It is not tamper-proof against a database superuser with direct access to Postgres.
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    actor_label: Mapped[str] = mapped_column(String(200), default="", nullable=False)

    action: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(60), default="", index=True, nullable=False)
    entity_id: Mapped[str] = mapped_column(String(120), default="", index=True, nullable=False)

    request_method: Mapped[str] = mapped_column(String(10), default="", nullable=False)
    request_path: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    request_ip: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(String(500), default="", nullable=False)

    before_json: Mapped[str] = mapped_column(Text, default="", nullable=False)
    after_json: Mapped[str] = mapped_column(Text, default="", nullable=False)

    prev_hash: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    event_hash: Mapped[str] = mapped_column(String(80), default="", nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True, nullable=False)


class UserPersona(Base):
    __tablename__ = "user_persona"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class UserDecisionProfile(Base):
    """
    A compact, human-readable summary of how the user tends to decide/verify/assume.
    Refreshed asynchronously and injected into prompts.
    """

    __tablename__ = "user_decision_profile"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class UserMemory(Base):
    __tablename__ = "user_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), default="paste", nullable=False)  # sql | odata | paste
    name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class KnowledgeReview(Base):
    """
    User-submitted "teach-back" knowledge points that require admin approval before being written to Qdrant.
    """

    __tablename__ = "knowledge_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    submitter_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(36), ForeignKey("threads.id"), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True, nullable=False)  # pending|approved|rejected
    title: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    teaching_note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    points_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    admin_comment: Mapped[str] = mapped_column(Text, default="", nullable=False)
    reviewed_by_user_id: Mapped[str] = mapped_column(String(36), default="", nullable=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    applied_count: Mapped[int] = mapped_column(default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class UserProactiveEvent(Base):
    __tablename__ = "user_proactive_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@retry(stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, min=1, max=5))
def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_schema()


def _migrate_schema() -> None:
    from sqlalchemy import text

    stmts = [
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS phone varchar(30) NULL",
        "ALTER TABLE IF EXISTS users ALTER COLUMN email DROP NOT NULL",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS name varchar(120) NOT NULL DEFAULT ''",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS allow_any_topic boolean NOT NULL DEFAULT false",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS profile text NOT NULL DEFAULT ''",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS last_proactive_at timestamptz NULL",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS proactive_pending_at timestamptz NULL",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS proactive_last_ask_at timestamptz NULL",
        "ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS proactive_pending_thread_id varchar(36) NOT NULL DEFAULT ''",
        "ALTER TABLE IF EXISTS threads ADD COLUMN IF NOT EXISTS owner_user_id varchar(36) NOT NULL DEFAULT ''",
        "ALTER TABLE IF EXISTS chat_messages ADD COLUMN IF NOT EXISTS user_id varchar(36) NOT NULL DEFAULT ''",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_phone_unique ON users(phone) WHERE phone IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS ix_threads_owner_user_id ON threads(owner_user_id)",
        "CREATE INDEX IF NOT EXISTS ix_chat_messages_user_id ON chat_messages(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_user_memory_user_id ON user_memory(user_id)",
        "CREATE INDEX IF NOT EXISTS ix_user_proactive_events_user_id ON user_proactive_events(user_id)",
        # Audit logs are append-only (defense in depth; API never exposes update/delete anyway).
        "CREATE OR REPLACE FUNCTION audit_logs_no_update_delete() RETURNS trigger AS $$ BEGIN RAISE EXCEPTION 'audit_logs is append-only'; END $$ LANGUAGE plpgsql",
        "DROP TRIGGER IF EXISTS audit_logs_block_update ON audit_logs",
        "CREATE TRIGGER audit_logs_block_update BEFORE UPDATE ON audit_logs FOR EACH ROW EXECUTE FUNCTION audit_logs_no_update_delete()",
        "DROP TRIGGER IF EXISTS audit_logs_block_delete ON audit_logs",
        "CREATE TRIGGER audit_logs_block_delete BEFORE DELETE ON audit_logs FOR EACH ROW EXECUTE FUNCTION audit_logs_no_update_delete()",
        # If there is no admin user yet, promote the earliest user as admin (single-tenant self-hosted default).
        "UPDATE users SET is_admin = true WHERE id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin = true)",
    ]
    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
