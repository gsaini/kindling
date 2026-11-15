"""SQLAlchemy tables for API keys and usage. Migration 0002 creates them; change both together."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


class ApiKeyRow(Base):
    __tablename__ = "api_keys"
    __table_args__ = (Index("ix_api_keys_hash", "hash", unique=True),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16))  # the first characters of the key, to recognise it
    hash: Mapped[str] = mapped_column(String(64))  # SHA-256 hex of the whole key; the key itself is never stored
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String(16)))
    models: Mapped[list[str] | None] = mapped_column(ARRAY(String(200)), nullable=True)  # None: every model
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None: the server default
    daily_token_limit: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # None: unlimited
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")


class UsageEventRow(Base):
    __tablename__ = "usage_events"
    __table_args__ = (
        Index("ix_usage_events_owner_created_at", "owner", "created_at"),
        Index("ix_usage_events_key_id_created_at", "key_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String(100))
    # No foreign key: usage history outlives the keys it was made with.
    key_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    key_name: Mapped[str] = mapped_column(String(100))
    endpoint: Mapped[str] = mapped_column(String(100))
    model: Mapped[str] = mapped_column(String(200))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    wall_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
