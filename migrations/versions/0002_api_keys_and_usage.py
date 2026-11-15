"""API keys and usage events.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner", sa.String(100), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("scopes", postgresql.ARRAY(sa.String(16)), nullable=False),
        sa.Column("models", postgresql.ARRAY(sa.String(200)), nullable=True),
        sa.Column("rate_limit_per_minute", sa.Integer(), nullable=True),
        sa.Column("daily_token_limit", sa.BigInteger(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=False),
    )
    op.create_index("ix_api_keys_owner", "api_keys", ["owner"])
    op.create_index("ix_api_keys_hash", "api_keys", ["hash"], unique=True)

    op.create_table(
        "usage_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("owner", sa.String(100), nullable=False),
        sa.Column("key_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("key_name", sa.String(100), nullable=False),
        sa.Column("endpoint", sa.String(100), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("wall_seconds", sa.Float(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_usage_events_owner_created_at", "usage_events", ["owner", "created_at"])
    op.create_index("ix_usage_events_key_id_created_at", "usage_events", ["key_id", "created_at"])


def downgrade() -> None:
    op.drop_table("usage_events")
    op.drop_table("api_keys")
