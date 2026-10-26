"""Скрытие заданий с сохранением истории использования обучающих данных."""

import sqlalchemy as sa
from alembic import op

revision = "0002_hidden_jobs"
down_revision = "586e4da808e9"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "jobs", sa.Column("hidden", sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade():
    op.drop_column("jobs", "hidden")
