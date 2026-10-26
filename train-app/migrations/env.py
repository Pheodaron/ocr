from alembic import context
from sqlalchemy import engine_from_config, pool

from train_app import models  # noqa: F401
from train_app.db import Base

config = context.config
engine = engine_from_config(
    config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool
)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
