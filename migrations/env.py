"""Alembic environment. Each platform environment has its own version directory."""
from alembic import context
from sqlalchemy import create_engine, pool

from kalshi_quant.shared.database.schema import build_metadata

config = context.config
environment = config.attributes["environment"]
target_metadata = build_metadata(environment)


def run_migrations_online() -> None:
    engine = create_engine(config.attributes["url"], poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=True, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
