"""Autogenerate a migration for both environments from the current schema.

    python scripts/make_migration.py "add new column"
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alembic import command  # noqa: E402

from kalshi_quant.shared.config import Env  # noqa: E402
from kalshi_quant.shared.database.engine import alembic_config  # noqa: E402

message = sys.argv[1] if len(sys.argv) > 1 else "schema change"
for env in Env:
    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite:///{tmp}/m.sqlite"
        cfg = alembic_config(env, url)
        command.upgrade(cfg, "head")
        command.revision(cfg, message=message, autogenerate=True)
