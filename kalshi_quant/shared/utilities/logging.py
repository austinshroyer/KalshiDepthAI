"""Structured logging: JSON lines to a file, readable lines to stderr."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

from kalshi_quant.shared.config import PROJECT_ROOT


def configure_logging(name: str, level: str = "INFO", directory: str = "logs", console: bool = True) -> None:
    path = PROJECT_ROOT / directory
    path.mkdir(parents=True, exist_ok=True)
    pre = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[*pre, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    file_handler = logging.handlers.RotatingFileHandler(path / f"{name}.jsonl", maxBytes=50_000_000, backupCount=5)
    file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(), foreign_pre_chain=pre))
    handlers: list[logging.Handler] = [file_handler]
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False), foreign_pre_chain=pre))
        handlers.append(stream)
    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "websockets", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
