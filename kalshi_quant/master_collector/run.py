"""Master collector: runs both pipelines concurrently in one event loop.

The only process that touches both environments. Each pipeline still has its own
database, writer, budget and websocket; a failure in one does not stop the other.
"""
from __future__ import annotations

import asyncio
import signal

from kalshi_quant.fifteen_minute.collector.pipeline import FifteenMinutePipeline
from kalshi_quant.general_markets.collector.pipeline import GeneralPipeline
from kalshi_quant.shared.collection.pipeline import Pipeline
from kalshi_quant.shared.config import Env, load_config
from kalshi_quant.shared.utilities.logging import configure_logging, get_logger

log = get_logger(__name__)


def build_pipeline(env: Env) -> Pipeline:
    return (GeneralPipeline if env is Env.GENERAL else FifteenMinutePipeline)(load_config(env))


async def run_pipelines(pipelines: list[Pipeline], duration_seconds: float | None = None) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: [p.request_stop() for p in pipelines])
    results = await asyncio.gather(*(p.run(duration_seconds) for p in pipelines), return_exceptions=True)
    for pipeline, result in zip(pipelines, results):
        if isinstance(result, Exception):
            log.error("pipeline_crashed", pipeline=pipeline.name, error=repr(result))


def collect(envs: list[Env], duration_seconds: float | None = None) -> None:
    name = "collect_all" if len(envs) > 1 else f"collect_{envs[0].value}"
    shared = load_config(envs[0]).shared
    configure_logging(name, shared.logging.level, shared.logging.directory)
    asyncio.run(run_pipelines([build_pipeline(e) for e in envs], duration_seconds))
