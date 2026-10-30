"""Loads a strategy file plus its optional YAML of the same name."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import yaml

from kalshi_quant.shared.config import EnvConfig
from kalshi_quant.shared.strategy.base import Strategy


def resolve_path(config: EnvConfig, name: str) -> Path:
    """A bare name resolves inside this environment's own strategy directory."""
    direct = Path(name)
    if direct.is_file():
        return direct.resolve()
    for inside in (config.strategy_dir() / name, config.strategy_dir() / direct.name):
        if inside.is_file():  # the second form accepts strategies/example.py
            return inside.resolve()
    raise FileNotFoundError(f"strategy {name!r} not found in {config.strategy_dir()}")


def load_class(path: Path, class_name: str | None = None) -> type[Strategy]:
    module_name = f"strategy_{path.stem}_{abs(hash(str(path))):x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses and pickling need the module registered
    spec.loader.exec_module(module)
    found = [c for _, c in inspect.getmembers(module, inspect.isclass)
             if issubclass(c, Strategy) and c is not Strategy and c.__module__ == module_name]
    if class_name:
        found = [c for c in found if c.__name__ == class_name]
    if len(found) != 1:
        raise ValueError(f"{path} must define exactly one Strategy subclass, found {len(found)}")
    return found[0]


def read_settings(path: Path) -> dict[str, Any]:
    yaml_path = path.with_suffix(".yaml")
    return (yaml.safe_load(yaml_path.read_text()) or {}) if yaml_path.exists() else {}


def load_strategy(config: EnvConfig, name: str, params: dict[str, Any] | None = None) -> tuple[Strategy, Path]:
    path = resolve_path(config, name)
    cls = load_class(path)
    if config.env.value not in cls.environments:
        raise ValueError(f"{cls.__name__} does not support the {config.env.value} environment")
    settings = read_settings(path)
    merged = {**(settings.get("parameters") or {}), **(params or {})}
    return cls(merged, settings.get("risk"), settings.get("universe")), path
