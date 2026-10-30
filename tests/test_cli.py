import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import synthetic
import yaml
from typer.testing import CliRunner

from kalshi_quant import cli
from kalshi_quant.shared.config import PROJECT_ROOT, load_config
from kalshi_quant.shared.database.engine import Database

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the CLI at temporary databases for this test only."""
    original = load_config

    def patched(env, config_dir=None):
        config = original(env, config_dir)
        config.database_url = f"sqlite:///{tmp_path}/{config.env.value}.sqlite"
        return config

    monkeypatch.setattr(cli, "load_config", patched)
    db = Database.for_config(patched("general"))
    synthetic.seed(db, days=20)
    db.dispose()
    f = Database.for_config(patched("15m"))
    synthetic.seed(f, days=5, duration_ms=900_000, step_ms=10_000, series="KXBTC15M")
    f.dispose()
    return tmp_path


def run(*args):
    result = runner.invoke(cli.app, list(args))
    return result


def test_backtest_both_environments_and_replay(workspace):
    general = run("backtest", "general", "--strategy", "example.py", "--json", "--set", "min_volume=100", "--set", "min_rise=0.02")
    assert general.exit_code == 0, general.output
    out = json.loads(general.output)
    assert run("replay", "general", out["run_id"]).exit_code == 0
    fifteen = run("backtest", "15m", "--strategy", "late_favorite.py")
    assert fifteen.exit_code == 0, fifteen.output
    assert "Execution and data quality" in fifteen.output


def test_strategy_directories_do_not_cross(workspace):
    assert run("backtest", "general", "--strategy", "late_favorite.py").exit_code != 0


def test_database_commands(workspace):
    for args in (("database", "stats"), ("database", "growth"), ("database", "query", "general", "--category", "Crypto", "--limit", "3"),
                 ("database", "backup", "--env", "general"), ("paper", "positions"), ("paper", "performance"), ("paper", "capital"),
                 ("paper", "sessions", "15m"), ("paper", "trades", "general"), ("runs", "general")):
        result = run(*args)
        assert result.exit_code == 0, (args, result.output, result.exception)


def test_validate_passes_on_clean_data(workspace):
    result = run("database", "validate")
    assert result.exit_code == 0, result.output


def test_empty_selection_is_a_clear_error(workspace):
    result = run("backtest", "general", "--strategy", "example.py", "--category", "Nothing")
    assert result.exit_code != 0


def test_optimize_runs_from_cli(workspace):
    ranges = workspace / "space.yaml"
    ranges.write_text(yaml.safe_dump({"entry_probability": {"values": [0.4, 0.5]}}))
    result = run("optimize", "general", "--strategy", "example.py", "--space", str(ranges), "--min-trades", "1", "--folds", "1")
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Verdict" in result.output


@pytest.mark.parametrize("script", ["collect_all.py", "collect_general_markets.py", "collect_15m_markets.py", "paper_trade.py",
                                    "paper_trade_general.py", "paper_trade_15m.py", "backtest.py", "backtest_general.py",
                                    "backtest_15m.py", "optimize_general.py", "optimize_15m.py"])
def test_entry_scripts_start(script):
    completed = subprocess.run([sys.executable, script, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


def test_commands_as_written_in_the_brief(workspace):
    result = run("backtest", "general", "--strategy", "strategies/example.py")
    assert result.exit_code == 0, (result.output, result.exception)
    for args in (["paper", "general", "--help"], ["paper", "15m", "--help"], ["collect", "--help"], ["optimize", "--help"]):
        assert run(*args).exit_code == 0
