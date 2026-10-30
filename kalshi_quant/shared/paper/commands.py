"""Read and control paper sessions through the database, so commands work across processes."""
from __future__ import annotations

import json
import os
import signal
from typing import Any

from sqlalchemy import text, update

from kalshi_quant.shared.database.engine import Database
from kalshi_quant.shared.execution.portfolio import Portfolio
from kalshi_quant.shared.utilities.timeutil import now_ms

STALE_MS = 60_000


def sessions(db: Database) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.*, st.state FROM paper_sessions s LEFT JOIN paper_state st USING (session_id)
            ORDER BY s.session_id""")).mappings().all()
    result = []
    for row in rows:
        item = dict(row)
        state = json.loads(row["state"]) if row["state"] else None
        portfolio = Portfolio.from_state(state["portfolio"]) if state else Portfolio(row["starting_capital"])
        if item["status"] == "running" and now_ms() - (item["heartbeat_ms"] or 0) > STALE_MS:
            item["status"] = "crashed_or_stopped"
        item.update(cash=portfolio.cash, open_positions=len(portfolio.positions), realized_pnl=portfolio.realized_pnl,
                    fees=portfolio.fees_paid, portfolio=portfolio)
        result.append(item)
    return result


def positions(db: Database) -> list[dict[str, Any]]:
    out = []
    for session in sessions(db):
        for p in session["portfolio"].positions.values():
            out.append({"session": session["session_id"], "strategy": session["strategy_name"], "ticker": p.ticker,
                        "side": p.side, "quantity": p.quantity, "avg_price": p.avg_price, "mark": p.mark_price,
                        "unrealized": None if p.mark_price is None else p.quantity * (p.mark_price - p.avg_price) - p.entry_fees})
    return out


def trades(db: Database, session_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    sql = "SELECT * FROM paper_trades" + (" WHERE session_id = :s" if session_id else "") + " ORDER BY exit_ms DESC LIMIT :n"
    with db.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), {"s": session_id, "n": limit}).mappings()]


def performance(db: Database) -> list[dict[str, Any]]:
    out = []
    with db.connect() as conn:
        for session in sessions(db):
            rows = conn.execute(text("SELECT pnl FROM paper_trades WHERE session_id = :s"),
                                {"s": session["session_id"]}).scalars().all()
            wins = [p for p in rows if p > 0]
            portfolio = session["portfolio"]
            equity = portfolio.equity()
            out.append({"session": session["session_id"], "strategy": session["strategy_name"], "status": session["status"],
                        "starting_capital": session["starting_capital"], "equity": equity, "cash": portfolio.cash,
                        "realized_pnl": portfolio.realized_pnl, "unrealized_pnl": portfolio.unrealized_pnl(),
                        "return": equity / session["starting_capital"] - 1, "trades": len(rows),
                        "win_rate": len(wins) / len(rows) if rows else None, "fees": portfolio.fees_paid})
    return out


def request_stop(db: Database, session_id: int) -> str:
    t = db.t["paper_sessions"]
    with db.begin() as conn:
        pid = conn.execute(text("SELECT pid FROM paper_sessions WHERE session_id = :s"), {"s": session_id}).scalar()
        conn.execute(update(t).where(t.c.session_id == session_id, t.c.status == "running").values(status="stop_requested"))
    return f"stop requested for session {session_id} (pid {pid}); the engine exits within seconds"


def signal_session(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
