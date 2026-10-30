"""Early momentum: if a 15-minute market moves decisively soon after opening, follow it."""
from kalshi_quant.shared.strategy.base import NO, YES, Param, Risk, Signal, Strategy


class EarlyMomentum(Strategy):
    name = "fifteen_early_momentum"
    environments = ("fifteen_minute",)
    parameters = {
        "buying_window_minutes": Param(5, 1, 14, 1, "only enter this long after open"),
        "move": Param(0.08, 0.02, 0.25, 0.01, "probability move required"),
        "lookback_seconds": Param(60, 30, 300, 30),
        "max_spread": Param(0.04, 0.01, 0.10, 0.01),
    }
    risk = Risk(position_fraction=0.05, stop_loss=0.20, take_profit=0.40, exit_before_close_seconds=30)
    history_seconds = 600

    def should_enter(self, view):
        since_open, spread = view.seconds_since_open, view.spread
        if since_open is None or spread is None or spread > self.p.max_spread:
            return None
        if not 0 <= since_open <= self.p.buying_window_minutes * 60:
            return None
        change = view.probability_change(self.p.lookback_seconds)
        if change is None or abs(change) < self.p.move:
            return None
        return Signal(YES if change > 0 else NO)
