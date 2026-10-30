"""The brief's example, written directly as a strategy.

Buy YES when implied probability is below 0.40, provided the market has traded at least
10,000 contracts ($1 notional each), probability rose at least 5 points over the last
3 minutes, and the spread is under 0.03.
"""
from kalshi_quant.shared.strategy.base import YES, Param, Risk, Signal, Strategy


class MomentumDip(Strategy):
    name = "general_momentum_dip"
    environments = ("general",)
    parameters = {
        "entry_probability": Param(0.40, 0.20, 0.50, 0.01, "buy below this probability"),
        "min_volume": Param(10_000, 1_000, 50_000, 1_000, "minimum cumulative contracts"),
        "min_rise": Param(0.05, 0.01, 0.15, 0.01, "minimum probability rise over the lookback"),
        "lookback_seconds": Param(180, 60, 900, 60),
        "max_spread": Param(0.03, 0.01, 0.10, 0.01),
    }
    risk = Risk(position_fraction=0.05, stop_loss=0.15, take_profit=0.30, max_holding_seconds=6 * 3600)
    history_seconds = 3600

    def should_enter(self, view):
        probability, spread, volume = view.probability, view.spread, view.volume
        if probability is None or spread is None or volume is None:
            return None
        rise = view.probability_change(self.p.lookback_seconds)
        if (probability < self.p.entry_probability and volume >= self.p.min_volume
                and rise is not None and rise >= self.p.min_rise and spread < self.p.max_spread):
            return Signal(YES)
        return None
