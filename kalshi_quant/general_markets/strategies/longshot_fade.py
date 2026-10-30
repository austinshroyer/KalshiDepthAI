"""Sell the long shot: buy NO on cheap YES contracts that have stopped rising."""
from kalshi_quant.shared.strategy.base import NO, Param, Risk, Signal, Strategy


class LongshotFade(Strategy):
    name = "general_longshot_fade"
    environments = ("general",)
    parameters = {
        "max_probability": Param(0.10, 0.03, 0.20, 0.01),
        "min_volume": Param(1_000, 100, 20_000, 100),
        "max_rise": Param(0.0, -0.02, 0.02, 0.01),
        "min_hours_to_close": Param(24, 1, 168, 1),
    }
    risk = Risk(position_fraction=0.02, max_market_exposure=0.04, stop_loss=0.25)
    history_seconds = 7200

    def should_enter(self, view):
        probability, to_close = view.probability, view.seconds_to_close
        if probability is None or to_close is None or view.volume is None:
            return None
        change = view.probability_change(3600)
        if (probability <= self.p.max_probability and view.volume >= self.p.min_volume
                and to_close >= self.p.min_hours_to_close * 3600 and change is not None and change <= self.p.max_rise):
            return Signal(NO)
        return None
