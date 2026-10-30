"""Late favorite: in the final minutes, buy a strong favorite while it is still discounted."""
from kalshi_quant.shared.strategy.base import NO, YES, Param, Risk, Signal, Strategy


class LateFavorite(Strategy):
    name = "fifteen_late_favorite"
    environments = ("fifteen_minute",)
    parameters = {
        "min_probability": Param(0.85, 0.70, 0.97, 0.01),
        "max_probability": Param(0.96, 0.80, 0.99, 0.01),
        "window_seconds": Param(180, 30, 420, 30, "enter within this long before close"),
    }
    risk = Risk(position_fraction=0.05, stop_loss=0.30)
    history_seconds = 300

    def should_enter(self, view):
        probability, to_close = view.probability, view.seconds_to_close
        if probability is None or to_close is None or not 0 < to_close <= self.p.window_seconds:
            return None
        favorite = max(probability, 1 - probability)
        if self.p.min_probability <= favorite <= self.p.max_probability:
            return Signal(YES if probability > 0.5 else NO)
        return None
