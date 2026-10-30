from kalshi_quant.shared.strategy.base import YES, Param, Risk, Signal, Strategy


class Threshold(Strategy):
    name = "threshold"
    parameters = {"entry": Param(0.45, 0.30, 0.60, 0.05), "window": Param(300, 60, 600, 60)}
    risk = Risk(position_fraction=0.05, max_market_exposure=0.2)
    history_seconds = 900

    def should_enter(self, view):
        if view.probability is None or view.seconds_since_open is None or view.seconds_since_open > self.p.window:
            return None
        return Signal(YES) if view.probability < self.p.entry else None
