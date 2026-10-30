"""Kalshi transaction fees.

Per the Kalshi fee schedule of July 2026: fee = M * 0.07 * contracts * price * (1 - price),
rounded up to a centicent, where M is the series ``fee_multiplier``. Fees accumulate per
order, so an order costs what one equivalent fill would. FCM-cleared accounts align to
$0.01 instead: set ``fee_rounding_dollars`` to 0.01 to model that. Settlement carries no
fee for binary outcomes. All simulated orders are takers, so maker fees never apply.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

EPSILON = 1e-9


@dataclass(frozen=True)
class FeeSchedule:
    taker_coefficient: float = 0.07
    rounding: float = 0.0001

    def raw_fee(self, price: float, contracts: float, multiplier: float = 1.0) -> float:
        return self.taker_coefficient * multiplier * contracts * price * (1 - price)

    def order_fee(self, fills: list[tuple[float, float]], multiplier: float = 1.0) -> float:
        total = sum(self.raw_fee(price, size, multiplier) for price, size in fills)
        if total <= 0:
            return 0.0
        return math.ceil(total / self.rounding - EPSILON) * self.rounding

    def describe(self) -> dict:
        return asdict(self)
