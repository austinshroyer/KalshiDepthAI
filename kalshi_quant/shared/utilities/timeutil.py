"""UTC time helpers. All stored timestamps are integer UTC milliseconds."""
from __future__ import annotations

import time
from datetime import datetime, timezone


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_ts_ms(value: "str | int | float | None") -> int | None:
    """Parse an RFC3339 string or Unix seconds into UTC milliseconds."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value * 1000)
    text = value.replace("Z", "+00:00")
    if "." in text:  # Kalshi sends up to 9 fractional digits; fromisoformat needs 6 or fewer
        head, tail = text.split(".", 1)
        sign = "+" if "+" in tail else "-"
        frac, _, offset = tail.partition(sign)
        text = f"{head}.{frac[:6].ljust(6, '0')}{sign}{offset}"
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def ms_to_iso(ms: "int | None") -> str | None:
    return None if ms is None else ms_to_dt(ms).isoformat().replace("+00:00", "Z")


def parse_date_ms(text: "str | None") -> int | None:
    """Parse YYYY-MM-DD or an ISO timestamp, treating naive values as UTC."""
    if not text:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
