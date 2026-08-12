#!/usr/bin/env python3
"""
The watch list: names you are tracking, each with an entry zone, a stop and a
target ladder, plus the rule that turns a live price into an alert.

This module is pure -- it holds state and decides *what* should be said about a
price. Fetching quotes, remembering what was already sent and pushing it to
Discord all live in watchbot.py.

A watch is the same shape a scanner setup is (entry / stop / targets), so a
setup can be moved onto the list untouched and the alert wording lines up with
the status the scanner already gives it.

State lives in data/watchlist.json so the list survives a restart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

WATCHLIST_PATH = Path(__file__).with_name("data") / "watchlist.json"

# Half-width of the entry zone when a watch is added with a single entry price
# instead of an explicit lo/hi band.
DEFAULT_ZONE_PCT = 0.5
# How close to the zone counts as "approaching".
DEFAULT_NEAR_PCT = 1.5

# Alert kinds, most urgent first. The key stored per ticker is derived from
# these; an alert fires when a ticker's key changes.
STOPPED, TARGET, IN_ZONE, APPROACHING, QUIET = "stopped", "target", "in_zone", "approaching", "quiet"

# Left-bar colours on the Discord embed: actionable states are warm, a broken
# stop is red, and merely-nearby is grey so it reads as context, not a call.
COLORS = {
    IN_ZONE: 0xE8912D,
    TARGET: 0xE8912D,
    STOPPED: 0xE23D3D,
    APPROACHING: 0x4F545C,
}

# States worth repeating while they persist. A target print or a broken stop is
# a one-off event -- repeating those is just noise.
SUSTAINED = {IN_ZONE, APPROACHING}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fmt(v: float) -> str:
    """Trim a price for display: 600.265 not 600.2650000001, 4.58 not 4.5800."""
    if v is None:
        return "-"
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return s or "0"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class Watch:
    ticker: str
    entry: float
    stop: float
    targets: list[float] = field(default_factory=list)
    entry_lo: float = 0.0
    entry_hi: float = 0.0
    name: str = ""
    setup_type: str = "manual"
    note: str = ""
    source: str = "manual"
    added_at: str = field(default_factory=_now)

    def __post_init__(self):
        self.ticker = (self.ticker or "").strip().upper()
        self.entry = float(self.entry)
        self.stop = float(self.stop)
        self.targets = sorted(float(t) for t in self.targets)
        if not self.entry_lo or not self.entry_hi:
            band = self.entry * DEFAULT_ZONE_PCT / 100
            self.entry_lo, self.entry_hi = self.entry - band, self.entry + band
        if self.entry_lo > self.entry_hi:
            self.entry_lo, self.entry_hi = self.entry_hi, self.entry_lo
        self.validate()

    def validate(self) -> None:
        if not self.ticker:
            raise ValueError("a watch needs a ticker")
        if self.entry <= 0:
            raise ValueError(f"{self.ticker}: entry must be positive")
        if self.stop <= 0:
            raise ValueError(f"{self.ticker}: stop must be positive")
        if self.stop >= self.entry:
            raise ValueError(f"{self.ticker}: stop {fmt(self.stop)} must sit below "
                             f"entry {fmt(self.entry)} (these rules are long-only)")
        bad = [t for t in self.targets if t <= self.entry]
        if bad:
            raise ValueError(f"{self.ticker}: targets must sit above entry "
                             f"{fmt(self.entry)} (got {', '.join(fmt(t) for t in bad)})")

    @property
    def t1(self) -> float | None:
        return self.targets[0] if self.targets else None

    @classmethod
    def from_setup(cls, s: dict) -> "Watch":
        """Adopt a scanner setup verbatim -- same entry zone, stop and ladder."""
        return cls(
            ticker=s["ticker"], entry=float(s["entry"]), stop=float(s["stop"]),
            targets=[float(t) for t in s.get("targets", [])],
            entry_lo=float(s.get("entry_lo") or 0), entry_hi=float(s.get("entry_hi") or 0),
            name=s.get("name", ""), setup_type=s.get("setup_type", "manual"),
            note=s.get("note", ""), source="scanner",
        )


class WatchList:
    """A ticker-keyed set of watches, persisted as one JSON file."""

    def __init__(self, watches: list[Watch] | None = None, path: Path = WATCHLIST_PATH):
        self.path = Path(path)
        self.watches: dict[str, Watch] = {w.ticker: w for w in (watches or [])}

    # ---- persistence ----
    @classmethod
    def load(cls, path: Path = WATCHLIST_PATH) -> "WatchList":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text())
        fields = set(Watch.__dataclass_fields__)
        watches = []
        for w in raw.get("watches", []):
            try:
                watches.append(Watch(**{k: v for k, v in w.items() if k in fields}))
            except (TypeError, ValueError) as exc:
                # One malformed row must not cost the whole list.
                print(f"  ! skipping bad watch {w.get('ticker', '?')}: {exc}")
        return cls(watches, path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": _now(),
                   "watches": [asdict(w) for w in self.sorted()]}
        self.path.write_text(json.dumps(payload, indent=1))

    # ---- edits ----
    def sorted(self) -> list[Watch]:
        return sorted(self.watches.values(), key=lambda w: w.ticker)

    def add(self, w: Watch, overwrite: bool = True) -> Watch:
        if w.ticker in self.watches and not overwrite:
            raise FileExistsError(f"{w.ticker} is already on the list")
        self.watches[w.ticker] = w
        return w

    def remove(self, ticker: str) -> Watch:
        ticker = (ticker or "").strip().upper()
        if ticker not in self.watches:
            raise KeyError(f"{ticker} is not on the watch list")
        return self.watches.pop(ticker)

    def tickers(self) -> list[str]:
        return sorted(self.watches)

    def __len__(self) -> int:
        return len(self.watches)


# --------------------------------------------------------------------------
# Price -> alert
# --------------------------------------------------------------------------

@dataclass
class Alert:
    ticker: str
    kind: str          # one of STOPPED / TARGET / IN_ZONE / APPROACHING / QUIET
    key: str           # dedupe key -- an alert fires when this changes
    title: str
    body: str
    price: float
    color: int = 0
    targets_hit: int = 0

    @property
    def sustained(self) -> bool:
        return self.kind in SUSTAINED


def evaluate(w: Watch, price: float, near_pct: float = DEFAULT_NEAR_PCT,
             targets_hit: int = 0) -> Alert:
    """Read one live price against one watch.

    `targets_hit` is the high-water mark already reported for this name, so a
    ladder that has printed T1 does not re-announce it on every wobble back
    through the level -- only a *new* target speaks up.
    """
    n_t = len(w.targets)
    hit = sum(1 for t in w.targets if price >= t)

    if price < w.stop:
        return Alert(w.ticker, STOPPED, "stopped",
                     f"{w.ticker} — stopped out",
                     f"{w.ticker} broke the {fmt(w.stop)} stop · setup is dead",
                     price, COLORS[STOPPED], targets_hit)

    if hit > targets_hit:
        nxt = next((t for t in w.targets if t > price), None)
        tail = f" — next target {fmt(nxt)}" if nxt else " — all targets taken"
        return Alert(w.ticker, TARGET, f"target:{hit}",
                     f"{w.ticker} — target hit",
                     f"{w.ticker} TARGET {hit}/{n_t} HIT{tail}",
                     price, COLORS[TARGET], hit)

    if w.entry_lo <= price <= w.entry_hi:
        bits = [f"stop {fmt(w.stop)}"]
        if w.t1:
            bits.append(f"target {fmt(w.t1)}")
        return Alert(w.ticker, IN_ZONE, "in_zone",
                     f"{w.ticker} — in entry zone",
                     f"{w.ticker} is READY TO ENTER ({fmt(price)}) · {', '.join(bits)}",
                     price, COLORS[IN_ZONE], hit)

    # Outside the zone: how far, as a percentage of the nearer edge. A breakout
    # trigger sits above price and a pullback entry below it, so both sides
    # count as approaching -- the direction of travel is not assumed.
    edge = w.entry_lo if price < w.entry_lo else w.entry_hi
    away = abs(price - edge) / edge * 100 if edge else 999.0
    if away <= near_pct:
        return Alert(w.ticker, APPROACHING, "approaching",
                     f"{w.ticker} — approaching entry",
                     f"{w.ticker} is approaching its entry zone ({fmt(w.entry)})",
                     price, COLORS[APPROACHING], hit)

    side = "below" if price < w.entry_lo else "above"
    return Alert(w.ticker, QUIET, "quiet",
                 f"{w.ticker} — watching",
                 f"{away:.1f}% {side} the {fmt(w.entry_lo)}–{fmt(w.entry_hi)} zone",
                 price, 0, hit)
