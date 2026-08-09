#!/usr/bin/env python3
"""
Setup scanner across the Nasdaq-100 and S&P 500 constituents.

Three scan rules run over every name and each hit is tagged with the rule that
fired:

  ote       a fresh impulse leg pulling back into the 0.62-0.79 retracement
  ma        an uptrend pulling back to a rising 21 EMA
  breakout  a volatility squeeze coiling under recent highs

Each hit carries a derived entry, stop and target ladder, and is classified the
same way a hand-kept tracker would read it -- invalidated / target hit / don't
chase / wait for pullback / in the zone.

    python3 scanner.py                     # print a ranked table
    python3 scanner.py --html setups.html  # write a shareable static page
    python3 scanner.py --types ote,ma --limit 40

The same entry points back the /scanner page in app.py.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from signals import confluence

CACHE_PATH = Path(__file__).with_name(".universe_cache.json")
# Full S&P 500 membership, committed so the scanner never silently degrades to a
# handful of names when the network or a parser dependency is unavailable.
BUNDLED_SP500 = Path(__file__).with_name("data") / "sp500.csv"

SETUP_LABELS = {"ote": "OTE pullback", "ma": "MA pullback", "breakout": "Squeeze breakout"}

# Ranked most-actionable first; drives ordering and the status colour.
STATUS_RANK = {
    "IN ENTRY ZONE": 0,
    "COILING": 1,             # breakout still under its trigger -- the normal pre-break state
    "WAIT FOR PULLBACK": 2,
    "TARGET HIT": 3,
    "BELOW ZONE": 4,          # pullback went deeper than the zone but is still above the stop
    "DON'T CHASE": 5,
    "INVALIDATED": 6,
}
MAX_RANK = max(STATUS_RANK.values())


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

INDEXES = ("sp100", "sp500", "ndx", "both")
SP100_SIZE = 100


def _norm_sym(s: str | None) -> str:
    """Yahoo uses a dash for share classes; sources variously use . or /."""
    return (s or "").strip().upper().replace(".", "-").replace("/", "-")


def _load_bundled_rows() -> list[dict]:
    if not BUNDLED_SP500.exists():
        return []
    with BUNDLED_SP500.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("symbol")]


def _load_bundled() -> dict[str, str]:
    """The committed S&P 500 list. Always available, no network, no lxml."""
    return {r["symbol"]: r["name"] for r in _load_bundled_rows()}


def _fetch_caps() -> dict[str, float]:
    """Market caps for the S&P 500, used only to rank the top 100."""
    url = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials"
           "/main/data/constituents-financials.csv")
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        sym = _norm_sym(row.get("Symbol"))
        raw = (row.get("Market Cap") or "").strip()
        if not sym or not raw:
            continue
        try:
            cap = float(raw)
        except ValueError:
            continue
        if cap > 0:
            out[sym] = cap
    return out


def _caps_from_yahoo(tickers: list[str]) -> dict[str, float]:
    """Market caps for names the CSV has no figure for. One request each, so it
    runs only for the gaps -- but without it a blank row silently drops a real
    company (Berkshire among them) out of a 'largest 100' ranking."""
    out: dict[str, float] = {}
    for t in tickers:
        try:
            cap = getattr(yf.Ticker(t).fast_info, "market_cap", None)
            if cap:
                out[t] = float(cap)
        except Exception:  # noqa: BLE001
            continue
    return out


def _top_by_cap(names: dict[str, str], n: int = SP100_SIZE) -> dict[str, str]:
    """The n largest by market cap. Live caps if reachable, else the snapshot
    committed alongside the constituent list."""
    caps: dict[str, float] = {}
    try:
        caps = _fetch_caps()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! live market caps unavailable ({exc}); using the bundled snapshot",
              file=sys.stderr)
    if not caps:
        caps = {r["symbol"]: float(r.get("market_cap") or 0) for r in _load_bundled_rows()}

    # top up from the bundled snapshot, then from Yahoo for whatever is still blank
    bundled = {r["symbol"]: float(r.get("market_cap") or 0) for r in _load_bundled_rows()}
    for t in names:
        if not caps.get(t) and bundled.get(t):
            caps[t] = bundled[t]
    gaps = sorted(t for t in names if not caps.get(t))
    if gaps:
        print(f"  filling {len(gaps)} missing market caps from Yahoo…", file=sys.stderr)
        caps.update(_caps_from_yahoo(gaps))

    ranked = sorted(names, key=lambda t: -caps.get(t, 0.0))
    still = [t for t in names if not caps.get(t)]
    if still:
        # A name with no cap sorts last and never makes the cut. Say so rather
        # than let a real company vanish from a "largest 100" ranking unnoticed.
        print(f"  ! {len(still)} constituents still have no market cap, so they cannot be "
              f"ranked and are excluded from the top {n}: {', '.join(still[:10])}"
              f"{'…' if len(still) > 10 else ''}", file=sys.stderr)
    return {t: names[t] for t in ranked[:n]}


def _fetch_sp500_csv() -> dict[str, str]:
    """Current S&P 500 from a maintained CSV dataset. Plain stdlib parsing, so
    unlike the Wikipedia tables this path does not need lxml installed."""
    url = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
           "/main/data/constituents.csv")
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        sym = _norm_sym(row.get("Symbol"))
        if sym:
            out[sym] = (row.get("Security") or sym).strip()
    if len(out) < 400:
        raise ValueError(f"only {len(out)} rows -- refusing a truncated index")
    return out


def _read_wiki_table(url: str, symbol_col: str, name_col: str) -> dict[str, str]:
    """Wikipedia table -- needs lxml. The only source of Nasdaq-100 membership
    reachable without a paid data feed."""
    tables = pd.read_html(url)
    for t in tables:
        cols = {str(c).strip() for c in t.columns}
        if symbol_col in cols and name_col in cols:
            out = {}
            for sym, name in zip(t[symbol_col], t[name_col]):
                sym = _norm_sym(sym)  # BRK.B / BRK/B -> BRK-B
                if sym and sym != "NAN":
                    out[sym] = str(name).strip()
            if out:
                return out
    raise ValueError(f"no table with {symbol_col}/{name_col} at {url}")


def _fetch_ndx() -> dict[str, str]:
    ndx = _read_wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "Company")
    if len(ndx) < 90:
        raise ValueError(f"only {len(ndx)} rows -- refusing a truncated Nasdaq-100")
    return ndx


def _read_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}
    # migrate the old flat {ticker: name} shape, which carried no membership
    if raw and not any(k in raw for k in ("sp500", "ndx")):
        return {"sp500": raw}
    return raw


def _write_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=0, sort_keys=True))
    except OSError:
        pass


def load_universe(refresh: bool = False, which: str = "ndx") -> dict[str, str]:
    """Ticker -> company name for the requested index.

    `which` is "sp100" (default -- the 100 largest S&P 500 names by market cap),
    "sp500", "ndx", or "both". Membership is cached per index so switching
    between them does not refetch.

    The S&P 500 has a committed fallback in data/sp500.csv, so it always
    resolves. The Nasdaq-100 does NOT: the only reachable source is the
    Wikipedia table, and no accurate offline list ships with this repo. That is
    deliberate -- an approximation reconstructed from market caps came out ~66%
    correct with junk tickers in it, and silently scanning the wrong 100 names
    is worse than failing loudly.
    """
    which = which if which in INDEXES else "sp100"
    cache = {} if refresh else _read_cache()
    # sp100 is a ranking of sp500, not a separate membership list
    want = ["sp500", "ndx"] if which == "both" else \
        ["sp500"] if which == "sp100" else [which]

    for key in want:
        if cache.get(key):
            continue
        try:
            cache[key] = _fetch_sp500_csv() if key == "sp500" else _fetch_ndx()
            print(f"  {key}: {len(cache[key])} constituents", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {key} unavailable: {exc}", file=sys.stderr)
            if key == "sp500":
                bundled = _load_bundled()
                if bundled:
                    print(f"  ! using the bundled S&P 500 ({len(bundled)} names)", file=sys.stderr)
                    cache[key] = bundled

    if cache:
        _write_cache(cache)

    out: dict[str, str] = {}
    for key in want:
        out.update(cache.get(key) or {})
    if which == "sp100" and out:
        out = _top_by_cap(out)
    if not out:
        raise RuntimeError(
            f"could not resolve the {which} universe. The Nasdaq-100 needs the Wikipedia "
            f"table (install lxml and allow en.wikipedia.org), or run with "
            f"--universe sp500, which ships with the repo.")
    return out


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def pivots(series: pd.Series, k: int, kind: str) -> list[int]:
    """Indices where the value is the extreme of its +/-k neighbourhood."""
    vals = series.to_numpy()
    out = []
    for i in range(k, len(vals) - k):
        w = vals[i - k: i + k + 1]
        if kind == "high" and vals[i] == w.max() and (w.argmax() == k):
            out.append(i)
        elif kind == "low" and vals[i] == w.min() and (w.argmin() == k):
            out.append(i)
    return out


# --------------------------------------------------------------------------
# Setup model
# --------------------------------------------------------------------------

@dataclass
class Setup:
    ticker: str
    name: str
    setup_type: str
    price: float
    entry: float
    entry_lo: float
    entry_hi: float
    stop: float
    targets: list[float]
    vol_ratio: float
    vol_trend: str
    vol_bars: list[float]
    context: str = ""
    status: str = ""
    headline: str = ""
    note: str = ""
    # "from here" economics -- what you get taking the trade at the current
    # price, which is what makes an extended name obviously not worth chasing
    risk_pct: float = 0.0
    reward_pct: float = 0.0
    rr: float = 0.0
    next_target: float = 0.0
    next_target_label: str = "T1"
    # setup quality measured at the planned entry; drives ranking and filtering
    entry_risk_pct: float = 0.0
    entry_reward_pct: float = 0.0
    entry_rr: float = 0.0
    pct_from_entry: float = 0.0
    targets_hit: int = 0
    score: float = 0.0
    confluence: dict = field(default_factory=dict)
    bias: float = 0.0
    extra: dict = field(default_factory=dict)


def _round(v: float) -> float:
    return round(float(v), 2)


def classify(s: Setup) -> None:
    """Fill the derived numbers, status and prose on a setup, in place."""
    price, entry, stop = s.price, s.entry, s.stop
    t1 = s.targets[0] if s.targets else None
    s.pct_from_entry = (price - entry) / entry * 100 if entry else 0.0
    s.targets_hit = sum(1 for t in s.targets if price >= t)

    # setup quality, judged at the planned entry
    s.entry_risk_pct = (entry - stop) / entry * 100 if entry else 0.0
    s.entry_reward_pct = (t1 - entry) / entry * 100 if t1 else 0.0
    s.entry_rr = (s.entry_reward_pct / s.entry_risk_pct) if s.entry_risk_pct > 0 and t1 else 0.0

    # economics of entering at the current price, against the next unhit target
    nxt = next((t for t in s.targets if t > price), s.targets[-1] if s.targets else None)
    s.next_target = _round(nxt) if nxt else 0.0
    s.next_target_label = f"T{s.targets.index(nxt) + 1}" if nxt in s.targets else "T1"
    s.risk_pct = (price - stop) / price * 100 if price else 0.0
    s.reward_pct = (nxt - price) / price * 100 if nxt else 0.0
    s.rr = (s.reward_pct / s.risk_pct) if s.risk_pct > 0 and s.reward_pct > 0 else 0.0

    n_t = len(s.targets)
    if price < stop:
        s.status = "INVALIDATED"
        s.headline = "INVALIDATED"
        s.note = f"below the {_round(stop)} stop · setup is dead"
    elif s.targets_hit:
        s.status = "TARGET HIT"
        s.headline = f"TARGET {s.targets_hit}/{n_t} HIT"
        nxt = next((t for t in s.targets if t > price), None)
        took = s.targets[s.targets_hit - 1]
        tail = f" · next target {_round(nxt)}" if nxt else " · all targets taken"
        s.note = f"took out {_round(took)}{tail} · pullback to {_round(entry)} {s.context}".strip()
    elif s.entry_lo <= price <= s.entry_hi:
        s.status = "IN ENTRY ZONE"
        s.headline = "IN ENTRY ZONE"
        s.note = (f"in the {_round(s.entry_lo)}–{_round(s.entry_hi)} zone now · "
                  f"risk to {_round(stop)} · {s.context}").strip(" ·")
    elif price < s.entry_lo:
        # A breakout entry sits ABOVE price by construction, so being under it is
        # the setup working as intended -- not a missed pullback.
        if s.setup_type == "breakout":
            s.status = "COILING"
            s.headline = "COILING"
            s.note = (f"{abs(s.pct_from_entry):.1f}% under the {_round(entry)} trigger · "
                      f"squeeze intact · break takes it to {_round(t1)}" if t1 else "squeeze intact")
        else:
            s.status = "BELOW ZONE"
            s.headline = "BELOW ZONE"
            s.note = (f"{abs(s.pct_from_entry):.1f}% below the {_round(s.entry_lo)}–"
                      f"{_round(s.entry_hi)} zone — deeper retrace, still above {_round(stop)}")
    elif t1 and abs(price - t1) < abs(price - entry):
        s.status = "DON'T CHASE"
        s.headline = "DON'T CHASE"
        s.note = (f"{abs(s.pct_from_entry):.1f}% above entry, closer to first target than entry "
                  f"— no fresh entry here · work the {_round(entry)} {s.context}").strip()
    else:
        s.status = "WAIT FOR PULLBACK"
        s.headline = "WAIT FOR PULLBACK"
        s.note = (f"{abs(s.pct_from_entry):.1f}% above entry — let it come back to the zone · "
                  f"{_round(entry)} {s.context}").strip()


def score(s: Setup) -> float:
    """Rank actionable, well-shaped setups above the rest."""
    rr_part = min(max(s.entry_rr, 0), 5) / 5
    # closer to the entry zone is better; 15% away scores zero
    dist = 0.0 if s.entry_lo <= s.price <= s.entry_hi else \
        min(abs(s.price - s.entry) / s.entry, 0.15) / 0.15
    near_part = 1 - dist
    vol_part = min(max(s.vol_ratio, 0.4), 2.0) / 2.0
    status_part = 1 - STATUS_RANK.get(s.status, MAX_RANK) / MAX_RANK
    risk_part = 1 - min(max(s.entry_risk_pct, 0), 20) / 20  # prefer tight risk
    # confluence pulls the score toward or away from the setup's own direction;
    # every rule here is long-only, so a bearish tape is a genuine demerit
    bias_part = (max(min(s.bias, 100), -100) + 100) / 200      # 0..1
    # weights sum to 1.0
    return round(100 * (0.26 * status_part + 0.20 * near_part + 0.16 * rr_part
                        + 0.12 * risk_part + 0.08 * vol_part + 0.18 * bias_part), 1)


# --------------------------------------------------------------------------
# Scan rules
# --------------------------------------------------------------------------

def _volume_profile(df: pd.DataFrame) -> tuple[float, str, list[float]]:
    vol = df["Volume"].astype(float)
    avg20 = vol.tail(20).mean()
    ratio = float(vol.iloc[-1] / avg20) if avg20 else 0.0
    r5, r20 = vol.tail(5).mean(), avg20
    trend = "rising" if r5 > r20 * 1.15 else "fading" if r5 < r20 * 0.85 else "steady"
    bars = vol.tail(20).to_numpy(dtype=float)
    peak = bars.max() or 1.0
    return round(ratio, 2), trend, [round(float(b / peak), 3) for b in bars]


def _targets_from(low: float, high: float, price: float) -> list[float]:
    leg = high - low
    raw = [high, low + leg * 1.272, low + leg * 1.618, low + leg * 2.0]
    return [_round(t) for t in raw if t > price * 0.999][:4]


def scan_ote(df: pd.DataFrame) -> Setup | None:
    """Impulse leg up, price retracing into the 0.62-0.79 pocket."""
    if len(df) < 80:
        return None
    close, price = df["Close"], float(df["Close"].iloc[-1])
    highs, lows = df["High"], df["Low"]

    ph, pl = pivots(highs, 4, "high"), pivots(lows, 4, "low")
    if not ph or not pl:
        return None
    hi_i = ph[-1]
    prior_lows = [i for i in pl if i < hi_i]
    if not prior_lows:
        return None
    lo_i = prior_lows[-1]

    hi, lo = float(highs.iloc[hi_i]), float(lows.iloc[lo_i])
    leg = hi - lo
    if leg <= 0 or leg / lo < 0.06:          # need a real impulse
        return None
    if len(df) - 1 - hi_i > 40:              # and a recent one
        return None
    if price > hi * 1.02:                    # already extended past the leg
        return None

    entry_hi = hi - leg * 0.62
    entry_lo = hi - leg * 0.79
    entry = hi - leg * 0.705
    stop = lo - float(atr(df).iloc[-1]) * 0.5
    if stop <= 0 or entry <= stop:
        return None

    ratio, trend, bars = _volume_profile(df)
    return Setup(
        ticker="", name="", setup_type="ote", price=price,
        entry=_round(entry), entry_lo=_round(entry_lo), entry_hi=_round(entry_hi),
        stop=_round(stop), targets=_targets_from(lo, hi, price),
        vol_ratio=ratio, vol_trend=trend, vol_bars=bars,
        context="OTE", extra={"leg_low": _round(lo), "leg_high": _round(hi)},
    )


def scan_ma_pullback(df: pd.DataFrame) -> Setup | None:
    """Established uptrend easing back into a rising 21 EMA."""
    if len(df) < 220:
        return None
    close = df["Close"]
    price = float(close.iloc[-1])
    e21 = ema(close, 21)
    s50 = close.rolling(50).mean()
    s200 = close.rolling(200).mean()
    if not (price > float(s50.iloc[-1]) > float(s200.iloc[-1])):
        return None
    if float(s50.iloc[-1]) <= float(s50.iloc[-10]):        # 50 must be rising
        return None
    if price > float(e21.iloc[-1]) * 1.12:                  # too far above to be a pullback
        return None

    entry = float(e21.iloc[-1])
    recent_low = float(df["Low"].tail(20).min())
    stop = min(recent_low, entry) - float(atr(df).iloc[-1]) * 0.3
    if stop <= 0 or entry <= stop:
        return None
    recent_high = float(df["High"].tail(60).max())
    # the prior high has to be far enough above the EMA to be worth targeting,
    # otherwise T1 lands on top of the entry and the setup can never pay
    if recent_high < entry * 1.03:
        return None

    ratio, trend, bars = _volume_profile(df)
    return Setup(
        ticker="", name="", setup_type="ma", price=price,
        entry=_round(entry), entry_lo=_round(entry * 0.985), entry_hi=_round(entry * 1.015),
        stop=_round(stop), targets=_targets_from(stop, recent_high, price),
        vol_ratio=ratio, vol_trend=trend, vol_bars=bars,
        context="21 EMA", extra={"ma50": _round(float(s50.iloc[-1]))},
    )


def scan_breakout(df: pd.DataFrame) -> Setup | None:
    """Volatility squeeze coiling just under a recent high."""
    if len(df) < 140:
        return None
    close = df["Close"]
    price = float(close.iloc[-1])
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    width = (sd20 * 4) / ma20
    if width.isna().iloc[-1]:
        return None
    pct = float((width.tail(120) < width.iloc[-1]).mean())
    if pct > 0.25:                                   # must be in the tightest quartile
        return None

    range_hi = float(df["High"].tail(20).max())
    range_lo = float(df["Low"].tail(20).min())
    if price < range_hi * 0.93:                      # coiling near the top of the range
        return None
    if float(close.iloc[-1]) < float(close.rolling(200).mean().iloc[-1]):
        return None

    entry = range_hi
    stop = range_lo - float(atr(df).iloc[-1]) * 0.3
    if stop <= 0 or entry <= stop:
        return None
    height = range_hi - range_lo
    targets = [_round(t) for t in
               (range_hi + height, range_hi + height * 1.618, range_hi + height * 2.0)
               if t > price]

    ratio, trend, bars = _volume_profile(df)
    return Setup(
        ticker="", name="", setup_type="breakout", price=price,
        entry=_round(entry), entry_lo=_round(entry * 0.995), entry_hi=_round(entry * 1.01),
        stop=_round(stop), targets=targets,
        vol_ratio=ratio, vol_trend=trend, vol_bars=bars,
        context="range high", extra={"squeeze_pct": round(pct * 100, 1)},
    )


SCANS = {"ote": scan_ote, "ma": scan_ma_pullback, "breakout": scan_breakout}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def fetch_history(tickers: list[str], period: str = "1y", chunk: int = 100,
                  asof: date | None = None, lookback_days: int = 500,
                  forward_days: int = 0) -> dict[str, pd.DataFrame]:
    """Batch-download daily bars, tolerating individual failures.

    With `asof` set the window is anchored on that date instead of today, and
    `forward_days` of bars PAST it are fetched too -- the scan only ever sees
    bars up to the cutoff, while the later bars are what outcomes are scored
    against. Splitting happens in split_asof(), never here.
    """
    kwargs: dict = {"interval": "1d", "group_by": "ticker",
                    "auto_adjust": False, "progress": False, "threads": True}
    if asof is None:
        kwargs["period"] = period
    else:
        kwargs["start"] = (asof - timedelta(days=lookback_days)).isoformat()
        # +5 days of slack so the last forward bar is not clipped by a weekend
        kwargs["end"] = (asof + timedelta(days=forward_days + 5)).isoformat()

    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i: i + chunk]
        print(f"  fetching {i + 1}-{i + len(batch)} of {len(tickers)}…", file=sys.stderr)
        try:
            raw = yf.download(batch, **kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! batch failed: {exc}", file=sys.stderr)
            continue
        for t in batch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = sub.dropna(subset=["Close"])
                if len(sub) >= 80:
                    frames[t] = sub
            except (KeyError, TypeError):
                continue
    return frames


def split_asof(df: pd.DataFrame, asof: date | None):
    """(bars the scan may see, bars after the cutoff used to score outcomes)."""
    if asof is None:
        return df, df.iloc[0:0]
    idx = pd.Index([ts.date() for ts in df.index])
    return df[idx <= asof], df[idx > asof]


def scan_universe(universe: dict[str, str], types: list[str], limit: int = 60,
                  period: str = "1y", min_rr: float = 0.9,
                  asof: date | None = None, forward_days: int = 0,
                  frames: dict[str, pd.DataFrame] | None = None,
                  futures_out: dict[str, pd.DataFrame] | None = None) -> dict:
    """Scan the universe. With `asof`, every rule sees only bars up to that
    date; bars after it are handed back through `futures_out` for the backtest
    to score against, and never influence a signal."""
    if frames is None:
        frames = fetch_history(list(universe), period=period, asof=asof,
                               forward_days=forward_days)
    setups: list[Setup] = []

    for ticker, full in frames.items():
        df, future = split_asof(full, asof)
        if len(df) < 80:
            continue
        if futures_out is not None:
            futures_out[ticker] = future
        try:
            conf = confluence(df)
        except Exception:  # noqa: BLE001
            conf = {}
        for kind in types:
            try:
                s = SCANS[kind](df)
            except Exception:  # noqa: BLE001 - one bad frame must not stop the scan
                continue
            if s is None or not s.targets:
                continue
            s.ticker = ticker
            s.name = universe.get(ticker, ticker)
            s.confluence = conf
            s.bias = float(conf.get("bias", 0.0))
            classify(s)
            if s.status == "INVALIDATED":
                continue          # a scanner surfaces live setups, not dead ones
            if s.entry_rr < min_rr:
                continue          # T1 too close to the entry to be worth taking
            s.score = score(s)
            setups.append(s)

    setups.sort(key=lambda s: (STATUS_RANK.get(s.status, MAX_RANK), -s.score))
    kept = setups[:limit]

    moves = [s.pct_from_entry for s in kept]
    best = max(kept, key=lambda s: s.pct_from_entry, default=None)
    worst = min(kept, key=lambda s: s.pct_from_entry, default=None)
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "asof": asof.isoformat() if asof else None,
        "scanned": len(frames),
        "universe_size": len(universe),
        "types": types,
        "summary": {
            "count": len(kept),
            "avg_from_entry": round(float(np.mean(moves)), 2) if moves else 0.0,
            "up": sum(1 for m in moves if m >= 0),
            "down": sum(1 for m in moves if m < 0),
            "best": {"ticker": best.ticker, "pct": round(best.pct_from_entry, 2)} if best else None,
            "worst": {"ticker": worst.ticker, "pct": round(worst.pct_from_entry, 2)} if worst else None,
            "by_type": {k: sum(1 for s in kept if s.setup_type == k) for k in types},
        },
        "setups": [asdict(s) for s in kept],
    }


def print_table(result: dict) -> None:
    print(f"\n{result['summary']['count']} setups from {result['scanned']} names "
          f"({', '.join(result['types'])})\n")
    hdr = (f"{'TICKER':<8}{'TYPE':<10}{'STATUS':<18}{'PRICE':>9}{'ENTRY':>9}{'STOP':>9}"
           f"{'NEXT':>9}{'RR@now':>8}{'RR@ent':>8}{'FROM':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in result["setups"]:
        print(f"{s['ticker']:<8}{s['setup_type']:<10}{s['status']:<18}"
              f"{s['price']:>9.2f}{s['entry']:>9.2f}{s['stop']:>9.2f}{s['next_target']:>9.2f}"
              f"{s['rr']:>8.1f}{s['entry_rr']:>8.1f}{s['pct_from_entry']:>7.1f}%")
    print()


def main():
    p = argparse.ArgumentParser(description="Scan the NDX/SPX universe for trade setups.")
    p.add_argument("--types", default="ote,ma,breakout", help="comma list: ote,ma,breakout")
    p.add_argument("--limit", type=int, default=60, help="max setups to keep (default 60)")
    p.add_argument("--min-rr", type=float, default=0.9, dest="min_rr",
                   help="drop setups whose reward:risk at the entry is below this (default 0.9)")
    p.add_argument("--period", default="1y", help="history window (default 1y)")
    p.add_argument("--html", metavar="PATH", help="write a static HTML card grid")
    p.add_argument("--json", metavar="PATH", help="write the raw scan result as JSON")
    p.add_argument("--refresh-universe", action="store_true", help="re-read the index lists")
    p.add_argument("--universe", default="sp100", choices=list(INDEXES),
                   help="which universe to scan (default: sp100, the 100 largest S&P 500 names)")
    args = p.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip() in SCANS]
    if not types:
        p.error(f"--types must name at least one of {list(SCANS)}")

    universe = load_universe(refresh=args.refresh_universe, which=args.universe)
    print(f"universe: {len(universe)} tickers ({args.universe})", file=sys.stderr)
    result = scan_universe(universe, types, limit=args.limit, period=args.period,
                           min_rr=args.min_rr)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.json}", file=sys.stderr)
    if args.html:
        from render import render_static
        Path(args.html).write_text(render_static(result))
        print(f"wrote {args.html}", file=sys.stderr)
    if not args.html and not args.json:
        print_table(result)


if __name__ == "__main__":
    main()
