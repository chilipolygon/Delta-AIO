#!/usr/bin/env python3
"""
Step-through replay: walk a past period one day at a time, decide which setups
to take, and watch the positions update as the days advance.

This is the interactive counterpart to backtest.py. The bulk backtest scores
every signal automatically and shows the outcome immediately; a replay shows
you ONLY what was knowable on the cursor date and makes you choose. That
distinction is the whole point, so the replay scan never carries forward
information -- no outcome badges, no R multiples on setups you have not taken.

Advancing a day does three things to the open book, in this order:

  1. stop first -- if the bar's low touched the stop, the position is closed
     there. When a single daily bar spans both the stop and a target, order
     within the bar is unknowable, so the pessimistic reading is taken. This
     matches backtest.evaluate() exactly; the two must not disagree.
  2. final target -- if the bar's high reached the last target, close there.
  3. otherwise mark to that day's close.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

_bar_cache: dict[tuple[str, str, str], pd.DataFrame] = {}


def _parse(d) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


def bars_for(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily bars for one ticker, cached per (ticker, window)."""
    key = (ticker.upper(), start.isoformat(), end.isoformat())
    if key in _bar_cache:
        return _bar_cache[key]
    try:
        df = yf.Ticker(ticker).history(
            start=start.isoformat(), end=(end + timedelta(days=3)).isoformat(),
            interval="1d", auto_adjust=False)
        df = df.dropna(subset=["Close"])
    except Exception:  # noqa: BLE001
        df = pd.DataFrame()
    _bar_cache[key] = df
    return df


def bar_on(ticker: str, day: date, window_start: date) -> pd.Series | None:
    """The bar for `day`, or None if the market was shut."""
    df = bars_for(ticker, window_start, day)
    if df.empty:
        return None
    for ts, row in zip(df.index, df.to_dict("records")):
        if ts.date() == day:
            return pd.Series(row)
    return None


def next_session(ticker: str, after: date, window_start: date, limit: int = 10) -> date | None:
    """The next date with a bar. Uses a real series rather than a weekday guess,
    so market holidays are skipped instead of producing an empty step."""
    df = bars_for(ticker, window_start, after + timedelta(days=limit + 5))
    if df.empty:
        return None
    for ts in df.index:
        if ts.date() > after:
            return ts.date()
    return None


def advance(pf, to_day: date, window_start: date) -> list[dict]:
    """Mark and resolve the open book at `to_day`. Returns what happened."""
    events: list[dict] = []
    for pos in list(pf.positions):
        bar = bar_on(pos.ticker, to_day, window_start)
        if bar is None:
            continue
        low, high, close = float(bar["Low"]), float(bar["High"]), float(bar["Close"])
        stop = pos.stop
        targets = sorted(pos.targets or [])

        # 1) stop first -- see the module docstring
        if stop is not None and low <= stop:
            exit_px = _underlying_to_exit(pos, stop)
            closed = pf.close(pos.id, exit_px, reason="STOPPED", when=to_day.isoformat())
            events.append({"kind": "stopped", "ticker": pos.ticker,
                           "price": round(stop, 2), "realized": closed.realized,
                           "r_multiple": closed.r_multiple,
                           "text": f"{pos.ticker} stopped out at {stop:,.2f}"})
            continue

        # 2) final target
        if targets and high >= targets[-1]:
            exit_px = _underlying_to_exit(pos, targets[-1])
            closed = pf.close(pos.id, exit_px, reason="ALL TARGETS", when=to_day.isoformat())
            events.append({"kind": "target", "ticker": pos.ticker,
                           "price": round(targets[-1], 2), "realized": closed.realized,
                           "r_multiple": closed.r_multiple,
                           "text": f"{pos.ticker} hit its final target at {targets[-1]:,.2f}"})
            continue

        # 3) still open -- note any interim targets tagged today, then mark
        hit = [t for t in targets if high >= t]
        if hit:
            events.append({"kind": "progress", "ticker": pos.ticker,
                           "price": round(hit[-1], 2),
                           "text": f"{pos.ticker} tagged {len(hit)}/{len(targets)} targets "
                                   f"(high {high:,.2f})"})
        events.append({"kind": "mark", "ticker": pos.ticker, "price": round(close, 2),
                       "text": f"{pos.ticker} marked at {close:,.2f}"})
    return events


def _underlying_to_exit(pos, underlying_px: float) -> float:
    """Positions are priced in their own instrument: a stock exits at the
    underlying level, an option at what the contract is worth there."""
    if pos.kind != "option" or not pos.contract:
        return float(underlying_px)
    from contracts import reprice
    return float(reprice(pos.contract, underlying_px))


def mark_positions(pf, day: date, window_start: date) -> None:
    prices = {}
    for pos in pf.positions:
        bar = bar_on(pos.ticker, day, window_start)
        if bar is not None:
            prices[pos.ticker] = float(bar["Close"])
    pf.mark_to_market(prices)


def state(pf) -> dict:
    r = pf.replay or {}
    return {
        "active": bool(r),
        "start": r.get("start"),
        "cursor": r.get("cursor"),
        "types": r.get("types"),
        "universe": r.get("universe"),
        "days_elapsed": r.get("days_elapsed", 0),
        "log": (r.get("log") or [])[-40:],
    }
