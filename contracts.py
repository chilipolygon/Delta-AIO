#!/usr/bin/env python3
"""
Pick a concrete option contract for a setup, and price it.

The scan itself never gets here -- selecting a contract needs the ticker's
option chain, which is one request per name. Contracts are resolved where the
chain is already being fetched: the detail drawer and the portfolio.

Everything priced here uses Black-Scholes with r=0 and no dividend, matching
the rest of the project. Model prices are for sizing and what-if only; the
quoted bid/ask is what you would actually trade against, and the spread is
reported so a wide, illiquid contract is visible rather than hidden.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from options_flow import _clean, _parse_expiry
from spx_dashboard import NY_TZ, years_to_expiry


def bs_price(spot, strike, iv, t, kind="call", r: float = 0.0) -> float:
    """Black-Scholes premium per share. Falls back to intrinsic at t<=0."""
    spot, strike, iv, t = float(spot), float(strike), float(iv), float(t)
    if t <= 0 or iv <= 0:
        return max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
    vol = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / vol
    d2 = d1 - vol
    if kind == "call":
        return spot * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)
    return strike * math.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_delta(spot, strike, iv, t, kind="call", r: float = 0.0) -> float:
    spot, strike, iv, t = float(spot), float(strike), float(iv), float(t)
    if t <= 0 or iv <= 0:
        itm = (spot > strike) if kind == "call" else (spot < strike)
        return (1.0 if kind == "call" else -1.0) if itm else 0.0
    vol = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / vol
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0


def _mid(row) -> float:
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    last = float(row.get("lastPrice") or 0)
    return last if last > 0 else max(bid, ask)


def pick_contract(ticker: str, setup: dict, min_dte: int = 30,
                  moneyness: str = "atm", spot: float | None = None) -> dict | None:
    """Choose one contract for a setup and return it fully priced.

    Direction follows the setup: every scan rule here is long-only, so it is a
    call. Expiry is the first listed one with at least `min_dte` days -- targets
    are swing levels, and buying less time than the thesis needs is the most
    common way an otherwise correct setup still loses.
    """
    t = yf.Ticker(ticker)
    try:
        available = list(t.options)
    except Exception:
        return None
    today = datetime.now(tz=NY_TZ).date()
    dated = [(e, _parse_expiry(e)) for e in available]
    future = [(e, d) for e, d in dated if d and (d - today).days >= 0]
    if not future:
        return None

    enough = [(e, d) for e, d in future if (d - today).days >= min_dte]
    expiry, exp_date = (enough[0] if enough else future[-1])
    dte = (exp_date - today).days

    try:
        chain = t.option_chain(expiry)
    except Exception:
        return None
    calls = _clean(chain.calls)
    if calls.empty:
        return None

    if spot is None:
        spot = float(setup.get("price") or 0)
    entry = float(setup.get("entry") or spot)
    targets = setup.get("targets") or []
    t1 = float(targets[0]) if targets else entry * 1.05

    # target strike by preference
    if moneyness == "otm":
        want = entry + (t1 - entry) * 0.5     # halfway to the first target
    elif moneyness == "itm":
        want = entry * 0.97
    else:
        want = entry                          # at the money relative to the entry

    calls = calls.dropna(subset=["strike"])
    row = calls.loc[(calls["strike"] - want).abs().idxmin()]
    strike = float(row["strike"])
    iv = float(row["impliedVolatility"]) if pd.notna(row.get("impliedVolatility")) else float("nan")
    yrs = years_to_expiry(expiry)

    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    mid = _mid(row)
    spread = (ask - bid) if (bid > 0 and ask > 0) else float("nan")
    spread_pct = (spread / mid * 100) if mid > 0 and not np.isnan(spread) else float("nan")

    use_iv = iv if (not np.isnan(iv) and iv > 0) else 0.30
    model = bs_price(spot, strike, use_iv, yrs, "call")
    delta = bs_delta(spot, strike, use_iv, yrs, "call")

    # what the same contract is worth at the setup's own levels, holding IV and
    # letting roughly half the remaining time pass
    later = max(yrs * 0.5, 1 / 365)
    at_t1 = bs_price(t1, strike, use_iv, later, "call")
    at_stop = bs_price(float(setup.get("stop") or entry * 0.95), strike, use_iv, later, "call")

    entry_px = mid if mid > 0 else model
    # A quote that disagrees badly with the model usually means a stale or
    # junk print on a thin strike. Every downstream number -- cost, breakeven,
    # the P&L estimates, and the position size -- is built on this price, so
    # the disagreement has to be visible rather than silently propagated.
    quote_ok = True
    if mid > 0 and model > 0:
        ratio = max(mid / model, model / mid)
        quote_ok = bool(ratio < 3.0)   # numpy comparison -> numpy bool otherwise
    return {
        "ticker": ticker.upper(),
        "symbol": str(row.get("contractSymbol") or f"{ticker.upper()}{exp_date:%y%m%d}C{int(strike * 1000):08d}"),
        "kind": "call",
        "expiry": expiry,
        "dte": dte,
        "strike": round(strike, 2),
        "moneyness": moneyness,
        "spot": round(float(spot), 2),
        "bid": round(bid, 2), "ask": round(ask, 2),
        "mid": round(mid, 2),
        "spread": None if np.isnan(spread) else round(spread, 2),
        "spread_pct": None if np.isnan(spread_pct) else round(spread_pct, 1),
        "iv": None if np.isnan(iv) else round(iv * 100, 1),
        "model_price": round(model, 2),
        "delta": round(delta, 3),
        "open_interest": int(row.get("openInterest") or 0),
        "volume": int(row.get("volume") or 0),
        "breakeven": round(strike + entry_px, 2),
        "cost_per_contract": round(entry_px * 100, 2),
        "est_value_at_t1": round(at_t1, 2),
        "est_value_at_stop": round(at_stop, 2),
        "est_pnl_at_t1": round((at_t1 - entry_px) * 100, 2),
        "est_pnl_at_stop": round((at_stop - entry_px) * 100, 2),
        "liquidity": _liquidity(row, spread_pct),
        "quote_ok": quote_ok,
        "quote_warning": (None if quote_ok else
                          f"quoted mid {mid:.2f} vs model {model:.2f} — treat the quote as stale; "
                          f"cost, breakeven and sizing all rest on it"),
        "note": (f"{moneyness.upper()} call, {dte}d out. Model prices assume r=0, flat IV and "
                 f"about half the remaining time elapsed at the target."),
    }


def _liquidity(row, spread_pct) -> str:
    oi = int(row.get("openInterest") or 0)
    vol = int(row.get("volume") or 0)
    if oi < 50 and vol < 10:
        return "thin — few contracts outstanding, expect slippage"
    if not np.isnan(spread_pct) and spread_pct > 15:
        return f"wide — {spread_pct:.0f}% bid/ask spread"
    if oi > 500 or vol > 100:
        return "liquid"
    return "moderate"


def reprice(contract: dict, spot: float, days_elapsed: int = 0) -> float:
    """Mark a held contract to a new spot, decaying time. Used for portfolio
    marks when a live quote is not being fetched."""
    iv = (contract.get("iv") or 30.0) / 100
    exp = _parse_expiry(contract["expiry"])
    if exp is None:
        return contract.get("mid", 0.0)
    remaining = max((exp - date.today()).days - days_elapsed, 0) / 365
    return round(bs_price(spot, contract["strike"], iv, remaining, contract.get("kind", "call")), 2)
