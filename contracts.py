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


def implied_vol(price: float, spot: float, strike: float, t: float,
                kind: str = "call", lo: float = 0.01, hi: float = 5.0) -> float | None:
    """Back the volatility out of a traded price by bisection.

    Yahoo's impliedVolatility field is frequently wrong on individual strikes --
    a 27%-vol contract can be published at 91% -- and every model number here
    (fair value, delta, the value at the target and at the stop) is built on
    that input. The quoted mid is real money changing hands, so where one
    exists it is the better source of truth.
    """
    price, spot, strike, t = float(price), float(spot), float(strike), float(t)
    if price <= 0 or t <= 0 or spot <= 0 or strike <= 0:
        return None
    intrinsic = max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
    if price < intrinsic - 1e-6:
        return None                      # below intrinsic: no vol solves it
    if bs_price(spot, strike, hi, t, kind) < price:
        return None                      # beyond the search range
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(spot, strike, mid, t, kind) < price:
            lo = mid
        else:
            hi = mid
    v = (lo + hi) / 2
    return round(v, 4) if 0.011 < v < 4.99 else None


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

    # Prefer the vol implied by the traded mid over Yahoo's IV field.
    solved = implied_vol(mid, spot, strike, yrs, "call") if mid > 0 else None
    use_iv = solved or (iv if (not np.isnan(iv) and iv > 0) else 0.30)
    iv_source = "solved from mid" if solved else ("chain" if not np.isnan(iv) else "assumed 30%")
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
        "iv": round(use_iv * 100, 1),
        "iv_source": iv_source,
        "iv_quoted": None if np.isnan(iv) else round(iv * 100, 1),
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
                          f"quoted mid {mid:.2f} vs model {model:.2f} — one of the two is wrong; "
                          f"cost, breakeven and sizing all rest on the quote"),
        "iv_warning": (f"chain IV {iv * 100:.0f}% but the mid implies {use_iv * 100:.0f}% — "
                       f"using the implied figure for all model values"
                       if (solved and not np.isnan(iv) and iv > 0
                           and max(iv / solved, solved / iv) > 1.5) else None),
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


def chain_table(ticker: str, setup: dict | None = None, expiry: str | None = None,
                spot: float | None = None, width: float = 0.18,
                max_rows: int = 60) -> dict | None:
    """The option chain around spot for one expiry, for manual selection.

    Rows carry what a chain ladder shows -- bid/ask/mid, IV, delta, open
    interest, volume -- plus, when a setup is supplied, the modelled value of
    each strike at that setup's first target and at its stop. IV is solved from
    the traded mid wherever one exists (see implied_vol).
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

    expiry = expiry if expiry in {e for e, _ in future} else future[0][0]
    exp_date = dict(future)[expiry]
    dte = (exp_date - today).days

    try:
        chain = t.option_chain(expiry)
    except Exception:
        return None

    if spot is None:
        spot = float((setup or {}).get("price") or 0)
    if not spot:
        hist = t.history(period="1d", interval="1d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

    yrs = years_to_expiry(expiry)
    targets = (setup or {}).get("targets") or []
    t1 = float(targets[0]) if targets else spot * 1.05
    stop = float((setup or {}).get("stop") or spot * 0.95)
    later = max(yrs * 0.5, 1 / 365)

    def side(df, kind):
        df = _clean(df).dropna(subset=["strike"])
        df = df[(df["strike"] >= spot * (1 - width)) & (df["strike"] <= spot * (1 + width))]
        if len(df) > max_rows:
            df = df.reindex((df["strike"] - spot).abs().sort_values().index[:max_rows])
        rows = []
        for _, r in df.sort_values("strike").iterrows():
            k = float(r["strike"])
            bid, ask = float(r.get("bid") or 0), float(r.get("ask") or 0)
            mid = _mid(r)
            quoted = float(r["impliedVolatility"]) if pd.notna(r.get("impliedVolatility")) else float("nan")
            solved = implied_vol(mid, spot, k, yrs, kind) if mid > 0 else None
            iv = solved or (quoted if (not np.isnan(quoted) and quoted > 0) else 0.30)
            spread_pct = ((ask - bid) / mid * 100) if (mid > 0 and bid > 0 and ask > 0) else None
            sym = str(r.get("contractSymbol") or "").strip()
            if not sym:   # some rows come back without one; build the OCC form
                sym = (f"{ticker.upper()}{exp_date:%y%m%d}"
                       f"{'C' if kind == 'call' else 'P'}{int(round(k * 1000)):08d}")
            rows.append({
                "strike": round(k, 2), "kind": kind,
                "symbol": sym,
                "expiry": expiry,
                "bid": round(bid, 2), "ask": round(ask, 2), "mid": round(mid, 2),
                "spread_pct": None if spread_pct is None else round(spread_pct, 1),
                "iv": round(iv * 100, 1),
                "iv_quoted": None if np.isnan(quoted) else round(quoted * 100, 1),
                "iv_source": "mid" if solved else ("chain" if not np.isnan(quoted) else "assumed"),
                "delta": round(bs_delta(spot, k, iv, yrs, kind), 3),
                "open_interest": int(r.get("openInterest") or 0),
                "volume": int(r.get("volume") or 0),
                "itm": bool(spot > k) if kind == "call" else bool(spot < k),
                "cost_per_contract": round((mid if mid > 0 else bs_price(spot, k, iv, yrs, kind)) * 100, 2),
                "breakeven": round(k + mid, 2) if kind == "call" else round(k - mid, 2),
                "at_t1": round(bs_price(t1, k, iv, later, kind), 2),
                "at_stop": round(bs_price(stop, k, iv, later, kind), 2),
                "liquidity": _liquidity(r, spread_pct if spread_pct is not None else float("nan")),
            })
        return rows

    return {
        "ticker": ticker.upper(), "spot": round(float(spot), 2),
        "expiry": expiry, "dte": dte,
        "expiries": [{"date": e, "dte": (d - today).days} for e, d in future[:14]],
        "calls": side(chain.calls, "call"),
        "puts": side(chain.puts, "put"),
        "t1": round(t1, 2), "stop": round(stop, 2),
        "note": ("Model values assume r=0, the IV shown, and about half the remaining "
                 "time elapsed at the target."),
    }
