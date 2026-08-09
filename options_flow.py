#!/usr/bin/env python3
"""
Per-ticker dealer positioning: gamma exposure (GEX) and vanna exposure (VEX)
by strike, plus the levels the regime engine reads off them.

This is the expensive half of the scanner -- each ticker needs its own option
chain -- so it runs only for enriched names and for the detail view, never for
the whole universe.

Sign convention matches spx_dashboard: dealers are assumed long calls and short
puts against customer flow, so put exposure is booked negative. That is a
convention, not observed positioning.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from spx_dashboard import NY_TZ, bs_gamma, years_to_expiry

CONTRACT_MULT = 100


def _parse_expiry(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def bs_vanna(spot, strike, iv, t, r: float = 0.0):
    """d(vega)/d(spot) = -phi(d1) * d2 / sigma, per contract per 1 vol point."""
    spot = float(spot)
    strike = np.asarray(strike, dtype=float)
    iv = np.asarray(iv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = iv * np.sqrt(t)
        d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t) / vol
        d2 = d1 - vol
        v = -norm.pdf(d1) * d2 / iv
    v[~np.isfinite(v)] = 0.0
    return v


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("openInterest", "volume"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0)
    out["impliedVolatility"] = pd.to_numeric(out.get("impliedVolatility"), errors="coerce")
    # Yahoo prints 0 or absurd IV on dead strikes; drop rather than let it skew
    out.loc[(out["impliedVolatility"] <= 0.01) | (out["impliedVolatility"] > 5), "impliedVolatility"] = np.nan
    return out


def exposure_profile(ticker: str, spot: float | None = None, n_expiries: int = 2) -> dict | None:
    """GEX/VEX by strike across the nearest n expiries. None if there is no chain."""
    t = yf.Ticker(ticker)
    try:
        available = list(t.options)
    except Exception:
        return None
    # Drop anything already expired. Gamma is floored at a 5-minute time to
    # expiry, so a stale date does not error -- it silently produces an
    # astronomically large node that would dominate every level on the page.
    today = datetime.now(tz=NY_TZ).date()
    expiries = [e for e in available
                if _parse_expiry(e) is not None and _parse_expiry(e) >= today][:n_expiries]
    if not expiries:
        return None

    if spot is None:
        hist = t.history(period="1d", interval="1d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])

    frames = []
    for exp in expiries:
        try:
            chain = t.option_chain(exp)
        except Exception:
            continue
        calls, puts = _clean(chain.calls), _clean(chain.puts)
        if calls.empty and puts.empty:
            continue
        yrs = years_to_expiry(exp)
        strikes = np.array(sorted(set(calls["strike"]).union(set(puts["strike"]))))
        scale = spot ** 2 * 0.01          # $ per 1% move

        def leg(df, sign):
            oi = df.set_index("strike")["openInterest"].reindex(strikes).fillna(0).to_numpy()
            iv = df.set_index("strike")["impliedVolatility"].reindex(strikes).to_numpy()
            gamma = bs_gamma(spot, strikes, iv, yrs)
            vanna = bs_vanna(spot, strikes, iv, yrs)
            return (sign * gamma * oi * CONTRACT_MULT * scale,
                    sign * vanna * oi * CONTRACT_MULT * spot * 0.01,
                    oi)

        cg, cv, coi = leg(calls, 1.0)
        pg, pv, poi = leg(puts, -1.0)
        frames.append(pd.DataFrame({"strike": strikes, "gex": cg + pg, "vex": cv + pv,
                                    "call_oi": coi, "put_oi": poi, "expiry": exp}))

    if not frames:
        return None

    allf = pd.concat(frames)
    prof = allf.groupby("strike", as_index=False)[["gex", "vex", "call_oi", "put_oi"]].sum()
    prof = prof.sort_values("strike").reset_index(drop=True)

    atm_iv = _atm_iv(t, expiries[0], spot)
    return {"ticker": ticker.upper(), "spot": round(spot, 2), "expiries": expiries,
            "atm_iv": atm_iv, "profile": prof,
            "levels": derive_levels(prof, spot)}


def _atm_iv(t, expiry: str, spot: float) -> float | None:
    try:
        chain = t.option_chain(expiry)
    except Exception:
        return None
    vals = []
    for df in (_clean(chain.calls), _clean(chain.puts)):
        if df.empty:
            continue
        i = (df["strike"] - spot).abs().idxmin()
        iv = df.loc[i, "impliedVolatility"]
        if pd.notna(iv):
            vals.append(float(iv))
    return round(float(np.mean(vals)) * 100, 1) if vals else None


def _zero_gamma(prof: pd.DataFrame, spot: float) -> float | None:
    """Strike where cumulative gamma flips sign -- the flip / zero-gamma level."""
    s = prof["strike"].to_numpy()
    cum = np.cumsum(prof["gex"].to_numpy())
    sign = np.sign(cum)
    idx = np.flatnonzero(sign[:-1] * sign[1:] < 0)
    if not len(idx):
        return None
    # take the crossing nearest spot
    best = min(idx, key=lambda i: abs((s[i] + s[i + 1]) / 2 - spot))
    y0, y1 = cum[best], cum[best + 1]
    if y1 == y0:
        return round(float(s[best]), 2)
    return round(float(s[best] + (s[best + 1] - s[best]) * (-y0) / (y1 - y0)), 2)


def derive_levels(prof: pd.DataFrame, spot: float) -> dict:
    """Flip, pin, and the -gamma fuel nodes above and below spot."""
    pos, neg = prof[prof["gex"] > 0], prof[prof["gex"] < 0]

    pin_row = pos.loc[pos["gex"].idxmax()] if not pos.empty else None
    below = neg[neg["strike"] < spot]
    above = neg[neg["strike"] > spot]
    fuel_dn = below.loc[below["gex"].idxmin()] if not below.empty else None
    fuel_up = above.loc[above["gex"].idxmin()] if not above.empty else None

    # nearest big +gamma node under spot -- the "floor" that can be a trapdoor
    pos_below = pos[pos["strike"] < spot]
    floor_row = pos_below.loc[pos_below["strike"].idxmax()] if not pos_below.empty else None
    pos_above = pos[pos["strike"] > spot]
    ceil_row = pos_above.loc[pos_above["strike"].idxmin()] if not pos_above.empty else None

    # Where a break actually travels to: the biggest node BEYOND the level that
    # breaks, not the fuel node sitting between spot and that level.
    def beyond(strikes_mask):
        sub = prof[strikes_mask]
        if sub.empty:
            return None
        return sub.loc[sub["gex"].abs().idxmax()]

    target_dn = beyond(prof["strike"] < floor_row["strike"]) if floor_row is not None else \
        beyond(prof["strike"] < spot)
    target_up = beyond(prof["strike"] > ceil_row["strike"]) if ceil_row is not None else \
        beyond(prof["strike"] > spot)

    flip = _zero_gamma(prof, spot)
    net_gex = float(prof["gex"].sum())
    net_vex = float(prof["vex"].sum())
    vanna_row = prof.loc[prof["vex"].abs().idxmax()] if not prof.empty else None

    def node(row):
        return None if row is None else {"strike": round(float(row["strike"]), 2),
                                         "gex": round(float(row["gex"]), 0),
                                         "vex": round(float(row["vex"]), 0)}

    return {
        "flip": flip,
        "flip_cushion": round(abs(spot - flip), 2) if flip else None,
        "pin": node(pin_row),
        "floor": node(floor_row),
        "ceiling": node(ceil_row),
        "fuel_down": node(fuel_dn),
        "fuel_up": node(fuel_up),
        "target_down": node(target_dn),
        "target_up": node(target_up),
        "net_gex": round(net_gex, 0),
        "net_vex": round(net_vex, 0),
        "vanna_node": node(vanna_row),
        "gex_regime": "long gamma" if net_gex >= 0 else "short gamma",
        "vex_lean": "call-side lean" if net_vex >= 0 else "put-side lean",
    }


def profile_payload(prof_dict: dict, window: float = 0.12, max_rows: int = 40) -> dict:
    """Trim the profile to a plottable window around spot and make it JSON-safe."""
    prof, spot = prof_dict["profile"], prof_dict["spot"]
    sel = prof[(prof["strike"] >= spot * (1 - window)) & (prof["strike"] <= spot * (1 + window))]
    if sel.empty:
        sel = prof
    if len(sel) > max_rows:                      # keep the biggest nodes, then re-sort
        sel = sel.reindex(sel["gex"].abs().sort_values(ascending=False).index[:max_rows])
        sel = sel.sort_values("strike")
    return {
        "strikes": [round(float(v), 2) for v in sel["strike"]],
        "gex": [round(float(v), 0) for v in sel["gex"]],
        "vex": [round(float(v), 0) for v in sel["vex"]],
        "call_oi": [int(v) for v in sel["call_oi"]],
        "put_oi": [int(v) for v in sel["put_oi"]],
    }
