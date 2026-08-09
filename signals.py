#!/usr/bin/env python3
"""
Daily-bar confluence signals: MACD, RSI, the 200 SMA, realized volatility,
volume behaviour and trend structure.

These are cheap -- everything here comes off bars the scanner already has, so
they run over the whole universe. Options-derived context (GEX/VEX/IV) is a
separate, far more expensive fetch and lives in options_flow.py.

Each signal reports a direction (+1 bullish / 0 neutral / -1 bearish) and a
short human phrase. confluence() rolls them into a -100..+100 bias score that
the regime engine folds into conviction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def realized_vol(close: pd.Series, n: int) -> float:
    r = np.log(close / close.shift(1)).dropna()
    if len(r) < n:
        return float("nan")
    return float(r.tail(n).std() * np.sqrt(252) * 100)


# --------------------------------------------------------------------------
# Individual reads
# --------------------------------------------------------------------------

def macd_state(close: pd.Series) -> dict:
    line, signal, hist = macd(close)
    above = line > signal
    # bars since the most recent cross
    flips = above.ne(above.shift(1))
    idx = np.flatnonzero(flips.to_numpy()[1:]) + 1
    bars_since = int(len(close) - 1 - idx[-1]) if len(idx) else int(len(close))

    bull = bool(above.iloc[-1])
    rising = bool(hist.iloc[-1] > hist.iloc[-2]) if len(hist) > 1 else False
    fresh = bars_since <= 5

    if bull and fresh:
        phrase, direction = f"fresh bull cross {bars_since}d ago", 1
    elif bull:
        phrase, direction = ("above signal, histogram building" if rising
                             else "above signal but histogram rolling"), (1 if rising else 0)
    elif not bull and fresh:
        phrase, direction = f"fresh bear cross {bars_since}d ago", -1
    else:
        phrase, direction = ("below signal, histogram still falling" if not rising
                             else "below signal but histogram curling up"), (-1 if not rising else 0)

    return {"bull": bull, "fresh": fresh, "bars_since_cross": bars_since,
            "hist": round(float(hist.iloc[-1]), 4), "hist_rising": rising,
            "direction": direction, "phrase": f"MACD {phrase}"}


def rsi_state(close: pd.Series) -> dict:
    r = rsi(close)
    v = float(r.iloc[-1])
    prev = float(r.iloc[-6]) if len(r) > 6 else v
    rising = v > prev
    if v >= 70:
        zone, direction = "overbought", -1
    elif v >= 55:
        zone, direction = "bullish", 1
    elif v > 45:
        zone, direction = "neutral", 0
    elif v > 30:
        zone, direction = "bearish", -1
    else:
        zone, direction = "oversold", 1     # stretched, mean-reversion lean
    return {"value": round(v, 1), "zone": zone, "rising": rising,
            "direction": direction,
            "phrase": f"RSI {v:.0f} ({zone}, {'rising' if rising else 'falling'})"}


def sma200_state(close: pd.Series) -> dict:
    if len(close) < 200:
        return {"value": None, "above": None, "slope_pct": None, "direction": 0,
                "phrase": "200 SMA n/a (needs 200 bars)"}
    sma = close.rolling(200).mean()
    v, price = float(sma.iloc[-1]), float(close.iloc[-1])
    above = price > v
    slope = (v - float(sma.iloc[-21])) / float(sma.iloc[-21]) * 100
    dist = (price - v) / v * 100
    direction = 1 if (above and slope > 0) else -1 if (not above and slope < 0) else 0
    return {"value": round(v, 2), "above": above, "dist_pct": round(dist, 1),
            "slope_pct": round(slope, 2), "direction": direction,
            "phrase": (f"{'above' if above else 'below'} a "
                       f"{'rising' if slope > 0 else 'falling'} 200 SMA ({dist:+.1f}%)")}


def volume_state(df: pd.DataFrame) -> dict:
    vol = df["Volume"].astype(float)
    avg20 = float(vol.tail(20).mean())
    ratio = float(vol.iloc[-1]) / avg20 if avg20 else 0.0
    r5 = float(vol.tail(5).mean())
    trend = "rising" if r5 > avg20 * 1.15 else "fading" if r5 < avg20 * 0.85 else "steady"
    # is the recent push happening on expanding volume?
    up_day = float(df["Close"].iloc[-1]) >= float(df["Close"].iloc[-2])
    direction = 1 if (ratio > 1.2 and up_day) else -1 if (ratio > 1.2 and not up_day) else 0
    return {"ratio": round(ratio, 2), "avg20": avg20, "trend": trend,
            "direction": direction,
            "phrase": f"volume {ratio:.2f}× 20d avg and {trend}"}


def trend_state(df: pd.DataFrame) -> dict:
    """Structure read: MA stack plus whether swings are still stepping up."""
    close = df["Close"]
    e21, s50 = ema(close, 21), close.rolling(50).mean()
    price = float(close.iloc[-1])
    stack_up = price > float(e21.iloc[-1]) > float(s50.iloc[-1])
    stack_dn = price < float(e21.iloc[-1]) < float(s50.iloc[-1])

    hi_recent, hi_prior = float(df["High"].tail(20).max()), float(df["High"].iloc[-60:-20].max()) if len(df) > 60 else float("nan")
    lo_recent, lo_prior = float(df["Low"].tail(20).min()), float(df["Low"].iloc[-60:-20].min()) if len(df) > 60 else float("nan")
    hh = bool(hi_recent > hi_prior) if not np.isnan(hi_prior) else False
    hl = bool(lo_recent > lo_prior) if not np.isnan(lo_prior) else False

    if stack_up and hh and hl:
        label, direction = "uptrend — higher highs and higher lows", 1
    elif stack_up:
        label, direction = "up-stacked MAs, structure mixed", 1
    elif stack_dn and not hh and not hl:
        label, direction = "downtrend — lower highs and lower lows", -1
    elif stack_dn:
        label, direction = "down-stacked MAs", -1
    else:
        label, direction = "range / no clean structure", 0
    return {"stack_up": stack_up, "stack_dn": stack_dn, "higher_high": hh,
            "higher_low": hl, "direction": direction, "phrase": label}


def vol_state(close: pd.Series) -> dict:
    """Realized vol now vs its own recent baseline -- an IV stand-in until the
    option chain is fetched, which only happens for enriched names."""
    hv20, hv60 = realized_vol(close, 20), realized_vol(close, 60)
    ratio = hv20 / hv60 if hv60 and not np.isnan(hv60) and hv60 > 0 else float("nan")
    if np.isnan(ratio):
        regime = "unknown"
    elif ratio > 1.25:
        regime = "expanding"
    elif ratio < 0.8:
        regime = "compressing"
    else:
        regime = "steady"
    return {"hv20": None if np.isnan(hv20) else round(hv20, 1),
            "hv60": None if np.isnan(hv60) else round(hv60, 1),
            "ratio": None if np.isnan(ratio) else round(ratio, 2),
            "regime": regime, "direction": 0,
            "phrase": (f"realized vol {hv20:.0f}% vs {hv60:.0f}% baseline ({regime})"
                       if not np.isnan(hv20) and not np.isnan(hv60) else "realized vol n/a")}


# --------------------------------------------------------------------------
# Roll-up
# --------------------------------------------------------------------------

WEIGHTS = {"macd": 0.26, "rsi": 0.16, "sma200": 0.24, "trend": 0.24, "volume": 0.10}


def confluence(df: pd.DataFrame) -> dict:
    close = df["Close"]
    parts = {
        "macd": macd_state(close),
        "rsi": rsi_state(close),
        "sma200": sma200_state(close),
        "trend": trend_state(df),
        "volume": volume_state(df),
        "vol": vol_state(close),
    }
    bias = sum(WEIGHTS[k] * parts[k]["direction"] for k in WEIGHTS)
    agree = sum(1 for k in WEIGHTS if parts[k]["direction"] > 0)
    disagree = sum(1 for k in WEIGHTS if parts[k]["direction"] < 0)
    parts["bias"] = round(bias * 100, 1)          # -100..+100
    parts["agree"] = agree
    parts["disagree"] = disagree
    parts["aligned"] = agree >= 3 and disagree == 0
    # A real conflict is the STRUCTURE disagreeing with itself. Momentum easing
    # inside an intact uptrend (trend+ / 200SMA+ with MACD- / RSI overbought) is
    # the pullback signature these rules hunt for, not a contradiction -- counting
    # it as one penalised exactly the setups the scanner exists to find.
    parts["conflicted"] = parts["trend"]["direction"] * parts["sma200"]["direction"] < 0
    return parts
