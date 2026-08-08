#!/usr/bin/env python3
"""
SPX options / gamma dashboard, sourced from Yahoo Finance.

Yahoo Finance does not publish an options chain for the ^SPX / ^GSPC index
itself, so this script pulls the SPY chain (the standard liquid proxy),
computes everything off of it, and then converts the dollar levels to
SPX-equivalent terms using the live SPX/SPY ratio -- the same approach
used by most retail 0DTE dashboards.

Usage:
    python3 spx_dashboard.py [--ticker SPY] [--index ^GSPC] [--expiry YYYY-MM-DD]

Requires: yfinance, pandas, numpy, scipy
    pip install yfinance pandas numpy scipy
"""

import argparse
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

RISK_FREE_RATE = 0.05
SECONDS_PER_YEAR = 365.25 * 24 * 3600
NY_TZ = ZoneInfo("America/New_York")
MIN_T_YEARS = (5 / 60) / (365.25 * 24)  # floor time-to-expiry at 5 minutes


# --------------------------------------------------------------------------
# Data fetch helpers
# --------------------------------------------------------------------------

def fetch_daily_levels(ticker: str):
    t = yf.Ticker(ticker)
    hist = t.history(period="5d", interval="1d", auto_adjust=False)
    if len(hist) < 2:
        raise RuntimeError(f"Not enough daily history returned for {ticker}")
    return {
        "spot": float(hist["Close"].iloc[-1]),
        "open": float(hist["Open"].iloc[-1]),
        "prev_close": float(hist["Close"].iloc[-2]),
        "pdh": float(hist["High"].iloc[-2]),
        "pdl": float(hist["Low"].iloc[-2]),
    }


def fetch_rsi(ticker: str, period: int = 14) -> float:
    t = yf.Ticker(ticker)
    hist = t.history(period="3mo", interval="1d", auto_adjust=False)
    closes = hist["Close"].dropna()
    delta = closes.diff().dropna()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def fetch_vwap(ticker: str) -> float:
    t = yf.Ticker(ticker)
    intraday = t.history(period="1d", interval="1m", auto_adjust=False)
    if intraday.empty:
        # market likely closed -- fall back to the most recent session
        intraday = t.history(period="5d", interval="1m", auto_adjust=False)
        last_day = intraday.index[-1].date()
        intraday = intraday[intraday.index.date == last_day]

    typical_price = (intraday["High"] + intraday["Low"] + intraday["Close"]) / 3
    vwap = (typical_price * intraday["Volume"]).cumsum() / intraday["Volume"].cumsum()
    return float(vwap.iloc[-1])


def fetch_last_close(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No daily history returned for {ticker}")
    return float(hist["Close"].iloc[-1])


def pick_expiry(t: yf.Ticker, requested: str | None) -> str:
    expiries = t.options
    if not expiries:
        raise RuntimeError("No options expiries returned for this ticker")
    if requested:
        if requested not in expiries:
            raise RuntimeError(f"Expiry {requested} not available. Choices: {expiries}")
        return requested
    return expiries[0]


# --------------------------------------------------------------------------
# Options analytics
# --------------------------------------------------------------------------

def years_to_expiry(expiry_str: str) -> float:
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
        hour=16, minute=0, second=0, tzinfo=NY_TZ
    )
    now = datetime.now(tz=NY_TZ)
    seconds = max((expiry_dt - now).total_seconds(), 0)
    years = seconds / SECONDS_PER_YEAR
    return max(years, MIN_T_YEARS)


def bs_gamma(spot: float, strike: np.ndarray, iv: np.ndarray, t_years: float,
             r: float = RISK_FREE_RATE) -> np.ndarray:
    """Black-Scholes gamma (same formula for calls and puts)."""
    iv = np.asarray(iv, dtype=float)
    iv = np.where(iv > 0, iv, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * t_years) / (iv * np.sqrt(t_years))
        gamma = norm.pdf(d1) / (spot * iv * np.sqrt(t_years))
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame):
    strikes = np.array(sorted(set(calls["strike"]).union(set(puts["strike"]))))
    call_k = calls["strike"].to_numpy()
    call_oi = calls["openInterest"].fillna(0).to_numpy()
    put_k = puts["strike"].to_numpy()
    put_oi = puts["openInterest"].fillna(0).to_numpy()

    total_payout = np.empty_like(strikes, dtype=float)
    for i, s in enumerate(strikes):
        call_loss = np.clip(s - call_k, 0, None) * call_oi
        put_loss = np.clip(put_k - s, 0, None) * put_oi
        total_payout[i] = call_loss.sum() + put_loss.sum()

    idx = int(np.argmin(total_payout))
    return float(strikes[idx])


def compute_gex(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, t_years: float):
    """Dollar gamma exposure per 1% move in the underlying, in $."""
    call_gamma = bs_gamma(spot, calls["strike"].to_numpy(), calls["impliedVolatility"].to_numpy(), t_years)
    put_gamma = bs_gamma(spot, puts["strike"].to_numpy(), puts["impliedVolatility"].to_numpy(), t_years)

    call_oi = calls["openInterest"].fillna(0).to_numpy()
    put_oi = puts["openInterest"].fillna(0).to_numpy()

    contract_mult = 100
    scale = spot ** 2 * 0.01  # dollar change per 1% move

    call_gex = float((call_gamma * call_oi * contract_mult * scale).sum())
    # Convention: dealers assumed long calls / short puts against customer flow,
    # so put gamma exposure is booked as negative.
    put_gex = -float((put_gamma * put_oi * contract_mult * scale).sum())

    return call_gex, put_gex


def net_gex_at_spot(calls: pd.DataFrame, puts: pd.DataFrame, hypothetical_spot: float, t_years: float) -> float:
    call_gex, put_gex = compute_gex(calls, puts, hypothetical_spot, t_years)
    return call_gex + put_gex


def compute_zero_gamma(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, t_years: float) -> float:
    """Find the underlying price where net GEX crosses zero, by scanning a
    range around spot and linearly interpolating the sign change closest to spot."""
    lo, hi = spot * 0.85, spot * 1.15
    grid = np.linspace(lo, hi, 121)
    net = np.array([net_gex_at_spot(calls, puts, s, t_years) for s in grid])

    sign_changes = np.where(np.diff(np.sign(net)) != 0)[0]
    if len(sign_changes) == 0:
        return float("nan")

    # pick the crossing nearest to current spot
    crossing_idx = sign_changes[np.argmin(np.abs(grid[sign_changes] - spot))]
    x0, x1 = grid[crossing_idx], grid[crossing_idx + 1]
    y0, y1 = net[crossing_idx], net[crossing_idx + 1]
    zero_gamma = x0 - y0 * (x1 - x0) / (y1 - y0)
    return float(zero_gamma)


def atm_strike(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    strikes = np.array(sorted(set(calls["strike"]).union(set(puts["strike"]))))
    return float(strikes[np.argmin(np.abs(strikes - spot))])


def mid_price(row: pd.Series) -> float:
    bid, ask, last = row.get("bid", np.nan), row.get("ask", np.nan), row.get("lastPrice", np.nan)
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return float(last) if last else 0.0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SPX options/gamma dashboard via Yahoo Finance (SPY proxy).")
    parser.add_argument("--ticker", default="SPY", help="Underlying ETF with a liquid options chain (default: SPY)")
    parser.add_argument("--index", default="^GSPC", help="Cash index ticker used for the SPX ratio (default: ^GSPC)")
    parser.add_argument("--expiry", default=None, help="Expiry date YYYY-MM-DD (default: nearest available)")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"Fetching {ticker} data from Yahoo Finance...", file=sys.stderr)
    levels = fetch_daily_levels(ticker)
    spot, prev_close = levels["spot"], levels["prev_close"]
    rsi = fetch_rsi(ticker)
    vwap = fetch_vwap(ticker)
    spx_spot = fetch_last_close(args.index)
    ratio = spx_spot / spot
    vix = fetch_last_close("^VIX")

    t = yf.Ticker(ticker)
    expiry = pick_expiry(t, args.expiry)
    chain = t.option_chain(expiry)
    calls, puts = chain.calls.copy(), chain.puts.copy()
    calls["openInterest"] = calls["openInterest"].fillna(0)
    puts["openInterest"] = puts["openInterest"].fillna(0)
    calls["volume"] = calls["volume"].fillna(0)
    puts["volume"] = puts["volume"].fillna(0)

    t_years = years_to_expiry(expiry)

    call_vol, put_vol = calls["volume"].sum(), puts["volume"].sum()
    call_oi, put_oi = calls["openInterest"].sum(), puts["openInterest"].sum()
    pc_volume = put_vol / call_vol if call_vol else float("nan")
    pc_oi = put_oi / call_oi if call_oi else float("nan")

    max_pain = compute_max_pain(calls, puts)

    call_wall = float(calls.loc[calls["openInterest"].idxmax(), "strike"]) if not calls.empty else float("nan")
    put_wall = float(puts.loc[puts["openInterest"].idxmax(), "strike"]) if not puts.empty else float("nan")

    strike_atm = atm_strike(calls, puts, spot)
    call_row = calls.loc[(calls["strike"] - strike_atm).abs().idxmin()]
    put_row = puts.loc[(puts["strike"] - strike_atm).abs().idxmin()]
    atm_iv = np.nanmean([call_row.get("impliedVolatility", np.nan), put_row.get("impliedVolatility", np.nan)])
    exp_move = mid_price(call_row) + mid_price(put_row)

    call_gex, put_gex = compute_gex(calls, puts, spot, t_years)
    net_gex = call_gex + put_gex
    gross_gex = call_gex + abs(put_gex)
    zero_gamma = compute_zero_gamma(calls, puts, spot, t_years)

    gap_pct = (levels["open"] - prev_close) / prev_close * 100
    vwap_vs_spot_pct = (spot - vwap) / vwap * 100
    flat_x10_diff = spx_spot - spot * 10
    iv_vs_vix = "OK" if abs(atm_iv * 100 - vix) <= 5 else "DIVERGENT"

    def to_spx(px):
        return px * ratio, px * 10

    print()
    print(f"=== {ticker} ===")
    print(f"    Spot            {spot:.2f}")
    print(f"    Prev close      {prev_close:.2f}    PDH {levels['pdh']:.2f} / PDL {levels['pdl']:.2f}")
    print(f"    Gap             {gap_pct:+.2f}%")
    print(f"    RSI(14)         {rsi:.1f}")
    print(f"    VWAP            {vwap:.2f}  ({vwap_vs_spot_pct:+.2f}% vs spot)")
    print(f"    SPX ratio       {ratio:.4f}  (flat x10 is off by {flat_x10_diff:+.1f} pts)")
    print()
    print(f"=== CHAIN {expiry} ===")
    print(f"    {len(calls)} calls / {len(puts)} puts")
    print(f"    P/C volume      {pc_volume:.3f}")
    print(f"    P/C OI          {pc_oi:.3f}   <- these measure different things")
    print(f"    Max pain        ${max_pain:.0f}  ({(max_pain - spot) / spot * 100:+.2f}% from spot)")
    print(f"    Call wall       ${call_wall:.0f}")
    print(f"    Put wall        ${put_wall:.0f}")
    print(f"    Exp move        +/-${exp_move:.2f}  (${spot - exp_move:.2f} - ${spot + exp_move:.2f})")
    print()
    print(f"    ATM IV          {atm_iv * 100:.1f}%  vs VIX {vix:.1f}   [{iv_vs_vix}]")
    print(f"    Call GEX        ${call_gex / 1e6:+.1f}M")
    print(f"    Put GEX         ${put_gex / 1e6:+.1f}M")
    print(f"    Net GEX         ${net_gex / 1e6:+.1f}M  ({'long' if net_gex >= 0 else 'short'} gamma)")
    print(f"    Gross GEX       ${gross_gex / 1e6:.1f}M  <- not the same as net")
    zg_str = f"${zero_gamma:.2f}" if not np.isnan(zero_gamma) else "n/a (no sign change in scanned range)"
    print(f"    Zero gamma      {zg_str}")
    print()
    print(f"=== SPX EQUIVALENTS (ratio {ratio:.4f}) ===")
    for label, px in [
        ("spot", spot),
        ("max pain", max_pain),
        ("call wall", call_wall),
        ("put wall", put_wall),
    ]:
        spx_val, x10_val = to_spx(px)
        print(f"    {label:<10} ${px:>8.2f} -> {spx_val:>9,.1f}  (x10: {x10_val:>9,.1f})")
    if not np.isnan(zero_gamma):
        spx_val, x10_val = to_spx(zero_gamma)
        print(f"    {'zero gamma':<10} ${zero_gamma:>8.2f} -> {spx_val:>9,.1f}  (x10: {x10_val:>9,.1f})")
    print()


if __name__ == "__main__":
    main()
