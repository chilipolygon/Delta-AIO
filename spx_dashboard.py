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


def market_status() -> dict:
    """Rough US equity session state, used to pause the dashboard's auto-refresh
    and the watch bot's polling out of hours.
    Weekends and clock only -- market holidays are not tracked."""
    now = datetime.now(tz=NY_TZ)
    minutes = now.hour * 60 + now.minute
    weekday = now.weekday() < 5
    open_m, close_m = 9 * 60 + 30, 16 * 60
    if not weekday:
        state = "weekend"
    elif minutes < 4 * 60:
        state = "closed"
    elif minutes < open_m:
        state = "premarket"
    elif minutes < close_m:
        state = "open"
    elif minutes < 20 * 60:
        state = "afterhours"
    else:
        state = "closed"
    return {"state": state, "is_open": state == "open",
            "now": now.isoformat(timespec="seconds"),
            "note": "holidays are not tracked"}


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


def gex_by_strike(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, t_years: float):
    """Per-strike dollar gamma exposure per 1% move, in $.

    Returns (strikes, call_gex_series, put_gex_series) aligned on the union of
    strikes. Convention: dealers assumed long calls / short puts against customer
    flow, so put gamma exposure is booked as negative.
    """
    strikes = np.array(sorted(set(calls["strike"]).union(set(puts["strike"]))))
    contract_mult = 100
    scale = spot ** 2 * 0.01  # dollar change per 1% move

    def leg(df, sign):
        oi = df.set_index("strike")["openInterest"].reindex(strikes).fillna(0).to_numpy()
        iv = df.set_index("strike")["impliedVolatility"].reindex(strikes).to_numpy()
        gamma = bs_gamma(spot, strikes, iv, t_years)
        return sign * gamma * oi * contract_mult * scale

    return strikes, leg(calls, 1.0), leg(puts, -1.0)


def compute_gex(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, t_years: float):
    """Aggregate dollar gamma exposure per 1% move in the underlying, in $."""
    _, call_leg, put_leg = gex_by_strike(calls, puts, spot, t_years)
    return float(call_leg.sum()), float(put_leg.sum())


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
# Report builder -- shared by the CLI and the web dashboard
# --------------------------------------------------------------------------

def build_report(ticker: str = "SPY", index: str = "^GSPC", expiry: str | None = None) -> dict:
    """Fetch everything and return a plain JSON-serializable dict of results."""
    ticker = ticker.upper()

    levels = fetch_daily_levels(ticker)
    spot, prev_close = levels["spot"], levels["prev_close"]
    rsi = fetch_rsi(ticker)
    vwap = fetch_vwap(ticker)
    spx_spot = fetch_last_close(index)
    ratio = spx_spot / spot
    vix = fetch_last_close("^VIX")

    t = yf.Ticker(ticker)
    expiry = pick_expiry(t, expiry)
    chain = t.option_chain(expiry)
    calls, puts = chain.calls.copy(), chain.puts.copy()
    for df in (calls, puts):
        df["openInterest"] = df["openInterest"].fillna(0)
        df["volume"] = df["volume"].fillna(0)

    t_years = years_to_expiry(expiry)

    call_vol, put_vol = calls["volume"].sum(), puts["volume"].sum()
    call_oi_total, put_oi_total = calls["openInterest"].sum(), puts["openInterest"].sum()

    max_pain = compute_max_pain(calls, puts)
    call_wall = float(calls.loc[calls["openInterest"].idxmax(), "strike"]) if not calls.empty else float("nan")
    put_wall = float(puts.loc[puts["openInterest"].idxmax(), "strike"]) if not puts.empty else float("nan")

    strike_atm = atm_strike(calls, puts, spot)
    call_row = calls.loc[(calls["strike"] - strike_atm).abs().idxmin()]
    put_row = puts.loc[(puts["strike"] - strike_atm).abs().idxmin()]
    atm_iv = float(np.nanmean([call_row.get("impliedVolatility", np.nan),
                               put_row.get("impliedVolatility", np.nan)]))
    exp_move = mid_price(call_row) + mid_price(put_row)

    strikes, call_leg, put_leg = gex_by_strike(calls, puts, spot, t_years)
    call_gex, put_gex = float(call_leg.sum()), float(put_leg.sum())
    net_gex = call_gex + put_gex
    zero_gamma = compute_zero_gamma(calls, puts, spot, t_years)

    call_oi_s = calls.set_index("strike")["openInterest"].reindex(strikes).fillna(0)
    put_oi_s = puts.set_index("strike")["openInterest"].reindex(strikes).fillna(0)

    def nan_to_none(x):
        x = float(x)
        return None if np.isnan(x) else x

    return {
        "ticker": ticker,
        "index": index,
        "expiry": expiry,
        "as_of": datetime.now(tz=NY_TZ).isoformat(timespec="seconds"),
        "t_years": t_years,
        "underlying": {
            "spot": spot,
            "prev_close": prev_close,
            "open": levels["open"],
            "pdh": levels["pdh"],
            "pdl": levels["pdl"],
            "gap_pct": (levels["open"] - prev_close) / prev_close * 100,
            "rsi": rsi,
            "vwap": vwap,
            "vwap_vs_spot_pct": (spot - vwap) / vwap * 100,
            "vix": vix,
        },
        "spx": {
            "spot": spx_spot,
            "ratio": ratio,
            "flat_x10_diff": spx_spot - spot * 10,
        },
        "chain": {
            "n_calls": int(len(calls)),
            "n_puts": int(len(puts)),
            "call_volume": float(call_vol),
            "put_volume": float(put_vol),
            "call_oi": float(call_oi_total),
            "put_oi": float(put_oi_total),
            "pc_volume": float(put_vol / call_vol) if call_vol else None,
            "pc_oi": float(put_oi_total / call_oi_total) if call_oi_total else None,
            "max_pain": max_pain,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "exp_move": exp_move,
            "atm_strike": strike_atm,
            "atm_iv": atm_iv,
        },
        "gex": {
            "call": call_gex,
            "put": put_gex,
            "net": net_gex,
            "gross": call_gex + abs(put_gex),
            "zero_gamma": nan_to_none(zero_gamma),
        },
        "by_strike": {
            "strikes": [float(s) for s in strikes],
            "call_gex": [float(v) for v in call_leg],
            "put_gex": [float(v) for v in put_leg],
            "net_gex": [float(v) for v in (call_leg + put_leg)],
            "call_oi": [float(v) for v in call_oi_s],
            "put_oi": [float(v) for v in put_oi_s],
        },
    }


def print_report(r: dict) -> None:
    ticker = r["ticker"]
    u, s, c, g = r["underlying"], r["spx"], r["chain"], r["gex"]
    spot, ratio = u["spot"], s["ratio"]
    atm_iv, vix = c["atm_iv"], u["vix"]
    zero_gamma = g["zero_gamma"]
    exp_move = c["exp_move"]
    iv_vs_vix = "OK" if abs(atm_iv * 100 - vix) <= 5 else "DIVERGENT"

    def to_spx(px):
        return px * ratio, px * 10

    print()
    print(f"=== {ticker} ===")
    print(f"    Spot            {spot:.2f}")
    print(f"    Prev close      {u['prev_close']:.2f}    PDH {u['pdh']:.2f} / PDL {u['pdl']:.2f}")
    print(f"    Gap             {u['gap_pct']:+.2f}%")
    print(f"    RSI(14)         {u['rsi']:.1f}")
    print(f"    VWAP            {u['vwap']:.2f}  ({u['vwap_vs_spot_pct']:+.2f}% vs spot)")
    print(f"    SPX ratio       {ratio:.4f}  (flat x10 is off by {s['flat_x10_diff']:+.1f} pts)")
    print()
    print(f"=== CHAIN {r['expiry']} ===")
    print(f"    {c['n_calls']} calls / {c['n_puts']} puts")
    print(f"    P/C volume      {c['pc_volume']:.3f}")
    print(f"    P/C OI          {c['pc_oi']:.3f}   <- these measure different things")
    print(f"    Max pain        ${c['max_pain']:.0f}  ({(c['max_pain'] - spot) / spot * 100:+.2f}% from spot)")
    print(f"    Call wall       ${c['call_wall']:.0f}")
    print(f"    Put wall        ${c['put_wall']:.0f}")
    print(f"    Exp move        +/-${exp_move:.2f}  (${spot - exp_move:.2f} - ${spot + exp_move:.2f})")
    print()
    print(f"    ATM IV          {atm_iv * 100:.1f}%  vs VIX {vix:.1f}   [{iv_vs_vix}]")
    print(f"    Call GEX        ${g['call'] / 1e6:+.1f}M")
    print(f"    Put GEX         ${g['put'] / 1e6:+.1f}M")
    print(f"    Net GEX         ${g['net'] / 1e6:+.1f}M  ({'long' if g['net'] >= 0 else 'short'} gamma)")
    print(f"    Gross GEX       ${g['gross'] / 1e6:.1f}M  <- not the same as net")
    zg_str = f"${zero_gamma:.2f}" if zero_gamma is not None else "n/a (no sign change in scanned range)"
    print(f"    Zero gamma      {zg_str}")
    print()
    print(f"=== SPX EQUIVALENTS (ratio {ratio:.4f}) ===")
    rows = [("spot", spot), ("max pain", c["max_pain"]),
            ("call wall", c["call_wall"]), ("put wall", c["put_wall"])]
    if zero_gamma is not None:
        rows.append(("zero gamma", zero_gamma))
    for label, px in rows:
        spx_val, x10_val = to_spx(px)
        print(f"    {label:<10} ${px:>8.2f} -> {spx_val:>9,.1f}  (x10: {x10_val:>9,.1f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="SPX options/gamma dashboard via Yahoo Finance (SPY proxy).")
    parser.add_argument("--ticker", default="SPY", help="Underlying ETF with a liquid options chain (default: SPY)")
    parser.add_argument("--index", default="^GSPC", help="Cash index ticker used for the SPX ratio (default: ^GSPC)")
    parser.add_argument("--expiry", default=None, help="Expiry date YYYY-MM-DD (default: nearest available)")
    args = parser.parse_args()

    print(f"Fetching {args.ticker.upper()} data from Yahoo Finance...", file=sys.stderr)
    print_report(build_report(args.ticker, args.index, args.expiry))


if __name__ == "__main__":
    main()
