#!/usr/bin/env python3
"""
Index-level signal for SPX (and NDX), built from the same rules as the stock
scanner plus dealer positioning.

Yahoo publishes no option chain for ^SPX / ^GSPC, so positioning is read from
the liquid proxy ETF (SPY for SPX, QQQ for NDX) and converted to index points
with the live index/ETF ratio -- the same approach spx_dashboard already uses.
Strikes convert cleanly; dollar exposures do not, so GEX/VEX magnitudes stay in
the proxy's own dollars and are labelled as such.

    python3 index_signal.py
    python3 index_signal.py --index ^NDX --proxy QQQ
    python3 index_signal.py --asof 2026-03-16      # price/tape only, see below
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from options_flow import exposure_profile, profile_payload
from regime import classify_regime
from scanner import SCANS, Setup, classify, score, split_asof
from signals import confluence

INDEX_PROXY = {"^GSPC": "SPY", "^SPX": "SPY", "^NDX": "QQQ", "^DJI": "DIA", "^RUT": "IWM"}
INDEX_NAME = {"^GSPC": "S&P 500", "^SPX": "S&P 500", "^NDX": "Nasdaq-100",
              "^DJI": "Dow Jones", "^RUT": "Russell 2000"}


def _history(ticker: str, asof: date | None, lookback_days: int = 500) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if asof is None:
        df = t.history(period="2y", interval="1d", auto_adjust=False)
    else:
        df = t.history(start=(asof - timedelta(days=lookback_days)).isoformat(),
                       end=(asof + timedelta(days=5)).isoformat(),
                       interval="1d", auto_adjust=False)
    return df.dropna(subset=["Close"])


def index_signal(index: str = "^GSPC", proxy: str | None = None,
                 types: list[str] | None = None, asof: date | None = None) -> dict:
    """Run every scan rule on the index itself, then layer positioning on top."""
    index = index.upper()
    proxy = (proxy or INDEX_PROXY.get(index, "SPY")).upper()
    types = types or list(SCANS)

    full = _history(index, asof)
    df, _ = split_asof(full, asof)
    if len(df) < 80:
        return {"error": f"not enough history for {index} (got {len(df)} bars)"}

    spot = float(df["Close"].iloc[-1])
    conf = confluence(df)

    setups = []
    for kind in types:
        try:
            s = SCANS[kind](df)
        except Exception:  # noqa: BLE001
            continue
        if s is None or not s.targets:
            continue
        s.ticker, s.name = index, INDEX_NAME.get(index, index)
        s.confluence, s.bias = conf, float(conf.get("bias", 0.0))
        classify(s)
        s.score = score(s)
        setups.append(s)
    setups.sort(key=lambda s: -s.score)

    out = {
        "index": index, "proxy": proxy, "name": INDEX_NAME.get(index, index),
        "asof": asof.isoformat() if asof else None,
        "spot": round(spot, 2),
        "as_of_bar": str(df.index[-1].date()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "confluence": conf,
        "setups": [s.__dict__ for s in setups],
        "options": None, "regime": None,
    }

    if asof is not None:
        # Yahoo only ever serves the CURRENT chain, so positioning as of a past
        # date is not retrievable. Say so rather than pinning today's dealer
        # book onto a historical bar, which would silently be a lie.
        out["options_error"] = (
            "Dealer positioning is not available for a past date — Yahoo serves only the "
            "current option chain. The price rules and tape read above are historical; "
            "GEX/VEX and the regime verdict are omitted rather than back-filled with today's chain.")
        return out

    prof = exposure_profile(proxy)
    if prof is None:
        out["options_error"] = f"No option chain returned for the {proxy} proxy."
        return out

    ratio = spot / prof["spot"] if prof["spot"] else 1.0
    levels = prof["levels"]
    out["ratio"] = round(ratio, 4)
    out["proxy_spot"] = prof["spot"]
    out["options"] = {
        "spot": prof["spot"], "expiries": prof["expiries"], "atm_iv": prof["atm_iv"],
        "levels": levels, "profile": profile_payload(prof),
        "index_levels": _to_index(levels, ratio),
        "ratio": round(ratio, 4),
    }
    # regime is judged on the proxy's own scale, where the strikes actually live
    out["regime"] = classify_regime(prof["spot"], levels, conf)
    return out


def _to_index(levels: dict, ratio: float) -> dict:
    """Strike levels in index points. Exposures are NOT converted -- they are
    proxy-chain dollars and have no meaningful index-point equivalent."""
    def conv(v):
        if v is None:
            return None
        if isinstance(v, dict) and "strike" in v:
            return {**v, "strike": round(v["strike"] * ratio, 1)}
        return v

    out = {k: conv(levels.get(k)) for k in
           ("pin", "floor", "ceiling", "fuel_down", "fuel_up", "target_down",
            "target_up", "vanna_node")}
    out["flip"] = round(levels["flip"] * ratio, 1) if levels.get("flip") else None
    out["flip_cushion"] = (round(levels["flip_cushion"] * ratio, 1)
                           if levels.get("flip_cushion") else None)
    return out


def print_signal(sig: dict) -> None:
    if "error" in sig:
        print(f"error: {sig['error']}", file=sys.stderr)
        return
    print(f"\n=== {sig['index']} ({sig['name']}) {sig['spot']:,.2f} · bar {sig['as_of_bar']} ===")
    c = sig["confluence"]
    print(f"    bias {c['bias']:+.0f} · {c['agree']} agree / {c['disagree']} disagree")
    for k in ("macd", "rsi", "sma200", "trend", "volume", "vol"):
        print(f"      {c[k]['phrase']}")

    if sig.get("regime"):
        r, o = sig["regime"], sig["options"]
        print(f"\n    {r['verdict']} — {r['regime']} · {r['regime_sub']}")
        print(f"    conviction {r['conviction']}/100 ({r['conviction_band']})")
        print(f"    proxy {sig['proxy']} {sig['proxy_spot']} · ratio {sig['ratio']} · "
              f"ATM IV {o['atm_iv']}%")
        idx = o["index_levels"]
        print(f"    flip      {idx['flip']}   (proxy {o['levels']['flip']})")
        for k in ("pin", "floor", "ceiling", "fuel_down", "fuel_up"):
            if idx.get(k):
                print(f"    {k:<9} {idx[k]['strike']}   (proxy {o['levels'][k]['strike']})")
        print(f"\n    {r['narrative']['see']}\n")
        print(f"    PLAY: {r['narrative']['play']}")
        print(f"    INVALID IF: {r['narrative']['invalid']}")
    elif sig.get("options_error"):
        print(f"\n    [no positioning] {sig['options_error']}")

    print(f"\n    {len(sig['setups'])} rule hit(s):")
    for s in sig["setups"]:
        print(f"      {s['setup_type']:<9} {s['status']:<18} entry {s['entry']:,.2f} "
              f"stop {s['stop']:,.2f} T1 {s['targets'][0]:,.2f} "
              f"R:R {s['entry_rr']:.1f}:1 score {s['score']}")
    print()


def main():
    p = argparse.ArgumentParser(description="Index-level setup + positioning signal.")
    p.add_argument("--index", default="^GSPC")
    p.add_argument("--proxy", default=None, help="options proxy ETF (default: SPY/QQQ by index)")
    p.add_argument("--types", default="ote,ma,breakout")
    p.add_argument("--asof", default=None, help="run as of a past date, YYYY-MM-DD (price/tape only)")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args()

    asof = None
    if args.asof:
        try:
            asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
        except ValueError:
            p.error("--asof must be YYYY-MM-DD")

    types = [t.strip() for t in args.types.split(",") if t.strip() in SCANS]
    sig = index_signal(args.index, args.proxy, types, asof)
    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(sig, indent=1, default=str))
        print(f"wrote {args.json}", file=sys.stderr)
    print_signal(sig)


if __name__ == "__main__":
    main()
