#!/usr/bin/env python3
"""
Backtest the scan rules by running them as of a past date and scoring each
setup against the bars that came after it.

    python3 backtest.py --asof 2026-03-16 --forward 30
    python3 backtest.py --asof 2026-01-05 --forward 45 --types ote --json bt.json

WHAT THIS DOES AND DOES NOT COVER
---------------------------------
Yahoo serves historical PRICE bars but NOT historical option chains -- only the
current one. So this backtests the price rules, the MACD/RSI/200 SMA/volume/
trend confluence, the status classification and the forward outcome. It cannot
backtest the GEX/VEX regime layer, because the dealer positioning that existed
on a past date is simply not retrievable from this data source. Any claim about
how the regime read would have performed historically would be fabricated, so
none is made.

Fills are assumed at the setup's own entry, and only when price actually traded
into the entry zone after the signal. A setup whose entry never filled is
reported as NO FILL rather than being scored as a win or a loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scanner import SCANS, fetch_history, load_universe, scan_universe


def evaluate(setup: dict, future: pd.DataFrame) -> dict:
    """Walk the bars after the signal and record what happened first.

    Order within a bar is unknowable from daily data, so when a bar's range
    spans both the stop and a target the stop is taken -- the pessimistic
    reading, chosen so results are not flattered by an ambiguity.
    """
    entry = setup["entry"]
    lo, hi = setup["entry_lo"], setup["entry_hi"]
    stop = setup["stop"]
    targets = list(setup["targets"])
    risk = entry - stop

    out = {"outcome": "NO FILL", "filled": False, "fill_bar": None, "bars_held": None,
           "targets_hit": 0, "exit": None, "r_multiple": None,
           "mfe_r": None, "mae_r": None, "forward_bars": int(len(future)),
           "fill_date": None, "exit_date": None}
    if future.empty or risk <= 0:
        return out

    highs = future["High"].to_numpy(dtype=float)
    lows = future["Low"].to_numpy(dtype=float)

    # 1) find the bar that trades into the entry zone
    fill_i = None
    for i in range(len(future)):
        if lows[i] <= hi and highs[i] >= lo:
            fill_i = i
            break
    if fill_i is None:
        return out

    out.update(filled=True, fill_bar=int(fill_i),
               fill_date=str(future.index[fill_i].date()))

    # 2) from the fill, walk forward to the first stop or target
    best_hi, worst_lo = entry, entry
    hit = 0
    for j in range(fill_i, len(future)):
        best_hi = max(best_hi, highs[j])
        worst_lo = min(worst_lo, lows[j])

        if lows[j] <= stop:
            out.update(outcome="STOPPED", exit=round(stop, 2), bars_held=int(j - fill_i),
                       targets_hit=hit, r_multiple=-1.0,
                       exit_date=str(future.index[j].date()))
            break
        while hit < len(targets) and highs[j] >= targets[hit]:
            hit += 1
        if hit >= len(targets):
            out.update(outcome="ALL TARGETS", exit=round(targets[-1], 2),
                       bars_held=int(j - fill_i), targets_hit=hit,
                       r_multiple=round((targets[-1] - entry) / risk, 2),
                       exit_date=str(future.index[j].date()))
            break
    else:
        last = float(future["Close"].iloc[-1])
        out.update(outcome=f"TARGET {hit}/{len(targets)}" if hit else "OPEN",
                   exit=round(last, 2), bars_held=int(len(future) - 1 - fill_i),
                   targets_hit=hit, r_multiple=round((last - entry) / risk, 2),
                   exit_date=str(future.index[-1].date()))

    out["mfe_r"] = round((best_hi - entry) / risk, 2)
    out["mae_r"] = round((worst_lo - entry) / risk, 2)
    return out


def summarize(rows: list[dict]) -> dict:
    filled = [r for r in rows if r["result"]["filled"]]
    scored = [r for r in filled if r["result"]["r_multiple"] is not None]
    rs = [r["result"]["r_multiple"] for r in scored]
    wins = [r for r in rs if r > 0]

    def by(key_fn):
        out: dict[str, dict] = {}
        for r in scored:
            k = key_fn(r)
            b = out.setdefault(k, {"n": 0, "wins": 0, "sum_r": 0.0})
            b["n"] += 1
            b["wins"] += 1 if r["result"]["r_multiple"] > 0 else 0
            b["sum_r"] += r["result"]["r_multiple"]
        for b in out.values():
            b["win_rate"] = round(100 * b["wins"] / b["n"], 1) if b["n"] else 0.0
            b["avg_r"] = round(b["sum_r"] / b["n"], 2) if b["n"] else 0.0
            b.pop("sum_r")
        return out

    return {
        "signals": len(rows),
        "filled": len(filled),
        "fill_rate": round(100 * len(filled) / len(rows), 1) if rows else 0.0,
        "scored": len(scored),
        "win_rate": round(100 * len(wins) / len(scored), 1) if scored else 0.0,
        "avg_r": round(float(np.mean(rs)), 2) if rs else 0.0,
        "median_r": round(float(np.median(rs)), 2) if rs else 0.0,
        "total_r": round(float(np.sum(rs)), 2) if rs else 0.0,
        "best_r": round(max(rs), 2) if rs else 0.0,
        "worst_r": round(min(rs), 2) if rs else 0.0,
        "stopped": sum(1 for r in scored if r["result"]["outcome"] == "STOPPED"),
        "all_targets": sum(1 for r in scored if r["result"]["outcome"] == "ALL TARGETS"),
        "by_rule": by(lambda r: r["setup"]["setup_type"]),
        "by_status": by(lambda r: r["setup"]["status"]),
    }


def run_backtest(universe: dict[str, str], types: list[str], asof: date,
                 forward_days: int = 30, limit: int = 200,
                 min_rr: float = 0.9) -> dict:
    futures: dict[str, pd.DataFrame] = {}
    scan = scan_universe(universe, types, limit=limit, min_rr=min_rr,
                         asof=asof, forward_days=forward_days, futures_out=futures)

    rows = []
    for s in scan["setups"]:
        fut = futures.get(s["ticker"])
        if fut is None:
            continue
        rows.append({"setup": s, "result": evaluate(s, fut.head(forward_days))})

    return {
        "asof": asof.isoformat(),
        "forward_days": forward_days,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "types": types,
        "scanned": scan["scanned"],
        "universe_size": scan["universe_size"],
        "summary": summarize(rows),
        "rows": rows,
        "note": ("Price rules and confluence only. Yahoo does not serve historical "
                 "option chains, so the GEX/VEX regime layer is not backtested."),
    }


def print_report(bt: dict) -> None:
    su = bt["summary"]
    print(f"\nBacktest as of {bt['asof']} · {bt['forward_days']} forward bars · "
          f"{', '.join(bt['types'])}")
    print(f"{bt['scanned']} names scanned · {su['signals']} signals · "
          f"{su['filled']} filled ({su['fill_rate']}%)\n")
    print(f"  win rate   {su['win_rate']}%   over {su['scored']} scored")
    print(f"  avg R      {su['avg_r']}      median {su['median_r']}")
    print(f"  total R    {su['total_r']}      best {su['best_r']} / worst {su['worst_r']}")
    print(f"  stopped    {su['stopped']}      all targets {su['all_targets']}\n")

    for label, book in (("BY RULE", su["by_rule"]), ("BY STATUS", su["by_status"])):
        print(f"  {label}")
        for k, b in sorted(book.items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {k:<18} n={b['n']:<4} win {b['win_rate']:>5}%   avg R {b['avg_r']:>6}")
        print()

    print(f"  {'TICKER':<8}{'RULE':<10}{'STATUS':<18}{'OUTCOME':<14}{'R':>7}{'BARS':>6}")
    print("  " + "-" * 63)
    for r in sorted(bt["rows"], key=lambda r: -(r["result"]["r_multiple"] or -99))[:25]:
        s, o = r["setup"], r["result"]
        rm = "—" if o["r_multiple"] is None else f"{o['r_multiple']:.2f}"
        bars = "—" if o["bars_held"] is None else o["bars_held"]
        print(f"  {s['ticker']:<8}{s['setup_type']:<10}{s['status']:<18}"
              f"{o['outcome']:<14}{rm:>7}{bars:>6}")
    print(f"\n  {bt['note']}\n")


def main():
    p = argparse.ArgumentParser(description="Backtest the scan rules from a past date.")
    p.add_argument("--asof", required=True, help="signal date, YYYY-MM-DD")
    p.add_argument("--forward", type=int, default=30, dest="forward_days",
                   help="bars after the signal to score against (default 30)")
    p.add_argument("--types", default="ote,ma,breakout")
    p.add_argument("--limit", type=int, default=200, help="max signals to score")
    p.add_argument("--min-rr", type=float, default=0.9, dest="min_rr")
    p.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    args = p.parse_args()

    try:
        asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
    except ValueError:
        p.error("--asof must be YYYY-MM-DD")
    if asof >= date.today():
        p.error(f"--asof must be in the past (got {asof}, today is {date.today()})")

    types = [t.strip() for t in args.types.split(",") if t.strip() in SCANS]
    if not types:
        p.error(f"--types must name at least one of {list(SCANS)}")

    universe = load_universe()
    print(f"universe: {len(universe)} tickers", file=sys.stderr)
    bt = run_backtest(universe, types, asof, args.forward_days, args.limit, args.min_rr)

    if args.json:
        Path(args.json).write_text(json.dumps(bt, indent=1, default=str))
        print(f"wrote {args.json}", file=sys.stderr)
    print_report(bt)


if __name__ == "__main__":
    main()
