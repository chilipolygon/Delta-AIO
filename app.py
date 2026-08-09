#!/usr/bin/env python3
"""
Localhost web dashboard for the SPX options/gamma report.

    pip install -r requirements.txt
    python3 app.py
    -> http://127.0.0.1:5000

All analytics come from spx_dashboard.build_report(); this module only adds
an HTTP layer, a short TTL cache so a page refresh doesn't hammer Yahoo, and
the static front end in templates/.
"""

import argparse
import math
import threading
from datetime import datetime
import time
import traceback

import numpy as np
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from backtest import run_backtest
from contracts import pick_contract
from portfolio import Portfolio, from_backtest, safe_name
from index_signal import INDEX_PROXY, index_signal
from options_flow import exposure_profile, profile_payload
from regime import classify_regime
from scanner import SCANS, load_universe, scan_universe
from spx_dashboard import NY_TZ, build_report

app = Flask(__name__)

CACHE_TTL_SECONDS = 60
SCAN_TTL_SECONDS = 900    # a full-universe scan is expensive; 15 minutes is plenty
DETAIL_TTL_SECONDS = 300  # one option chain per click, so cache it for 5 minutes
INDEX_TTL_SECONDS = 120   # the index panel is one chain, refreshed often while live
BACKTEST_TTL_SECONDS = 3600   # a historical result cannot change; cache it hard


def _parse_asof(raw: str | None):
    """Return (date|None, error|None). Future dates are rejected -- scanning
    'as of' tomorrow would silently just be a live scan wearing a date."""
    if not raw:
        return None, None
    try:
        d = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None, f"asof must be YYYY-MM-DD, got {raw!r}"
    if d >= datetime.now(tz=NY_TZ).date():
        return None, f"asof must be a past date (got {d})"
    return d, None


def market_status() -> dict:
    """Rough US equity session state, used to pause auto-refresh out of hours.
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


_cache: dict[tuple, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def json_safe(obj):
    """Make a payload JSON-serializable.

    Two distinct hazards, both of which reach the browser as an opaque error:

    1. json.dumps writes bare NaN/Infinity, which are not valid JSON and make
       JSON.parse throw. A thin chain legitimately produces NaN for RSI, ATM IV
       or the P/C ratios.
    2. numpy scalars (bool_, int64, float64) are not serializable at all and
       raise a 500. They leak out of pandas comparisons all over this codebase
       -- `a < b` on numpy floats is a numpy bool, not a Python bool -- so this
       is caught generically here rather than at each call site.
    """
    if isinstance(obj, np.generic):        # numpy scalar -> nearest Python type
        obj = obj.item()
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


@app.errorhandler(Exception)
def api_errors_stay_json(exc):
    """Never let an API route answer with Flask's HTML error page -- the front
    end parses these as JSON, and an HTML body turns a clear server-side error
    into an opaque 'Unexpected token <' in the browser."""
    code = exc.code if isinstance(exc, HTTPException) else 500
    if request.path.startswith("/api/"):
        if not isinstance(exc, HTTPException):
            app.logger.error("unhandled on %s: %s", request.path, traceback.format_exc())
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), code
    return exc if isinstance(exc, HTTPException) else ("Internal Server Error", 500)


def cached_report(ticker: str, index: str, expiry: str | None) -> dict:
    key = (ticker.upper(), index.upper(), expiry)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]

    report = json_safe(build_report(ticker, index, expiry))

    with _cache_lock:
        _cache[key] = (time.time(), report)
    return report


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scanner")
def scanner_page():
    # the live page fetches /api/scan; only the static export bakes data in
    return render_template("scanner.html", embedded="null")


@app.route("/api/setup/<ticker>")
def api_setup(ticker: str):
    """Full detail for one name: the setup, its confluence, dealer positioning
    and the regime read. This is the only path that fetches an option chain."""
    ticker = ticker.upper()
    key = ("setup", ticker)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < DETAIL_TTL_SECONDS:
            return jsonify(hit[1])

    setup = _find_setup(ticker)
    if setup is None:
        return jsonify({"error": f"{ticker} is not in the current scan results"}), 404

    payload = {"setup": setup, "ticker": ticker, "options": None, "regime": None}
    conf = setup.get("confluence") or {}

    prof = exposure_profile(ticker, spot=setup.get("price"))
    if prof is None:
        payload["options_error"] = (
            f"Yahoo returned no option chain for {ticker}; the regime read needs one.")
    else:
        levels = prof["levels"]
        payload["options"] = {
            "spot": prof["spot"], "expiries": prof["expiries"], "atm_iv": prof["atm_iv"],
            "levels": levels, "profile": profile_payload(prof),
        }
        if conf:
            payload["regime"] = classify_regime(prof["spot"], levels, conf)

    # the chain is already being fetched here, so this is where a concrete
    # contract can be resolved without a second round trip
    try:
        payload["contract"] = pick_contract(
            ticker, setup,
            min_dte=request.args.get("min_dte", 30, type=int),
            moneyness=request.args.get("moneyness", "atm"),
            spot=setup.get("price"))
    except Exception as exc:  # noqa: BLE001
        app.logger.error("contract pick failed for %s: %s", ticker, exc)
        payload["contract"] = None

    payload = json_safe(payload)
    with _cache_lock:
        _cache[key] = (time.time(), payload)
    return jsonify(payload)


def _find_setup(ticker: str) -> dict | None:
    """Pull a ticker's best setup out of whatever scan is already cached."""
    best = None
    with _cache_lock:
        entries = [v for k, v in _cache.items() if k[0] == "scan"]
    for _, result in entries:
        for s in result.get("setups", []):
            if s["ticker"] == ticker and (best is None or s["score"] > best["score"]):
                best = s
    return best


@app.route("/api/scan")
def api_scan():
    types = [t for t in request.args.get("types", "ote,ma,breakout").split(",") if t in SCANS]
    limit = min(max(request.args.get("limit", 60, type=int), 1), 300)
    asof, err = _parse_asof(request.args.get("asof"))
    if err:
        return jsonify({"error": err}), 400

    key = ("scan", tuple(types), limit, asof)
    now = time.time()
    ttl = BACKTEST_TTL_SECONDS if asof else SCAN_TTL_SECONDS
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return jsonify(hit[1])

    result = json_safe(scan_universe(load_universe(), types, limit=limit, asof=asof))
    result["market"] = market_status()

    with _cache_lock:
        _cache[key] = (time.time(), result)
    return jsonify(result)


@app.route("/api/expiries")
def api_expiries():
    ticker = request.args.get("ticker", "SPY").upper()
    try:
        return jsonify({"ticker": ticker, "expiries": list(yf.Ticker(ticker).options)})
    except Exception as exc:
        app.logger.error("expiries failed: %s", traceback.format_exc())
        return jsonify({"error": str(exc)}), 502


@app.route("/api/report")
def api_report():
    ticker = request.args.get("ticker", "SPY")
    index = request.args.get("index", "^GSPC")
    expiry = request.args.get("expiry") or None
    try:
        return jsonify(cached_report(ticker, index, expiry))
    except Exception as exc:
        app.logger.error("report failed: %s", traceback.format_exc())
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502


@app.route("/api/market")
def api_market():
    return jsonify(market_status())


@app.route("/api/index")
def api_index():
    index = request.args.get("index", "^GSPC").upper()
    proxy = request.args.get("proxy") or INDEX_PROXY.get(index)
    types = [t for t in request.args.get("types", "ote,ma,breakout").split(",") if t in SCANS]
    asof, err = _parse_asof(request.args.get("asof"))
    if err:
        return jsonify({"error": err}), 400

    key = ("index", index, proxy, tuple(types), asof)
    now = time.time()
    ttl = BACKTEST_TTL_SECONDS if asof else INDEX_TTL_SECONDS
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return jsonify(hit[1])

    payload = json_safe(index_signal(index, proxy, types, asof))
    payload["market"] = market_status()
    with _cache_lock:
        _cache[key] = (time.time(), payload)
    return jsonify(payload)


@app.route("/api/backtest")
def api_backtest():
    asof, err = _parse_asof(request.args.get("asof"))
    if err or asof is None:
        return jsonify({"error": err or "asof is required for a backtest"}), 400
    forward = min(max(request.args.get("forward", 30, type=int), 1), 250)
    types = [t for t in request.args.get("types", "ote,ma,breakout").split(",") if t in SCANS]
    limit = min(max(request.args.get("limit", 200, type=int), 1), 500)

    key = ("backtest", asof, forward, tuple(types), limit)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < BACKTEST_TTL_SECONDS:
            return jsonify(hit[1])

    result = json_safe(run_backtest(load_universe(), types, asof, forward, limit))
    with _cache_lock:
        _cache[key] = (time.time(), result)
    return jsonify(result)


# --------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------

@app.route("/portfolio")
def portfolio_page():
    return render_template("portfolio.html")


@app.route("/api/portfolios")
def api_portfolios():
    return jsonify({"portfolios": Portfolio.list_all()})


@app.route("/api/portfolio/<name>")
def api_portfolio(name: str):
    try:
        p = Portfolio.load(safe_name(name))
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404

    # mark open positions against the latest close we can cheaply get
    tickers = sorted({pos.ticker for pos in p.positions})
    prices: dict[str, float] = {}
    if tickers and request.args.get("mark", "1") != "0":
        try:
            raw = yf.download(tickers, period="5d", interval="1d", group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
            for t in tickers:
                try:
                    sub = raw[t] if len(tickers) > 1 else raw
                    prices[t] = float(sub["Close"].dropna().iloc[-1])
                except Exception:  # noqa: BLE001
                    continue
        except Exception as exc:  # noqa: BLE001
            app.logger.error("mark-to-market fetch failed: %s", exc)
    p.mark_to_market(prices)

    return jsonify(json_safe({**p.to_dict(), "stats": p.stats(),
                              "marked": sorted(prices), "unmarked": sorted(set(tickers) - set(prices))}))


@app.route("/api/portfolio", methods=["POST"])
def api_portfolio_create():
    body = request.get_json(silent=True) or {}
    try:
        p = Portfolio.create(body.get("name", ""), float(body.get("cash", 0)),
                             float(body.get("risk_pct", 1.0)),
                             overwrite=bool(body.get("overwrite")))
    except (ValueError, FileExistsError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(json_safe({**p.to_dict(), "stats": p.stats()}))


@app.route("/api/portfolio/<name>/open", methods=["POST"])
def api_portfolio_open(name: str):
    body = request.get_json(silent=True) or {}
    try:
        p = Portfolio.load(safe_name(name))
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404

    ticker = (body.get("ticker") or "").upper()
    setup = body.get("setup") or _find_setup(ticker)
    if not setup:
        return jsonify({"error": f"no setup for {ticker} in the current scan"}), 404

    kind = body.get("kind", "stock")
    contract = body.get("contract")
    if kind == "option" and not contract:
        contract = pick_contract(ticker, setup, min_dte=int(body.get("min_dte", 30)),
                                 moneyness=body.get("moneyness", "atm"),
                                 spot=setup.get("price"))
        if not contract:
            return jsonify({"error": f"no option chain available for {ticker}"}), 400
    try:
        pos = p.open_from_setup(setup, kind=kind, contract=contract,
                                qty=body.get("qty"), source=body.get("source", "scanner"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    p.save()
    return jsonify(json_safe({"position": asdict_pos(pos), "stats": p.stats()}))


@app.route("/api/portfolio/<name>/close", methods=["POST"])
def api_portfolio_close(name: str):
    body = request.get_json(silent=True) or {}
    try:
        p = Portfolio.load(safe_name(name))
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 404
    try:
        pos = p.close(body["id"], float(body["price"]), body.get("reason", "manual"))
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    p.save()
    return jsonify(json_safe({"closed": asdict_pos(pos), "stats": p.stats()}))


@app.route("/api/portfolio/from-backtest", methods=["POST"])
def api_portfolio_from_backtest():
    """Replay a backtest into a saved portfolio."""
    body = request.get_json(silent=True) or {}
    asof, err = _parse_asof(body.get("asof"))
    if err or asof is None:
        return jsonify({"error": err or "asof is required"}), 400
    forward = min(max(int(body.get("forward", 30)), 1), 250)
    types = [t for t in (body.get("types") or ["ote", "ma", "breakout"]) if t in SCANS]
    try:
        name = safe_name(body.get("name", ""))
        cash = float(body.get("cash", 0))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    bt = json_safe(run_backtest(load_universe(), types, asof, forward,
                                int(body.get("limit", 200))))
    try:
        p = from_backtest(bt, name, cash, float(body.get("risk_pct", 1.0)),
                          kind=body.get("kind", "stock"),
                          overwrite=bool(body.get("overwrite", True)))
    except (ValueError, FileExistsError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(json_safe({**p.to_dict(), "stats": p.stats(),
                              "backtest_summary": bt["summary"]}))


def asdict_pos(pos):
    from dataclasses import asdict as _asdict
    return {**_asdict(pos), "cost_basis": pos.cost_basis, "multiplier": pos.multiplier}



def main():
    parser = argparse.ArgumentParser(description="Serve the SPX gamma dashboard on localhost.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug reloader")
    args = parser.parse_args()

    print(f"SPX dashboard -> http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
