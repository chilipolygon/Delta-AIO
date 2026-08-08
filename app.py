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
import threading
import time
import traceback

import yfinance as yf
from flask import Flask, jsonify, render_template, request

from spx_dashboard import build_report

app = Flask(__name__)

CACHE_TTL_SECONDS = 60
_cache: dict[tuple, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


def cached_report(ticker: str, index: str, expiry: str | None) -> dict:
    key = (ticker.upper(), index.upper(), expiry)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]

    report = build_report(ticker, index, expiry)

    with _cache_lock:
        _cache[key] = (time.time(), report)
    return report


@app.route("/")
def index():
    return render_template("index.html")


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
