#!/usr/bin/env python3
"""
Watch-list price bot: poll every name on the watch list and post to Discord
when one of them does something worth knowing about.

    export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."

    python3 watchbot.py add NIO --entry 4.58 --stop 4.30 --targets 5,5.5
    python3 watchbot.py add RGTI --auto        # levels from the scanner rules
    python3 watchbot.py import --index sp100   # take the scanner's best setups
    python3 watchbot.py list
    python3 watchbot.py run                    # poll forever, alert on changes
    python3 watchbot.py run --once --console   # one pass, printed, nothing sent

Four things get announced, and only when they change:

    approaching entry   price is within --near of the entry zone
    in entry zone       price is inside it -- the actionable one
    target hit          a new rung of the target ladder printed
    stopped out         price broke the stop

Repeats are deliberate but rare: a name that simply sits in its zone is
re-announced every --repeat-after minutes, while one-off events (a target, a
broken stop) are said once. State lives in data/watch_state.json, so a restart
picks up where it left off instead of replaying the day.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from spx_dashboard import market_status
from watchlist import (DEFAULT_NEAR_PCT, QUIET, Alert, Watch, WatchList,
                       evaluate, fmt)

STATE_PATH = Path(__file__).with_name("data") / "watch_state.json"

DEFAULT_INTERVAL = 60          # seconds between polls
DEFAULT_REPEAT_AFTER = 30      # minutes before a still-true state is repeated
BOT_NAME = "Talon"
MAX_EMBEDS = 10                # Discord's per-message cap
HTTP_TIMEOUT = 20

# Sessions the bot is willing to poll in, by --hours mode.
HOURS_MODES = {
    "open": {"open"},
    "extended": {"premarket", "open", "afterhours"},
    "always": {"open", "premarket", "afterhours", "closed", "weekend"},
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Quotes
# --------------------------------------------------------------------------

def _last_close(frame: pd.DataFrame) -> float | None:
    try:
        s = frame["Close"].dropna()
        return float(s.iloc[-1]) if len(s) else None
    except (KeyError, TypeError, IndexError):
        return None


def _download(tickers: list[str], **kwargs) -> dict[str, float]:
    try:
        raw = yf.download(tickers, group_by="ticker", auto_adjust=False,
                          progress=False, threads=True, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a failed poll must not kill the loop
        log(f"  ! quote fetch failed: {exc}")
        return {}
    if raw is None or raw.empty:
        return {}
    out: dict[str, float] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            price = _last_close(raw[t] if multi else raw)
        except (KeyError, TypeError):
            continue
        if price is not None:
            out[t] = price
    return out


def quotes(tickers: list[str], prepost: bool = True) -> dict[str, float]:
    """Last price for each ticker.

    Intraday minute bars are the live read; anything they miss -- a thin name
    with no prints today, or a session that has not opened -- falls back to the
    most recent daily close so the name is still evaluated rather than skipped.
    """
    if not tickers:
        return {}
    prices = _download(tickers, period="1d", interval="1m", prepost=prepost)
    missing = [t for t in tickers if t not in prices]
    if missing:
        prices.update(_download(missing, period="5d", interval="1d"))
    return prices


# --------------------------------------------------------------------------
# What has already been said
# --------------------------------------------------------------------------

class SentState:
    """Per-ticker memory of the last alert, so the bot repeats itself only on
    purpose. Persisted, because a restart is not news."""

    def __init__(self, data: dict | None = None, path: Path = STATE_PATH):
        self.path = Path(path)
        self.tickers: dict[str, dict] = (data or {}).get("tickers", {})

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "SentState":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        try:
            return cls(json.loads(path.read_text()), path=path)
        except (json.JSONDecodeError, OSError) as exc:
            log(f"  ! unreadable state file ({exc}); starting fresh")
            return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"updated_at": _now(), "tickers": self.tickers}, indent=1))

    def targets_hit(self, ticker: str) -> int:
        return int(self.tickers.get(ticker, {}).get("targets_hit", 0))

    def forget(self, ticker: str) -> None:
        self.tickers.pop(ticker, None)

    def prune(self, keep: set[str]) -> None:
        for t in set(self.tickers) - keep:
            del self.tickers[t]

    def should_send(self, a: Alert, repeat_after: int, silent_start: bool) -> bool:
        if a.kind == QUIET:
            return False
        prev = self.tickers.get(a.ticker)
        if prev is None:
            # First sighting: say what is true now, unless the run was asked to
            # come up quietly and only report changes from here.
            return not silent_start
        if prev.get("key") != a.key:
            return True
        if a.sustained and repeat_after > 0:
            try:
                last = datetime.fromisoformat(prev["sent_at"])
            except (KeyError, ValueError):
                return True
            age_min = (datetime.now(last.tzinfo) - last).total_seconds() / 60
            return age_min >= repeat_after
        return False

    def record(self, a: Alert, sent: bool) -> None:
        prev = self.tickers.get(a.ticker, {})
        self.tickers[a.ticker] = {
            "key": a.key,
            "kind": a.kind,
            # The high-water mark only ever climbs, so a dip back through a
            # target does not re-arm it.
            "targets_hit": max(a.targets_hit, int(prev.get("targets_hit", 0))),
            "sent_at": _now() if sent else prev.get("sent_at", ""),
            "last_price": round(a.price, 6),
            "last_seen": _now(),
        }


# --------------------------------------------------------------------------
# Discord
# --------------------------------------------------------------------------

def embed(a: Alert) -> dict:
    """One alert as a Discord embed: a coloured left bar, the headline, the
    detail and the price that triggered it."""
    e = {
        "title": a.title,
        "description": a.body,
        "fields": [{"name": "price", "value": fmt(a.price), "inline": False}],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if a.color:
        e["color"] = a.color
    return e


class Discord:
    def __init__(self, url: str, username: str = BOT_NAME):
        if not url:
            raise ValueError("no webhook URL")
        self.url, self.username = url, username

    def _post(self, payload: dict) -> None:
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "delta-aio-watchbot"},
        )
        urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read()

    def send(self, alerts: list[Alert]) -> int:
        """Post alerts in batches. Returns how many made it out."""
        done = 0
        for i in range(0, len(alerts), MAX_EMBEDS):
            batch = alerts[i: i + MAX_EMBEDS]
            payload = {"username": self.username, "embeds": [embed(a) for a in batch]}
            for attempt in range(3):
                try:
                    self._post(payload)
                    done += len(batch)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429:  # rate limited -- Discord says for how long
                        wait = 1.0
                        try:
                            wait = float(json.loads(exc.read()).get("retry_after", 1.0))
                        except (ValueError, OSError):
                            pass
                        log(f"  rate limited, waiting {wait:.1f}s")
                        time.sleep(min(wait, 30) + 0.25)
                        continue
                    log(f"  ! discord rejected the post ({exc.code}): {exc.reason}")
                    break
                except (urllib.error.URLError, OSError) as exc:
                    log(f"  ! discord post failed ({exc}); retry {attempt + 1}/3")
                    time.sleep(2 ** attempt)
            else:
                log("  ! giving up on this batch")
        return done


def print_alert(a: Alert) -> None:
    log(f"  {a.title:<34} {a.body}  (price {fmt(a.price)})")


# --------------------------------------------------------------------------
# Poll
# --------------------------------------------------------------------------

def poll_once(wl: WatchList, state: SentState, near_pct: float = DEFAULT_NEAR_PCT,
              repeat_after: int = DEFAULT_REPEAT_AFTER, silent_start: bool = False,
              prepost: bool = True) -> tuple[list[Alert], list[str]]:
    """One sweep of the whole list. Returns (alerts to send, tickers with no quote)."""
    tickers = wl.tickers()
    prices = quotes(tickers, prepost=prepost)
    missing = [t for t in tickers if t not in prices]

    to_send: list[Alert] = []
    for ticker in tickers:
        price = prices.get(ticker)
        if price is None:
            continue
        a = evaluate(wl.watches[ticker], price, near_pct=near_pct,
                     targets_hit=state.targets_hit(ticker))
        send = state.should_send(a, repeat_after, silent_start)
        state.record(a, send)
        if send:
            to_send.append(a)
    return to_send, missing


def run(wl: WatchList, webhook: str | None, interval: int = DEFAULT_INTERVAL,
        once: bool = False, near_pct: float = DEFAULT_NEAR_PCT,
        repeat_after: int = DEFAULT_REPEAT_AFTER, hours: str = "open",
        silent_start: bool = False, console: bool = False) -> None:
    if not len(wl):
        print("watch list is empty — add a name first:\n"
              "  python3 watchbot.py add NIO --entry 4.58 --stop 4.30 --targets 5,5.5")
        return

    sink = None if console else Discord(webhook)
    state = SentState.load()
    state.prune(set(wl.tickers()))     # names taken off the list stop being tracked
    sessions = HOURS_MODES[hours]

    log(f"watching {len(wl)} name{'s' if len(wl) != 1 else ''} "
        f"({', '.join(wl.tickers())}) every {interval}s · "
        f"{'console only' if console else 'posting to Discord'} · hours={hours}")

    first, idle, last_missing = True, False, None
    while True:
        market = market_status()
        if market["state"] not in sessions:
            if once:
                log(f"market is {market['state']} — nothing to poll (--hours {hours})")
                return
            if not idle:      # say it once, not every time round the loop
                log(f"market is {market['state']} — idling until it opens")
                idle = True
            time.sleep(min(interval, 300))
            continue
        if idle:
            log(f"market is {market['state']} — resuming")
            idle = False

        alerts, missing = poll_once(wl, state, near_pct=near_pct,
                                    repeat_after=repeat_after,
                                    silent_start=silent_start and first,
                                    prepost=market["state"] != "open")
        if missing and missing != last_missing:
            # A permanently bad symbol would otherwise say this every poll.
            log(f"  no quote for {', '.join(missing)}")
        last_missing = missing
        for a in alerts:
            print_alert(a)
        if alerts and sink:
            sent = sink.send(alerts)
            if sent < len(alerts):
                log(f"  ! only {sent}/{len(alerts)} alerts reached Discord")
        state.save()
        first = False

        if once:
            if not alerts:
                log("  nothing changed")
            return
        time.sleep(interval)


# --------------------------------------------------------------------------
# Building the list
# --------------------------------------------------------------------------

def auto_watch(ticker: str, types: list[str]) -> Watch:
    """Let the scanner's own rules pick the levels for one name."""
    from scanner import scan_universe

    ticker = ticker.strip().upper()
    result = scan_universe({ticker: ticker}, types, limit=1, min_rr=0.0)
    if not result.get("scanned"):
        raise ValueError(f"no daily bars came back for {ticker} — check the symbol "
                         f"and that Yahoo is reachable")
    setups = result.get("setups", [])
    if not setups:
        raise ValueError(
            f"no {'/'.join(types)} setup on {ticker} right now — pass levels by hand:\n"
            f"  python3 watchbot.py add {ticker} --entry E --stop S --targets T1,T2")
    return Watch.from_setup(setups[0])


def import_setups(index: str, types: list[str], limit: int,
                  statuses: list[str] | None) -> list[Watch]:
    from scanner import load_universe, scan_universe

    universe = load_universe(which=index)
    result = scan_universe(universe, types, limit=max(limit * 3, limit))
    picked = []
    for s in result["setups"]:
        if statuses and s["status"] not in statuses:
            continue
        picked.append(Watch.from_setup(s))
        if len(picked) >= limit:
            break
    return picked


def print_list(wl: WatchList) -> None:
    if not len(wl):
        print("\nwatch list is empty — add one with:\n"
              "  python3 watchbot.py add NIO --entry 4.58 --stop 4.30 --targets 5,5.5\n")
        return
    hdr = (f"\n{'TICKER':<8}{'TYPE':<10}{'ENTRY':>10}{'ZONE':>18}{'STOP':>10}"
           f"{'TARGETS':>26}  SOURCE")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for w in wl.sorted():
        zone = f"{fmt(w.entry_lo)}–{fmt(w.entry_hi)}"
        tgts = ", ".join(fmt(t) for t in w.targets) or "-"
        print(f"{w.ticker:<8}{w.setup_type:<10}{w.entry:>10.2f}{zone:>18}{w.stop:>10.2f}"
              f"{tgts:>26}  {w.source}")
    print(f"\n{len(wl)} name{'s' if len(wl) != 1 else ''} on the list\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _targets(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(x) for x in str(raw).replace(" ", "").split(",") if x]


def _webhook(args) -> str | None:
    url = getattr(args, "webhook", None) or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url and not getattr(args, "console", False):
        sys.exit("no Discord webhook — set DISCORD_WEBHOOK_URL or pass --webhook, "
                 "or use --console to print alerts instead of sending them")
    return url


def main():
    ap = argparse.ArgumentParser(description="Watch-list price bot with Discord alerts.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show the watch list")

    a = sub.add_parser("add", help="add or replace a name")
    a.add_argument("ticker")
    a.add_argument("--entry", type=float, help="entry price (midpoint of the zone)")
    a.add_argument("--stop", type=float)
    a.add_argument("--targets", help="comma list, e.g. 5,5.5,6")
    a.add_argument("--zone", help="explicit entry zone as LO:HI, e.g. 4.55:4.62")
    a.add_argument("--auto", action="store_true",
                   help="derive entry/stop/targets from the scanner rules")
    a.add_argument("--types", default="ote,ma,breakout", help="rules to try with --auto")
    a.add_argument("--note", default="")

    r = sub.add_parser("rm", help="remove a name")
    r.add_argument("ticker")

    i = sub.add_parser("import", help="fill the list from a scanner run")
    i.add_argument("--index", default="sp100", choices=["sp100", "sp500", "ndx", "both"])
    i.add_argument("--types", default="ote,ma,breakout")
    i.add_argument("--limit", type=int, default=10)
    i.add_argument("--status", default="IN ENTRY ZONE,WAIT FOR PULLBACK,COILING",
                   help="comma list of scanner statuses to take, or 'any'")
    i.add_argument("--replace", action="store_true", help="clear the list first")

    sub.add_parser("clear", help="empty the watch list")

    p = sub.add_parser("run", help="poll the list and alert on changes")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="seconds between polls")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--near", type=float, default=DEFAULT_NEAR_PCT,
                   help="%% from the zone that counts as approaching")
    p.add_argument("--repeat-after", type=int, default=DEFAULT_REPEAT_AFTER, dest="repeat_after",
                   help="minutes before a still-true state repeats (0 = never)")
    p.add_argument("--hours", default="open", choices=sorted(HOURS_MODES),
                   help="sessions to poll in (default: regular hours only)")
    p.add_argument("--silent-start", action="store_true", dest="silent_start",
                   help="seed state on the first pass without alerting")
    p.add_argument("--console", action="store_true", help="print alerts, send nothing")
    p.add_argument("--webhook", help="Discord webhook URL (or set DISCORD_WEBHOOK_URL)")

    t = sub.add_parser("test", help="post a sample alert to the webhook")
    t.add_argument("--webhook")

    args = ap.parse_args()
    try:
        _dispatch(args)
    except KeyboardInterrupt:
        # State is written after every pass, so there is nothing to flush here.
        print("\nstopped")
    except (ValueError, LookupError, OSError) as exc:
        # Bad levels, an unknown ticker, an unreadable list -- all of these are
        # the operator's problem to fix, not a bug worth a traceback.
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        sys.exit(f"error: {msg}")


def _dispatch(args) -> None:
    wl = WatchList.load()

    if args.cmd == "list":
        print_list(wl)

    elif args.cmd == "add":
        if args.auto:
            w = auto_watch(args.ticker, [x for x in args.types.split(",") if x])
            if args.note:
                w.note = args.note
        else:
            if args.entry is None or args.stop is None:
                sys.exit("--entry and --stop are required (or use --auto)")
            lo = hi = 0.0
            if args.zone:
                try:
                    lo, hi = (float(x) for x in args.zone.split(":"))
                except ValueError:
                    sys.exit("--zone must look like LO:HI, e.g. 4.55:4.62")
            w = Watch(ticker=args.ticker, entry=args.entry, stop=args.stop,
                      targets=_targets(args.targets), entry_lo=lo, entry_hi=hi,
                      note=args.note)
        wl.add(w)
        wl.save()
        # Levels changed, so whatever was last said about this name is stale.
        state = SentState.load()
        state.forget(w.ticker)
        state.save()
        print(f"watching {w.ticker}: entry {fmt(w.entry)} "
              f"({fmt(w.entry_lo)}–{fmt(w.entry_hi)}) · stop {fmt(w.stop)} · "
              f"targets {', '.join(fmt(t) for t in w.targets) or '-'}")

    elif args.cmd == "rm":
        w = wl.remove(args.ticker)
        wl.save()
        state = SentState.load()
        state.forget(w.ticker)
        state.save()
        print(f"removed {w.ticker}")

    elif args.cmd == "clear":
        wl.watches.clear()
        wl.save()
        SentState().save()
        print("watch list cleared")

    elif args.cmd == "import":
        statuses = None if args.status.lower() == "any" else \
            [s.strip().upper() for s in args.status.split(",") if s.strip()]
        picked = import_setups(args.index, [x for x in args.types.split(",") if x],
                               args.limit, statuses)
        if args.replace:
            wl.watches.clear()
            SentState().save()
        for w in picked:
            wl.add(w)
        wl.save()
        print(f"added {len(picked)} setup{'s' if len(picked) != 1 else ''} to the watch list")
        print_list(wl)

    elif args.cmd == "run":
        run(wl, _webhook(args), interval=args.interval, once=args.once,
            near_pct=args.near, repeat_after=args.repeat_after, hours=args.hours,
            silent_start=args.silent_start, console=args.console)

    elif args.cmd == "test":
        sample = Watch(ticker="RGTI", entry=18.20, stop=17.32, targets=[20.0])
        alert = evaluate(sample, 18.18)
        n = Discord(_webhook(args)).send([alert])
        print(f"sent {n} test alert to Discord" if n else "nothing sent — see the error above")


if __name__ == "__main__":
    main()
