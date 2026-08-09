#!/usr/bin/env python3
"""
Paper portfolio: start with an amount of cash, take setups from the scanner as
stock or option positions, and track equity over time.

This is a SIMULATION. Nothing is routed anywhere, fills are assumed at the
prices recorded on the position, and option marks come from Black-Scholes with
flat IV unless a live quote is fetched. It is a way to see what a strategy
would have done, not a broker.

State lives in data/portfolios/<name>.json so a run survives a restart.

    python3 portfolio.py list
    python3 portfolio.py new --name main --cash 25000
    python3 portfolio.py show --name main
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path

PORTFOLIO_DIR = Path(__file__).with_name("data") / "portfolios"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,48}$")

# Sizing defaults -- deliberately conservative
DEFAULT_RISK_PCT = 1.0        # of current equity, risked from entry to stop
MAX_POSITION_PCT = 20.0       # of equity in any single position
MIN_OPTION_RISK_FRACTION = 0.25   # see size_option()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    name = (name or "").strip()
    if not SAFE_NAME.match(name):
        raise ValueError("portfolio name must be 1-49 chars of letters, digits, space, _ . or -")
    return name


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def size_stock(equity: float, entry: float, stop: float, risk_pct: float) -> tuple[int, float]:
    """Shares such that a stop-out costs about risk_pct of equity."""
    per_share = entry - stop
    if per_share <= 0 or entry <= 0:
        return 0, 0.0
    budget = equity * risk_pct / 100
    qty = int(budget // per_share)
    cap = int((equity * MAX_POSITION_PCT / 100) // entry)
    qty = max(0, min(qty, cap))
    return qty, round(qty * per_share, 2)


def size_option(equity: float, contract: dict, risk_pct: float) -> tuple[int, float]:
    """Contracts such that the setup being wrong costs about risk_pct of equity.

    Risk per contract is the modelled premium lost between here and the stop --
    not the whole premium, which would badly undersize, and not zero, which the
    model can imply for a far-dated contract. It is floored at a fraction of the
    premium so a flattering reprice cannot produce an enormous position.
    """
    prem = float(contract.get("mid") or contract.get("model_price") or 0)
    if prem <= 0:
        return 0, 0.0
    at_stop = float(contract.get("est_value_at_stop") or 0)
    per_contract = max((prem - at_stop) * 100, prem * 100 * MIN_OPTION_RISK_FRACTION)
    budget = equity * risk_pct / 100
    qty = int(budget // per_contract)
    # never spend more than the position cap on premium
    cap = int((equity * MAX_POSITION_PCT / 100) // (prem * 100))
    qty = max(0, min(qty, cap))
    return qty, round(qty * per_contract, 2)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class Position:
    id: str
    ticker: str
    kind: str                 # "stock" | "option"
    qty: int
    entry_price: float        # per share, or per share of premium
    opened_at: str
    setup_type: str = ""
    status_at_open: str = ""
    stop: float | None = None
    targets: list[float] = field(default_factory=list)
    contract: dict | None = None
    note: str = ""
    source: str = "manual"    # manual | scanner | backtest
    # filled on close
    closed_at: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    realized: float | None = None
    r_multiple: float | None = None
    # transient mark
    mark: float | None = None
    market_value: float | None = None
    unrealized: float | None = None

    @property
    def multiplier(self) -> int:
        return 100 if self.kind == "option" else 1

    @property
    def cost_basis(self) -> float:
        return round(self.qty * self.entry_price * self.multiplier, 2)


@dataclass
class Portfolio:
    name: str
    starting_cash: float
    cash: float
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    risk_pct: float = DEFAULT_RISK_PCT
    positions: list[Position] = field(default_factory=list)
    closed: list[Position] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    origin: dict = field(default_factory=dict)   # e.g. the backtest that built it

    # ---- persistence ----
    @property
    def path(self) -> Path:
        return PORTFOLIO_DIR / f"{self.name}.json"

    def save(self) -> None:
        PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        self.path.write_text(json.dumps(self.to_dict(), indent=1))

    def to_dict(self) -> dict:
        # asdict() only serializes fields, so the computed properties the UI
        # needs (cost_basis, multiplier) have to be added explicitly or the
        # cost column silently renders empty.
        def pos_dict(p):
            if isinstance(p, dict):
                return p
            return {**asdict(p), "cost_basis": p.cost_basis, "multiplier": p.multiplier}

        d = asdict(self)
        d["positions"] = [pos_dict(p) for p in self.positions]
        d["closed"] = [pos_dict(p) for p in self.closed]
        return d

    @classmethod
    def load(cls, name: str) -> "Portfolio":
        name = safe_name(name)
        path = PORTFOLIO_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"no portfolio named {name!r}")
        raw = json.loads(path.read_text())
        fields = {f for f in Position.__dataclass_fields__}
        def mk(p):
            return Position(**{k: v for k, v in p.items() if k in fields})
        raw["positions"] = [mk(p) for p in raw.get("positions", [])]
        raw["closed"] = [mk(p) for p in raw.get("closed", [])]
        return cls(**raw)

    @classmethod
    def create(cls, name: str, cash: float, risk_pct: float = DEFAULT_RISK_PCT,
               overwrite: bool = False) -> "Portfolio":
        name = safe_name(name)
        if cash <= 0:
            raise ValueError("starting cash must be positive")
        path = PORTFOLIO_DIR / f"{name}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"portfolio {name!r} already exists")
        p = cls(name=name, starting_cash=float(cash), cash=float(cash), risk_pct=risk_pct)
        p.equity_curve = [{"t": _now(), "equity": float(cash), "cash": float(cash)}]
        p.save()
        return p

    @staticmethod
    def list_all() -> list[dict]:
        if not PORTFOLIO_DIR.exists():
            return []
        out = []
        for f in sorted(PORTFOLIO_DIR.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                out.append({"name": d["name"], "starting_cash": d["starting_cash"],
                            "cash": d["cash"], "open": len(d.get("positions", [])),
                            "closed": len(d.get("closed", [])),
                            "updated_at": d.get("updated_at"),
                            "origin": d.get("origin", {})})
            except Exception:
                continue
        return out

    # ---- trading ----
    def equity(self) -> float:
        held = sum((p.market_value if p.market_value is not None else p.cost_basis)
                   for p in self.positions)
        return round(self.cash + held, 2)

    def open_from_setup(self, setup: dict, kind: str = "stock",
                        contract: dict | None = None, qty: int | None = None,
                        source: str = "scanner", when: str | None = None) -> Position:
        equity = self.equity()
        entry = float(setup.get("entry") or setup.get("price") or 0)
        stop = float(setup.get("stop") or 0)

        if kind == "option":
            if not contract:
                raise ValueError("an option position needs a contract")
            price = float(contract.get("mid") or contract.get("model_price") or 0)
            if qty is None:
                qty, _ = size_option(equity, contract, self.risk_pct)
        else:
            price = entry
            if qty is None:
                qty, _ = size_stock(equity, entry, stop, self.risk_pct)

        if qty <= 0:
            raise ValueError("position sizes to zero — risk budget too small for this stop")
        cost = qty * price * (100 if kind == "option" else 1)
        if cost > self.cash:
            raise ValueError(f"cost {cost:,.2f} exceeds cash {self.cash:,.2f}")

        pos = Position(
            id=uuid.uuid4().hex[:12],
            ticker=setup.get("ticker", "?"), kind=kind, qty=qty,
            entry_price=round(price, 4), opened_at=when or _now(),
            setup_type=setup.get("setup_type", ""),
            status_at_open=setup.get("status", ""),
            stop=stop or None, targets=list(setup.get("targets") or []),
            contract=contract, source=source,
            note=setup.get("note", ""),
        )
        self.cash = round(self.cash - cost, 2)
        self.positions.append(pos)
        self._snapshot(when)
        return pos

    def close(self, position_id: str, price: float, reason: str = "manual",
              when: str | None = None) -> Position:
        idx = next((i for i, p in enumerate(self.positions) if p.id == position_id), None)
        if idx is None:
            raise KeyError(f"no open position {position_id!r}")
        pos = self.positions.pop(idx)
        mult = pos.multiplier
        proceeds = pos.qty * float(price) * mult
        pos.closed_at = when or _now()
        pos.exit_price = round(float(price), 4)
        pos.exit_reason = reason
        pos.realized = round(proceeds - pos.cost_basis, 2)

        if pos.kind == "stock" and pos.stop:
            per_share_risk = pos.entry_price - pos.stop
            if per_share_risk > 0:
                pos.r_multiple = round((price - pos.entry_price) / per_share_risk, 2)
        elif pos.kind == "option" and pos.contract:
            at_stop = float(pos.contract.get("est_value_at_stop") or 0)
            risk = max(pos.entry_price - at_stop, pos.entry_price * MIN_OPTION_RISK_FRACTION)
            if risk > 0:
                pos.r_multiple = round((price - pos.entry_price) / risk, 2)

        self.cash = round(self.cash + proceeds, 2)
        self.closed.append(pos)
        self._snapshot(when)
        return pos

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """prices: ticker -> last underlying price."""
        from contracts import reprice
        for p in self.positions:
            spot = prices.get(p.ticker)
            if spot is None:
                p.mark = p.entry_price
            elif p.kind == "option" and p.contract:
                p.mark = reprice(p.contract, spot)
            else:
                p.mark = round(float(spot), 4)
            p.market_value = round(p.qty * p.mark * p.multiplier, 2)
            p.unrealized = round(p.market_value - p.cost_basis, 2)

    def _snapshot(self, when: str | None = None) -> None:
        """`when` lets a historical replay stamp the real bar date instead of
        the wall clock, so the equity curve is plottable against time."""
        self.equity_curve.append({"t": when or _now(), "equity": self.equity(),
                                  "cash": self.cash})

    # ---- reporting ----
    def stats(self) -> dict:
        eq = self.equity()
        realized = round(sum(p.realized or 0 for p in self.closed), 2)
        unreal = round(sum(p.unrealized or 0 for p in self.positions), 2)
        wins = [p for p in self.closed if (p.realized or 0) > 0]
        rs = [p.r_multiple for p in self.closed if p.r_multiple is not None]
        invested = round(sum(p.cost_basis for p in self.positions), 2)
        peak, dd = 0.0, 0.0
        for pt in self.equity_curve:
            peak = max(peak, pt["equity"])
            if peak > 0:
                dd = min(dd, (pt["equity"] - peak) / peak * 100)
        return {
            "equity": eq,
            "cash": round(self.cash, 2),
            "invested": invested,
            "starting_cash": self.starting_cash,
            "total_pnl": round(eq - self.starting_cash, 2),
            "total_pnl_pct": round((eq - self.starting_cash) / self.starting_cash * 100, 2)
            if self.starting_cash else 0.0,
            "realized": realized,
            "unrealized": unreal,
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed),
            "win_rate": round(100 * len(wins) / len(self.closed), 1) if self.closed else 0.0,
            "avg_r": round(sum(rs) / len(rs), 2) if rs else 0.0,
            "exposure_pct": round(invested / eq * 100, 1) if eq else 0.0,
            "max_drawdown_pct": round(dd, 2),
        }


# --------------------------------------------------------------------------
# Backtest -> portfolio
# --------------------------------------------------------------------------

def from_backtest(bt: dict, name: str, starting_cash: float,
                  risk_pct: float = DEFAULT_RISK_PCT, kind: str = "stock",
                  overwrite: bool = True) -> Portfolio:
    """Replay a backtest into a portfolio: every FILLED signal is taken at its
    entry and closed at the outcome the backtest already scored.

    Sizing uses equity at the time of each trade, so wins compound and losses
    shrink the next position. Signals that never filled are skipped, matching
    how the backtest scores them.
    """
    p = Portfolio.create(name, starting_cash, risk_pct, overwrite=overwrite)
    # the curve starts on the signal date; stamping it "now" would put the
    # opening point in the future relative to every trade and invert the x-axis
    if p.equity_curve and bt.get("asof"):
        p.equity_curve[0]["t"] = bt["asof"]
    p.origin = {"type": "backtest", "asof": bt.get("asof"),
                "forward_days": bt.get("forward_days"), "types": bt.get("types"),
                "position_kind": kind, "note": bt.get("note")}

    rows = [r for r in bt.get("rows", []) if r["result"].get("filled")]

    # Replay in the order the trades actually happened. Interleaving matters:
    # sizing reads equity at the time, so a different order compounds
    # differently. Fills and exits are queued together and applied by date.
    events = []
    for r in rows:
        res = r["result"]
        events.append((res.get("fill_date") or "", 0, r))
        events.append((res.get("exit_date") or res.get("fill_date") or "", 1, r))
    events.sort(key=lambda e: (e[0], e[1]))

    live: dict[int, str] = {}
    skipped: list[dict] = []
    for when, phase, r in events:
        s, res = r["setup"], r["result"]
        key = id(r)
        if phase == 0:
            try:
                pos = p.open_from_setup(s, kind=kind, source="backtest", when=when or None)
            except ValueError as exc:
                # Capital is finite: with positions held concurrently, later
                # signals can be unaffordable. Record them -- a portfolio that
                # quietly took 5 of 27 signals would badly misrepresent both the
                # strategy and what this starting balance can actually carry.
                skipped.append({"ticker": s.get("ticker"), "date": when,
                                "setup_type": s.get("setup_type"),
                                "r_multiple": res.get("r_multiple"),
                                "reason": str(exc)})
                continue
            live[key] = pos.id
        else:
            pos_id = live.pop(key, None)
            exit_px = res.get("exit")
            if pos_id is None or exit_px is None:
                continue
            p.close(pos_id, float(exit_px), reason=res.get("outcome", "backtest"),
                    when=when or None)

    p.equity_curve.sort(key=lambda pt: str(pt.get("t") or ""))
    taken = len(p.closed) + len(p.positions)
    missed_r = round(sum(x["r_multiple"] or 0 for x in skipped), 2)
    p.origin.update({
        "signals_filled": len(rows),
        "signals_taken": taken,
        "signals_skipped": len(skipped),
        "skipped_reason": ("insufficient capital while other positions were open"
                           if skipped else None),
        "skipped_total_r": missed_r,
        "skipped_detail": skipped[:50],
    })
    p.save()
    return p


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Paper portfolio for scanner setups.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list saved portfolios")

    n = sub.add_parser("new", help="create a portfolio")
    n.add_argument("--name", required=True)
    n.add_argument("--cash", type=float, required=True)
    n.add_argument("--risk", type=float, default=DEFAULT_RISK_PCT, help="%% of equity per trade")
    n.add_argument("--overwrite", action="store_true")

    sh = sub.add_parser("show", help="show a portfolio")
    sh.add_argument("--name", required=True)

    args = ap.parse_args()

    if args.cmd == "list":
        rows = Portfolio.list_all()
        if not rows:
            print("no portfolios yet — create one with:  python3 portfolio.py new --name main --cash 25000")
            return
        print(f"\n{'NAME':<20}{'START':>12}{'CASH':>12}{'OPEN':>6}{'CLOSED':>8}  UPDATED")
        for r in rows:
            print(f"{r['name']:<20}{r['starting_cash']:>12,.0f}{r['cash']:>12,.0f}"
                  f"{r['open']:>6}{r['closed']:>8}  {r['updated_at']}")
        print()
    elif args.cmd == "new":
        p = Portfolio.create(args.name, args.cash, args.risk, overwrite=args.overwrite)
        print(f"created {p.path} with ${p.starting_cash:,.2f} and {p.risk_pct}% risk per trade")
    else:
        p = Portfolio.load(args.name)
        st = p.stats()
        print(f"\n=== {p.name} ===")
        for k, v in st.items():
            print(f"  {k:<18} {v}")
        if p.positions:
            print(f"\n  {'TICKER':<8}{'KIND':<8}{'QTY':>6}{'ENTRY':>10}{'COST':>12}")
            for pos in p.positions:
                print(f"  {pos.ticker:<8}{pos.kind:<8}{pos.qty:>6}{pos.entry_price:>10.2f}"
                      f"{pos.cost_basis:>12,.2f}")
        print()


if __name__ == "__main__":
    main()
