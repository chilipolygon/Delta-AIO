# Delta-AIO

Two Yahoo Finance tools that share one localhost app:

- **SPX gamma dashboard** (`/`) — options positioning for the index.
- **Setup scanner** (`/scanner`) — scans the Nasdaq-100 + S&P 500 for trade
  setups and posts them as a card grid.
- **Paper portfolio** (`/portfolio`) — start with an amount of cash, take setups
  from the scanner, and track equity.

```
pip install -r requirements.txt
python3 app.py            # -> http://127.0.0.1:5000
```

---

# Setup scanner

Three rules run over every constituent, and each hit is tagged with the rule
that fired:

| Rule | Fires when | Entry / stop / targets |
|---|---|---|
| `ote` | a fresh impulse leg (swing low → swing high) is retracing | entry = 0.62–0.79 pocket, stop = below the swing low − ½ ATR, targets = leg high then the 1.272 / 1.618 / 2.0 extensions |
| `ma` | price > 50MA > 200MA with a rising 50MA, easing into the 21 EMA | entry = 21 EMA, stop = 20-bar low − ⅓ ATR, targets = 60-bar high then extensions |
| `breakout` | Bollinger width in its tightest quartile, coiling near the range high | entry = 20-bar range high, stop = range low − ⅓ ATR, targets = 1× / 1.618× / 2× the measured move |

### Confluence — every hit is graded on the tape

Alongside the rule that fired, each name is read on MACD (cross, freshness,
histogram), RSI (zone + direction), the 200 SMA (side, distance, slope), volume
(vs 20-day average, expanding or fading), trend structure (MA stack plus whether
swings still step up) and realized volatility (20d vs 60d). These roll into a
**−100…+100 bias** that feeds the ranking score, so a setup with a bearish tape
sinks even when its geometry is clean.

These all come off daily bars, so they run over the whole universe for free.

*A note on "conflicted":* momentum easing inside an intact uptrend — trend and
200 SMA positive while MACD and RSI cool off — is the **pullback signature these
rules hunt for**, not a contradiction. Only the structure disagreeing with itself
(trend vs 200 SMA) counts as a conflict; treating the former as one penalised
exactly the setups the scanner exists to find.

### Click a card → dealer positioning and the full read

Option chains are far too slow to fetch for 500 names (each ticker is its own
request), so they are fetched **on click**, for one ticker, and cached 5 minutes.
The detail drawer shows:

- **A verdict** — SHORT THE BREAK / LONG THE BREAK / FADE THE EDGES / BUY THE DIP
  / TRADE THE TREND / SIT ON HANDS — with a **conviction score out of 100**, plus
  *What I see · The play · Invalid if*.
- **GEX levels**: flip (zero-gamma) and its cushion, pin, floor, ceiling, the
  −gamma fuel nodes above and below spot, the target beyond each break, and net.
- **VEX levels**: the dominant vanna node, net vanna and its lean, and ATM IV.
- **Trade ladder**: OTE entry, TP1…TPn and the stop, against real strikes.
- **Gamma exposure by strike**: +GEX gold, −GEX purple, with spot, the flip and
  the ladder rungs drawn across it. A table view carries the same numbers.

Regimes are read off the gamma structure around spot:

| Regime | Shape | Reading |
|---|---|---|
| `TRAPDOOR` | a +gamma floor under spot with much larger −gamma around it | the floor looks like support; losing it accelerates the fall |
| `SQUEEZE FUEL` | −gamma stacked above spot | a break up feeds itself |
| `PIN MAGNET` | one dominant +gamma node at spot | chop, fade the edges |
| `LONG GAMMA` | net +gamma, spot above the flip | dips get absorbed |
| `SHORT GAMMA` | spot below the flip, net −gamma | moves extend, ranges fail |
| `NO CLEAR PIN` | no node dominates its neighbours | no structural edge |

If the tape flatly contradicts the structure (a bullish regime with a −25 bias or
worse), the verdict downgrades to STAND ASIDE rather than pretending to agree.

**This is a description of where dealer hedging adds or removes energy — not a
direction bet and not advice.** GEX/VEX assume the standard dealer convention
(long calls, short puts), which is a convention rather than observed positioning,
and open interest updates once daily pre-open.

### SPX / index signal

A panel at the top of `/scanner` runs the same three rules on the index itself
and layers dealer positioning on top. Yahoo publishes no chain for `^SPX` /
`^GSPC`, so positioning is read from the proxy ETF (SPY for SPX, QQQ for NDX)
and **strikes** are converted to index points with the live ratio. Dollar
exposures are *not* converted — they are proxy-chain dollars and have no
meaningful index-point equivalent, so they stay labelled as the proxy's.

```
python3 index_signal.py                      # SPX via SPY
python3 index_signal.py --index ^NDX --proxy QQQ
python3 index_signal.py --asof 2026-03-16    # price/tape only — see below
```

Click the panel for the same detail drawer the stock cards open.

### Backtesting from a past date

Switch the scanner's **Mode** to *as of date · backtest*, pick a date and a
forward window. Every rule then sees only bars up to the cutoff, and the bars
after it are used to score what happened — never to generate the signal.

```
python3 backtest.py --asof 2026-03-16 --forward 30
python3 backtest.py --asof 2026-01-05 --forward 45 --types ote --json bt.json
```

Each signal is scored honestly:

- A fill requires price to actually trade **into the entry zone** after the
  signal. One that never did is reported `NO FILL`, not counted as a win or loss.
- When a single daily bar's range spans **both** the stop and a target, the stop
  is taken. Intrabar order is unknowable from daily data, and the pessimistic
  reading is the one that doesn't flatter the results.
- Outcomes carry R multiple, bars held, and MFE/MAE in R.

> **What a backtest here cannot cover.** Yahoo serves historical price bars but
> **only the current option chain**. So the price rules, the confluence and the
> outcomes are genuinely historical, while **GEX/VEX and the regime verdict are
> not backtestable** from this data source. Rather than pin today's dealer book
> onto a past bar — which would look like a result and be a fiction — the as-of
> paths omit positioning entirely and say so.

### Auto-refresh during the session

In live mode an auto-refresh interval (1/5/15 min) re-runs the scan, and the
header shows the session state — `open`, `premarket`, `afterhours`, `weekend`.
Out of hours the refresh skips the expensive universe scan and only keeps the
index panel current, since the tape isn't moving. Auto-refresh is disabled in
as-of mode, where the result cannot change. *Market holidays are not tracked* —
the check is weekday and clock only.

### The exact contract

Open a card's detail drawer and it resolves **one concrete option contract** for
the setup: the full OCC symbol, strike, expiry and DTE, live bid/ask/mid and the
spread, IV, delta, open interest and volume, cost per contract, breakeven, and
the modelled value at T1 and at the stop.

Direction follows the setup (every rule here is long-only, so a call). Expiry is
the first listed one with at least 30 days — targets are swing levels, and buying
less time than the thesis needs is a common way a correct setup still loses.
Strike defaults to the entry (`--moneyness atm`; `otm` aims halfway to T1).

Contracts are resolved **in the drawer, not the grid** — each one needs that
ticker's chain, and 500 of those is not a thing you can do per scan.

Two honesty guards: liquidity is labelled (thin / wide / moderate / liquid), and
if the quoted mid disagrees with the Black-Scholes model by more than 3× the
contract is flagged as a stale quote. Cost, breakeven, the P&L estimates *and the
position size* all rest on that price, so a junk print must not pass silently.

---

# Paper portfolio

```
python3 portfolio.py new --name main --cash 25000
python3 portfolio.py list
```

Or use `/portfolio`: pick a starting balance and risk-per-trade, then add setups
from the scanner's detail drawer as stock or options. The page shows equity and
cash, realized/unrealized, win rate, average R, max drawdown, an equity curve,
and tables of open positions (marked to the last close) and closed trades.

**Sizing** risks a fixed % of *current* equity — so wins compound and losses
shrink the next position — from the entry to the setup's own stop, capped at 20%
of equity in any single position.

- *stock*: `shares = risk budget ÷ (entry − stop)`
- *option*: risk per contract is the modelled premium lost between here and the
  stop, **not** the whole premium (which badly undersizes) and not zero (which a
  far-dated contract's reprice can imply). It is floored at 25% of premium so a
  flattering reprice cannot produce an enormous position.

### Backtest → portfolio

`/portfolio` can replay a backtest into a saved portfolio, sized against equity
at the time of each trade. Fills and exits are applied **in date order**, so
positions overlap the way they really would.

That has a consequence worth stating plainly: **capital is finite, and a replay
usually cannot take every signal.** With positions held concurrently, later ones
can be unaffordable. Those are recorded and reported on the page — "took 5 of 27
filled signals, the 22 it skipped were worth +14.6R" — because a portfolio that
quietly took a fifth of the signals would misrepresent both the strategy and what
that starting balance can actually carry.

Options mode prices contracts from the **current** chain, not the one that
existed on the as-of date. Stock mode is the honest one for a historical replay.

**All of this is a simulation.** Nothing is routed anywhere. Fills are assumed at
the recorded price with no slippage or commission, and option marks come from
Black-Scholes with flat IV rather than a live quote.

---

### Setup status

Every hit is then classified the way a hand-kept tracker reads it:

`IN ENTRY ZONE` → `COILING` (a breakout still under its trigger — the normal
pre-break state) → `WAIT FOR PULLBACK` → `TARGET n/N HIT` → `BELOW ZONE` →
`DON'T CHASE` (price is closer to T1 than to the entry). Invalidated setups —
price below the stop — are dropped rather than shown.

**Two different R:R numbers appear on each card, and the distinction matters:**

- *risk / reward / R:R* are measured **from the current price** to the stop and
  to the next unhit target — what you actually get taking the trade right now.
  This is why an extended name shows a terrible ratio and gets flagged
  don't-chase.
- *at entry … R:R* is the setup's quality **at its planned entry**. This drives
  ranking and the "Min R:R at entry" filter.

```
python3 scanner.py                       # ranked table in the terminal
python3 scanner.py --html setups.html    # self-contained shareable page
python3 scanner.py --types ote,breakout --limit 40 --min-rr 1.5
python3 scanner.py --refresh-universe    # re-read the index membership lists
```

#### The universe

All **503 S&P 500 constituents** ship with the repo in `data/sp500.csv` (503, not
500 — GOOG/GOOGL and other dual share classes each count). Membership resolves in
this order:

1. `.universe_cache.json` on disk, **unless** it holds fewer than 400 tickers, in
   which case it is treated as stale and ignored.
2. A live fetch of the current S&P 500 from a maintained CSV dataset. This is
   plain-stdlib parsing, so it does **not** need `lxml`, and it refuses a
   response with under 400 rows rather than accepting a truncated index.
3. The Nasdaq-100 from Wikipedia, merged on top. This one *does* need `lxml`; if
   it fails the scan continues with the S&P 500 alone and says so.
4. The bundled `data/sp500.csv`.

The point of steps 1 and 4 is that **a scan is never silently run against a
handful of names** — offline, without `lxml`, or behind a firewall you still get
the full index. Refresh membership with `--refresh-universe`.

A full scan pulls a year of daily bars for ~500 names and takes about a minute,
so `/api/scan` caches for 15 minutes.

Filters sit in one row above the grid: rule, status, sort, minimum R:R at entry,
and whether to show every rule that fired on a ticker or only its best-scoring
one. A table view carries every number on the cards.

**These levels are derived by rule, not judgement.** They are daily-bar
approximations and say nothing about whether a trade is a good idea.

---

# SPX gamma dashboard

```
pip install -r requirements.txt
python3 app.py            # -> http://127.0.0.1:5000
python3 app.py --port 8080
```

The page shows the index spot as the headline figure, a KPI row (gap, RSI, VWAP,
ATM IV vs VIX, expected move, P/C open interest), a key-levels ladder plotting
the walls, max pain and zero gamma against the expected-move band and yesterday's
range, plus two per-strike charts: net gamma exposure (blue where dealers are long
gamma, red where short) and open interest (calls vs puts). Every chart has a hover
tooltip and a table view.

Controls sit in one row above the charts: chain source, index, expiry, strike
window, whether levels are quoted in SPX points or underlying dollars, and an
auto-refresh interval. Results are cached for 60 seconds so a page refresh does
not re-hit Yahoo. Light and dark themes both ship; the toggle is top-right.

## Terminal report

Yahoo does not publish an options chain for the `^SPX` / `^GSPC` index itself, so
the script pulls the **SPY** chain, computes everything from it, and converts the
dollar levels into SPX-equivalent terms using the live SPX/SPY ratio. The ratio
drifts away from a flat x10 (dividends, tracking), which the output makes explicit.

```
python3 spx_dashboard.py                      # nearest expiry
python3 spx_dashboard.py --expiry 2026-08-10  # specific expiry
python3 spx_dashboard.py --ticker QQQ --index ^NDX
```

### What it computes

| Field | Source |
|---|---|
| Spot, prev close, PDH/PDL, gap | daily bars |
| RSI(14) | Wilder-smoothed daily closes |
| VWAP | 1-minute bars, typical price |
| P/C volume and OI | chain totals (two different measures — volume is today's flow, OI is standing position) |
| Max pain | strike minimizing total in-the-money payout against open interest |
| Call / put wall | strike with the largest call / put open interest |
| Expected move | ATM straddle mid |
| ATM IV vs VIX | chain IV at the ATM strike, sanity-checked against `^VIX` |
| Call / put / net / gross GEX | Black-Scholes gamma × OI × 100 × spot² × 1%, dealer convention: long calls, short puts |
| Zero gamma | interpolated price where net GEX crosses zero |

### Caveats

- GEX uses the standard retail dealer-positioning assumption (dealers long calls,
  short puts). It is a convention, not observed positioning.
- Yahoo's `openInterest` updates once daily, before the open; intraday GEX is
  therefore based on the prior session's OI.
- Yahoo's `impliedVolatility` is its own calculation and can be noisy or missing
  on illiquid far-out strikes; those rows contribute zero gamma.
