# Delta-AIO

Two Yahoo Finance tools that share one localhost app:

- **SPX gamma dashboard** (`/`) — options positioning for the index.
- **Setup scanner** (`/scanner`) — scans the Nasdaq-100 + S&P 500 for trade
  setups and posts them as a card grid.

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

The universe comes from the Wikipedia S&P 500 / Nasdaq-100 tables and is cached
to `.universe_cache.json`; if both are unreachable and no cache exists, it falls
back to a built-in mega-cap list and says so. A full scan pulls a year of daily
bars for ~500 names and takes about a minute, so `/api/scan` caches for 15
minutes.

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
