# Delta-AIO

## spx_dashboard.py

SPX options / gamma dashboard sourced from Yahoo Finance.

Yahoo does not publish an options chain for the `^SPX` / `^GSPC` index itself, so
the script pulls the **SPY** chain, computes everything from it, and converts the
dollar levels into SPX-equivalent terms using the live SPX/SPY ratio. The ratio
drifts away from a flat x10 (dividends, tracking), which the output makes explicit.

### Install

```
pip install -r requirements.txt
```

### Run

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
