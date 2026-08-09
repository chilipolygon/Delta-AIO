#!/usr/bin/env python3
"""
Regime engine: turns dealer positioning plus daily-bar confluence into a
verdict, a conviction score and the prose that explains both.

Regimes are read off the gamma structure around spot:

  TRAPDOOR      a +gamma floor under spot with -gamma stacked beneath it -- the
                floor looks like support but breaking it accelerates the fall
  SQUEEZE FUEL  -gamma stacked above spot -- a break up accelerates
  PIN MAGNET    a dominant +gamma node at spot -- expect chop around it
  LONG GAMMA    net +gamma and price above the flip -- dips get bought, ranges hold
  SHORT GAMMA   price below the flip with net -gamma -- moves extend, ranges fail
  NO CLEAR PIN  no node dominates its neighbours -- no structural edge

Nothing here is a trade recommendation. It is a description of where dealer
hedging flows are likely to add or remove energy, which is one input among many.
"""

from __future__ import annotations


def _m(v: float | None) -> str:
    """Format a dollar exposure compactly: 1_300_000 -> '1.3M'."""
    if v is None:
        return "n/a"
    a = abs(v)
    sign = "-" if v < 0 else "+"
    if a >= 1e9:
        return f"{sign}{a / 1e9:.1f}B"
    if a >= 1e6:
        return f"{sign}{a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}{a / 1e3:.0f}K"
    return f"{sign}{a:.0f}"


def _dominance(levels: dict) -> float:
    """How much the biggest node stands out from the next one, 0..1."""
    mags = [abs(n["gex"]) for n in (levels.get("pin"), levels.get("fuel_down"),
                                    levels.get("fuel_up"), levels.get("floor"))
            if n]
    if len(mags) < 2:
        return 0.0
    mags.sort(reverse=True)
    return 0.0 if mags[0] == 0 else min((mags[0] - mags[1]) / mags[0], 1.0)


def classify_regime(spot: float, levels: dict, conf: dict) -> dict:
    """Pick the regime and build the verdict, conviction and narrative."""
    flip = levels.get("flip")
    cushion = levels.get("flip_cushion")
    pin, floor, ceil = levels.get("pin"), levels.get("floor"), levels.get("ceiling")
    fuel_dn, fuel_up = levels.get("fuel_down"), levels.get("fuel_up")
    net_gex = levels.get("net_gex") or 0
    dom = _dominance(levels)
    bias = conf.get("bias", 0.0)

    above_flip = flip is not None and spot > flip
    cushion_pct = (cushion / spot * 100) if cushion else None
    thin_cushion = cushion_pct is not None and cushion_pct < 1.5

    # --- pick the regime -------------------------------------------------
    # A trapdoor is a +gamma floor that looks like support while much larger
    # -gamma sits below spot around it. The -gamma node may be above or below
    # the floor strike -- what matters is that it dwarfs the floor, so losing
    # the floor accelerates instead of holding.
    trapdoor = (floor and fuel_dn and abs(fuel_dn["gex"]) > abs(floor["gex"]) * 1.5)
    squeeze = (fuel_up and ceil and abs(fuel_up["gex"]) > abs(ceil["gex"] if ceil else 0) * 1.5
               and fuel_up["strike"] > spot)
    pinned = pin and abs(pin["strike"] - spot) / spot < 0.01 and dom > 0.35

    if dom < 0.15 and not pinned:
        regime, sub = "NO CLEAR PIN", "structure is contested"
    elif trapdoor:
        regime, sub = "TRAPDOOR", "cascade risk"
    elif squeeze:
        regime, sub = "SQUEEZE FUEL", "break accelerates up"
    elif pinned:
        regime, sub = "PIN MAGNET", "chop around the node"
    elif above_flip and net_gex >= 0:
        regime, sub = "LONG GAMMA", "dips get absorbed"
    else:
        regime, sub = "SHORT GAMMA", "moves extend"

    # --- verdict ---------------------------------------------------------
    verdicts = {
        "TRAPDOOR": ("SHORT THE BREAK", "SHORT the break · do NOT fade the floor", "bear"),
        "SQUEEZE FUEL": ("LONG THE BREAK", "LONG the break · do NOT fade the ceiling", "bull"),
        "PIN MAGNET": ("FADE THE EDGES", "Range-bound · fade extremes toward the pin", "neutral"),
        "LONG GAMMA": ("BUY THE DIP", "Dips toward the flip get bought", "bull"),
        "SHORT GAMMA": ("TRADE THE TREND", "Moves extend · fade nothing", "bear" if bias < 0 else "bull"),
        "NO CLEAR PIN": ("SIT ON HANDS", "NEUTRAL · low conviction", "neutral"),
    }
    verdict, verdict_sub, lean = verdicts[regime]

    # confluence can veto a directional call it flatly disagrees with
    if lean == "bull" and bias < -25:
        verdict, verdict_sub = "STAND ASIDE", "structure says up, the tape says down"
        lean = "neutral"
    elif lean == "bear" and bias > 25:
        verdict, verdict_sub = "STAND ASIDE", "structure says down, the tape says up"
        lean = "neutral"

    # --- conviction ------------------------------------------------------
    conviction = 100 * (
        0.34 * dom
        + 0.22 * min(abs(bias) / 60, 1.0)
        + 0.16 * (1.0 if conf.get("aligned") else 0.4 if not conf.get("conflicted") else 0.0)
        + 0.14 * (1.0 if thin_cushion and regime in ("TRAPDOOR", "SQUEEZE FUEL") else 0.5)
        + 0.14 * min(abs(net_gex) / 5e6, 1.0)
    )
    if regime == "NO CLEAR PIN":
        conviction = min(conviction, 45)
    if conf.get("conflicted"):
        conviction *= 0.8
    conviction = int(round(max(0, min(100, conviction))))
    band = "HIGH CONVICTION" if conviction >= 70 else \
           "MEDIUM CONVICTION" if conviction >= 45 else "LOW CONVICTION"

    return {
        "regime": regime, "regime_sub": sub, "verdict": verdict,
        "verdict_sub": verdict_sub, "lean": lean,
        "conviction": conviction, "conviction_band": band,
        "dominance": round(dom, 2),
        "narrative": _narrative(regime, spot, levels, conf, cushion_pct),
        "tags": _tags(regime, levels, conf, thin_cushion),
    }


def _narrative(regime: str, spot: float, levels: dict, conf: dict, cushion_pct) -> dict:
    flip, pin = levels.get("flip"), levels.get("pin")
    floor, ceil = levels.get("floor"), levels.get("ceiling")
    fuel_dn, fuel_up = levels.get("fuel_down"), levels.get("fuel_up")
    tgt_dn, tgt_up = levels.get("target_down"), levels.get("target_up")
    macd_p = conf["macd"]["phrase"]
    rsi_p = conf["rsi"]["phrase"]
    sma_p = conf["sma200"]["phrase"]
    trend_p = conf["trend"]["phrase"]
    vol_p = conf["volume"]["phrase"]
    tape = f"Tape: {macd_p}; {rsi_p}; {trend_p}; price {sma_p}; {vol_p}."
    cush = (f"Flip cushion {cushion_pct:.1f}%" if cushion_pct is not None else "No flip in range")
    # "thinning" is a claim about a NARROW cushion -- a 10% cushion is the
    # opposite of thinning, and saying so contradicts the number beside it
    thin = cushion_pct is not None and cushion_pct < 1.5
    cush_thin = f"{cush} and thinning" if thin else cush

    if regime == "TRAPDOOR":
        see = (f"−gamma is STACKING below at ${fuel_dn['strike']} ({_m(fuel_dn['gex'])}), right under the "
               f"{_m(floor['gex'])} floor at ${floor['strike']}. {cush_thin}. Lose the floor → dealers "
               f"flip short-gamma and it cascades toward ${(tgt_dn or fuel_dn)['strike']}. "
               f"A big +floor here is a trapdoor, not safety.")
        # the cascade travels PAST the floor, not to the fuel node above it
        dest = (tgt_dn or fuel_dn)["strike"]
        play = (f"Do NOT fade the floor. Wait for a break and hold below ${floor['strike']}, then short; the "
                f"stacked −gamma accelerates it toward ${dest}.")
        invalid = (f"a reclaim back above ${floor['strike']}"
                   + (f" or the flip ${flip}" if flip else "") + " — the cushion rebuilds and the cascade is off.")
    elif regime == "SQUEEZE FUEL":
        see = (f"−gamma is stacked ABOVE at ${fuel_up['strike']} ({_m(fuel_up['gex'])}). {cush}. "
               f"Through it, dealers chase and the move feeds itself.")
        play = (f"Do NOT fade the ceiling. Take the break and hold above "
                f"${ceil['strike'] if ceil else fuel_up['strike']}; the −gamma pocket at ${fuel_up['strike']} "
                f"pulls toward ${(tgt_up or fuel_up)['strike']}.")
        invalid = (f"a rejection back below ${ceil['strike'] if ceil else fuel_up['strike']} — the fuel goes unused.")
    elif regime == "PIN MAGNET":
        see = (f"A dominant +gamma node at ${pin['strike']} ({_m(pin['gex'])}) is pinning spot. "
               f"Dealers sell strength and buy weakness into it, which compresses range. {cush}.")
        play = (f"Fade the extremes back toward ${pin['strike']}. Size small — the edge is the range, not a trend.")
        invalid = f"a decisive break away from ${pin['strike']}" + (f" or through the flip ${flip}" if flip else "")
    elif regime == "LONG GAMMA":
        see = (f"Net {_m(levels.get('net_gex'))} gamma with spot above the flip"
               f"{f' at ${flip}' if flip else ''}. Dealer hedging dampens moves — dips get absorbed. {cush}.")
        play = (f"Buy weakness toward the flip{f' ${flip}' if flip else ''} rather than chasing strength; "
                f"the first real resistance is ${ceil['strike'] if ceil else 'the next +node'}.")
        invalid = f"losing the flip{f' ${flip}' if flip else ''} — below it the dampening flips to amplification."
    elif regime == "SHORT GAMMA":
        see = (f"Spot is below the flip{f' at ${flip}' if flip else ''} with net {_m(levels.get('net_gex'))} gamma. "
               f"Dealer hedging AMPLIFIES moves here, so ranges fail and trends extend. {cush}.")
        play = ("Trade with the move, not against it. Fade nothing; let the trend run and trail the stop.")
        invalid = f"reclaiming the flip{f' ${flip}' if flip else ''} — the amplification switches off."
    else:  # NO CLEAR PIN
        see = ("No node dominates near spot — structure is contested. Low-conviction chop: scalp the range or "
               f"stand aside.{f' Pivot is the flip at ${flip}.' if flip else ''}")
        play = ("No node dominates — there's no edge. Stand aside or scalp only."
                + (f" Watch the flip ${flip}: a decisive break either way is the first real tell." if flip else ""))
        invalid = "n/a — re-gaze once a node builds or price commits to a side of the flip."

    return {"see": see + " " + tape, "play": play, "invalid": invalid}


def _tags(regime: str, levels: dict, conf: dict, thin_cushion: bool) -> list[str]:
    tags = []
    if regime == "TRAPDOOR":
        tags.append("−gamma building below in the fall direction")
    if regime == "SQUEEZE FUEL":
        tags.append("−gamma stacked above — break accelerates")
    if thin_cushion:
        tags.append("flip near spot — cushion thinning")
    if regime == "NO CLEAR PIN":
        tags.append("no node ≫ its neighbours")
    if conf.get("aligned"):
        tags.append("MACD/RSI/trend/200SMA aligned")
    if conf.get("conflicted"):
        tags.append("signals conflicted — size down")
    if conf["macd"]["fresh"]:
        tags.append(f"fresh MACD {'bull' if conf['macd']['bull'] else 'bear'} cross")
    if conf["rsi"]["zone"] in ("overbought", "oversold"):
        tags.append(f"RSI {conf['rsi']['zone']}")
    if conf["vol"]["regime"] == "compressing":
        tags.append("realized vol compressing")
    return tags[:6]
