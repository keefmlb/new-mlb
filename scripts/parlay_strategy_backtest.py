"""Backtest a concrete sim-parlay strategy on the graded sim-pick record.

Replays the user's actual rules on real logged sim picks + graded outcomes
(data/bets/sim_picks.json — each pick carries sim_hit, offered odds, W/L):

  * Only legs with sim_hit >= CONF (default 0.85).
  * Build parlays targeting ~ TARGET American odds (default +500): sort the
    day's eligible legs by sim confidence, greedily add legs until the combined
    price reaches the target, then start the next parlay. Leftover legs form a
    final (shorter) parlay.
  * Flat 1u stake per parlay. A parlay wins only if EVERY leg won (graded from
    actual boxscores, so real in-game correlation is already baked into whether
    it hit — no independence assumption on the OUTCOME; the PAYOUT uses the
    product of decimal odds, standard parlay math).
  * Daily boosts: one 20% and one 10% profit boost, applied to that day's two
    biggest-payout parlays (rational use of the better boost on the bigger tab).

Reports realized hit rate, ROI (with and without the boosts to isolate their
value), and the running bankroll — plus a small grid over CONF / TARGET and a
cross-game-only variant (one leg per game, which removes same-game correlation
and matches books that won't allow those legs together).

CAVEAT: sample is small (the sim-pick record only spans a couple of weeks) and
same-game parlays may pay less at a real book than the multiplied price used
here. Read the direction, not the decimal.

Run:  python -m scripts.parlay_strategy_backtest
"""
from __future__ import annotations
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICKS = ROOT / "data" / "bets" / "sim_picks.json"


def _dec(o) -> float | None:
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if not o:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def _american(dec: float) -> int:
    if dec <= 1:
        return 0
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def _build_parlays(legs: list[dict], target_dec: float, max_legs: int = 8,
                   one_per_game: bool = False) -> list[list[dict]]:
    """Greedily bucket confidence-sorted legs into ~target-priced parlays."""
    legs = sorted(legs, key=lambda p: -(p.get("sim_hit") or 0))
    if one_per_game:
        seen, uniq = set(), []
        for p in legs:
            g = p.get("game_pk")
            if g in seen:
                continue
            seen.add(g); uniq.append(p)
        legs = uniq
    parlays, cur, cur_dec = [], [], 1.0
    for p in legs:
        d = _dec(p.get("odds"))
        if d is None:
            continue
        cur.append(p); cur_dec *= d
        if cur_dec >= target_dec or len(cur) >= max_legs:
            parlays.append(cur); cur, cur_dec = [], 1.0
    if cur:
        parlays.append(cur)
    return parlays


def _settle(parlay: list[dict]) -> tuple[float, bool]:
    """Return (decimal_price, won) for a parlay from graded leg outcomes."""
    dec = 1.0
    won = True
    for p in parlay:
        d = _dec(p.get("odds"))
        dec *= d if d else 1.0
        if p.get("outcome") != "W":
            won = False
    return dec, won


def _run(picks: list[dict], conf: float, target_dec: float,
         one_per_game: bool, use_boosts: bool) -> dict:
    by_date: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        if p.get("outcome") in ("W", "L") and (p.get("sim_hit") or 0) >= conf:
            by_date[p["date"]].append(p)

    n_par = wins = 0
    staked = profit = 0.0
    bankroll = 0.0
    curve = []
    for date in sorted(by_date):
        parlays = _build_parlays(by_date[date], target_dec, one_per_game=one_per_game)
        # rank parlays by potential payout so boosts land on the biggest tabs
        settled = [(_settle(par)) for par in parlays]
        order = sorted(range(len(parlays)), key=lambda i: -settled[i][0])
        boosts = [0.20, 0.10] if use_boosts else []
        boost_for = {}
        for k, idx in enumerate(order):
            boost_for[idx] = boosts[k] if k < len(boosts) else 0.0
        for i, par in enumerate(parlays):
            dec, won = settled[i]
            n_par += 1
            staked += 1.0
            if won:
                wins += 1
                pr = (dec - 1.0) * (1.0 + boost_for.get(i, 0.0))
                profit += pr; bankroll += pr
            else:
                profit -= 1.0; bankroll -= 1.0
        curve.append((date, round(bankroll, 2)))
    roi = profit / staked if staked else 0.0
    return {"parlays": n_par, "wins": wins,
            "hit": wins / n_par if n_par else 0.0,
            "staked": staked, "profit": profit, "roi": roi,
            "bankroll": bankroll, "curve": curve}


def main():
    if not PICKS.exists():
        print("No sim_picks.json yet."); return
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    graded = [p for p in picks if p.get("outcome") in ("W", "L")]
    hi = [p for p in graded if (p.get("sim_hit") or 0) >= 0.85]
    dates = sorted({p["date"] for p in hi})
    print(f"Graded sim picks: {len(graded)}   >=85%: {len(hi)}   "
          f"dates: {len(dates)} ({dates[0]}..{dates[-1]})" if dates else "no >=85% picks")

    print("\n=== YOUR RULE: >=85% conf, target +500, 20%+10% daily boosts ===")
    base = _run(picks, 0.85, 6.0, one_per_game=False, use_boosts=False)
    boosted = _run(picks, 0.85, 6.0, one_per_game=False, use_boosts=True)
    print(f"  no boosts : {base['wins']}/{base['parlays']} won "
          f"({base['hit']:.0%})   ROI {base['roi']:+.1%}   "
          f"bankroll {base['bankroll']:+.2f}u")
    print(f"  w/ boosts : {boosted['wins']}/{boosted['parlays']} won "
          f"({boosted['hit']:.0%})   ROI {boosted['roi']:+.1%}   "
          f"bankroll {boosted['bankroll']:+.2f}u")
    print("  bankroll by day (boosted):",
          "  ".join(f"{d[5:]}:{b:+.1f}" for d, b in boosted["curve"]))

    print("\n=== GRID (with boosts; one_per_game removes same-game correlation) ===")
    print(f"  {'conf':>5} {'target':>7} {'1/gm':>5} {'par':>4} {'hit':>5} "
          f"{'ROI':>7} {'bank':>7}")
    for conf in (0.85, 0.88, 0.90):
        for tgt_am, tgt_dec in (("+300", 4.0), ("+500", 6.0), ("+700", 8.0)):
            for opg in (False, True):
                r = _run(picks, conf, tgt_dec, one_per_game=opg, use_boosts=True)
                if not r["parlays"]:
                    continue
                print(f"  {conf:>5.2f} {tgt_am:>7} {str(opg):>5} {r['parlays']:>4} "
                      f"{r['hit']:>4.0%} {r['roi']:>+7.1%} {r['bankroll']:>+7.2f}")

    print("\nRead: ROI here counts the boosts. 'bank' is cumulative units at 1u/"
          "parlay.\nSmall sample — treat as a first read, not proof. one_per_game="
          "True is the\nhonest cross-game number (no same-game correlation, and "
          "bookable anywhere).")


if __name__ == "__main__":
    main()
