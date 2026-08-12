"""Backtest the ACTUAL daily strategy: one parlay per market, built to ~+500,
with a 20% profit boost.

The user does not bet the sim leaderboard as singles. Each day they take the
top-ranked legs of a market (total bases, pitcher strikeouts), stack them until
the combined price reaches roughly +500, and apply a 20% profit boost to the
ticket. One ticket per market per day, flat stake.

That is a very different bet from the singles analysis: a parlay wins only if
EVERY leg wins, so the ~9pp per-leg overconfidence compounds multiplicatively.
This replays the real graded record (data/bets/sim_picks.json, post-purge) to
measure it directly instead of inferring it from per-leg numbers.

Legs are taken in the sim's own rank order (rank 1 first), which is how the
board presents them. A leg is only usable if it has a real American price and a
graded W/L.

Run:  python -m scripts.daily_parlay_audit
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bet_tracker import roi_ci  # noqa: E402

PICKS = ROOT / "data" / "bets" / "sim_picks.json"
BOOST = 0.20          # profit boost applied to the ticket


def _dec(a) -> float:
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))


def _american(d: float) -> int:
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def _usable(r) -> bool:
    o = r.get("odds")
    return (r.get("outcome") in ("W", "L")
            and isinstance(o, (int, float)) and o and not (-100 < o < 100))


def _build(legs: list[dict], target_dec: float, max_legs: int = 12):
    """Stack rank-ordered legs until the price reaches target. Returns the legs
    used, or None if the pool can't get there."""
    used, dec = [], 1.0
    for r in legs:
        used.append(r)
        dec *= _dec(r["odds"])
        if dec >= target_dec:
            return used, dec
        if len(used) >= max_legs:
            break
    return (used, dec) if used else (None, None)


def _run(rows, markets, target_dec, pool_depth, boost):
    by_day = defaultdict(list)
    for r in rows:
        if r["market"] in markets and (r.get("rank") or 99) <= pool_depth:
            by_day[(r["date"], r["market"])].append(r)
    n = wins = 0
    profits = []
    legs_used = []
    for key in sorted(by_day):
        legs = sorted(by_day[key], key=lambda r: (r.get("rank") or 99))
        used, dec = _build(legs, target_dec)
        if not used or len(used) < 2:
            continue
        n += 1
        legs_used.append(len(used))
        if all(r["outcome"] == "W" for r in used):
            wins += 1
            profits.append((dec - 1.0) * (1.0 + boost))
        else:
            profits.append(-1.0)
    if not n:
        return None
    roi = sum(profits) / n
    return {"n": n, "wins": wins, "hit": wins / n, "roi": roi,
            "ci": roi_ci(profits), "bank": sum(profits),
            "avg_legs": sum(legs_used) / len(legs_used)}


def _line(label, res):
    if not res:
        print(f"  {label:34s} (no tickets)")
        return
    ci = res["ci"]
    cis = f"[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
    print(f"  {label:34s} tickets {res['n']:3d}  legs {res['avg_legs']:.1f}  "
          f"hit {res['hit']:5.1%} ({res['wins']:2d}W)  ROI {res['roi']:+7.1%} "
          f"{cis:18s} bank {res['bank']:+6.1f}u")


def main():
    data = json.loads(PICKS.read_text(encoding="utf-8"))
    rows = [r for r in data if _usable(r)]
    print(f"Graded, priced legs available: {len(rows)}")

    targets = {"+300": 4.0, "+500": 6.0, "+700": 8.0}
    print("\n" + "=" * 96)
    print(f"YOUR STRATEGY — one ticket/day/market, {int(BOOST*100)}% boost, "
          "legs taken in sim rank order")
    print("=" * 96)
    for mkts, label in ((("prop_tb",), "TB only"),
                        (("prop_pitcher_k",), "Pitcher K only")):
        print(f"\n{label}")
        for tname, tdec in targets.items():
            for depth in (3, 5, 8):
                _line(f"target {tname}, pool top-{depth}",
                      _run(rows, mkts, tdec, depth, BOOST))

    print("\n" + "=" * 96)
    print("BOOST VALUE — same tickets, with and without the 20%")
    print("=" * 96)
    for mkts, label in ((("prop_tb",), "TB"), (("prop_pitcher_k",), "K")):
        for b, bl in ((0.0, "no boost"), (BOOST, "20% boost")):
            _line(f"{label} +500 top-5, {bl}",
                  _run(rows, mkts, 6.0, 5, b))

    print("\nREAD: a parlay needs EVERY leg to win, so the ~9pp per-leg "
          "overconfidence\ncompounds. Breakeven at +500 is 16.7% (14.3% with "
          "the boost).")


if __name__ == "__main__":
    main()
