"""All-time record for the top-N sim leaderboard picks, PER MARKET.

The user plays two tickets a day: the top N total-bases picks as one parlay,
and the top N pitcher-strikeout picks as another. They are separate bets with
very different price structures, so pooling them hides both.

Grades three levels:
  LEG    — how often one top-N pick wins.
  TICKET — how often ALL N legs win, which is what actually pays.
  ROI    — with the daily boost, bootstrapped for a confidence interval.

Only tickets whose every leg carries a usable price are staked; counting a
ticket as stake while being unable to pay it out understates ROI.
Pushes void the leg and shrink the parlay rather than losing it.

Usage: python -m scripts.top6_record [n_legs]
"""
from __future__ import annotations
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict

PICKS = "data/bets/sim_picks.json"
MARKETS = {"prop_pitcher_k": "PITCHER K", "prop_tb": "TOTAL BASES"}
BOOST = 0.20
BOOT = 20000



def _top_n(rows: list[dict], n: int) -> list[dict]:
    """Top `n` picks AS THE APP RANKED THEM.

    Uses the stored `rank` — the position sim_tracker logged off the already
    diversified board. Re-sorting on sim_hit invents an order the user never
    saw: the two disagree on 27% of date-market top-3 sets, because
    _diversify_lines reorders before logging. Rows are deduped on rank first,
    since several board views (leaderboard / builder75) log the same date and
    can repeat a rank.
    """
    seen, out = set(), []
    for e in sorted(rows, key=lambda r: ((r.get("rank") is None),
                                         r.get("rank") or 999,
                                         -(r.get("sim_hit") or 0))):
        r = e.get("rank")
        if r is not None:
            if r in seen:
                continue
            seen.add(r)
        out.append(e)
        if len(out) >= n:
            break
    return out

def _dec(odds) -> float | None:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if not o:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def _wilson(w: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, z = w / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def _report(label: str, rows: list[dict], n_legs: int) -> None:
    by_date: dict[str, list] = defaultdict(list)
    for e in rows:
        by_date[e["date"]].append(e)

    print(f"\n{'=' * 66}\n{label} — top {n_legs} each day\n{'=' * 66}")

    # leg level by rank
    print(f"\n  {'rank':>4s} {'W':>4s} {'L':>4s} {'hit%':>7s} {'95% CI':>17s}")
    tw = tl = 0
    for rk in range(1, n_legs + 1):
        sel = [e for e in rows if e.get("rank") == rk]
        w = sum(1 for e in sel if e["outcome"] == "W")
        l = sum(1 for e in sel if e["outcome"] == "L")
        if not (w + l):
            continue
        tw += w
        tl += l
        lo, hi = _wilson(w, w + l)
        print(f"  {rk:4d} {w:4d} {l:4d} {w/(w+l)*100:6.1f}%   "
              f"[{lo*100:4.1f}%, {hi*100:5.1f}%]")
    if tw + tl:
        lo, hi = _wilson(tw, tw + tl)
        print(f"   all {tw:4d} {tl:4d} {tw/(tw+tl)*100:6.1f}%   "
              f"[{lo*100:4.1f}%, {hi*100:5.1f}%]  <- leg rate")

    # ticket level
    rets: list[float] = []
    odds: list[float] = []
    dist: dict[int, int] = defaultdict(int)
    skipped = 0
    for d in sorted(by_date):
        day = _top_n(by_date[d], n_legs)
        if len(day) < n_legs:
            continue
        live = [e for e in day if e["outcome"] in ("W", "L")]
        if not live:
            continue
        dec, ok = 1.0, True
        for e in live:
            dv = _dec(e.get("odds"))
            if dv is None:
                ok = False
                break
            dec *= dv
        if not ok:                      # unpriced: cannot stake or pay it
            skipped += 1
            continue
        odds.append(dec)
        nw = sum(1 for e in live if e["outcome"] == "W")
        dist[nw] += 1
        rets.append(1.0 + (dec - 1.0) * (1.0 + BOOST)
                    if nw == len(live) else 0.0)

    n = len(rets)
    if not n:
        print("\n  no fully-priced tickets")
        return
    won = sum(1 for r in rets if r > 0)
    lo, hi = _wilson(won, n)
    roi = sum(rets) / n - 1.0
    random.seed(7)
    boot = sorted(sum(random.choice(rets) for _ in range(n)) / n - 1.0
                  for _ in range(BOOT))
    p_pos = sum(1 for b in boot if b > 0) / BOOT
    print(f"\n  tickets {n} (+{skipped} unpriced, excluded)   "
          f"hit {won} = {won/n*100:.1f}%  [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"  median ticket odds  +{(st.median(odds)-1)*100:.0f}")
    print(f"  ROI with {int(BOOST*100)}% boost  {roi:+.1%}   "
          f"95% CI [{boot[int(.025*BOOT)]:+.1%}, {boot[int(.975*BOOT)]:+.1%}]"
          f"   P(>0) {p_pos:.2f}")
    print("  legs won per ticket: " + "  ".join(
        f"{k}/{n_legs}:{dist[k]}" for k in sorted(dist, reverse=True)))


def _report_combined(allrows: list[dict], per_market: int) -> None:
    """ONE ticket a day taking the top `per_market` from EACH market.

    Reported separately from the single-market tickets because the correlation
    structure differs: strikeout legs move together (a high-K slate lifts every
    pitcher on it), total-bases legs behave independently. Mixing them dilutes
    the K correlation that makes those parlays beat their independent price,
    so this is a real trade-off rather than a strictly better build.
    """
    by_date: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in allrows:
        if e.get("market") in MARKETS:
            by_date[e["date"]][e["market"]].append(e)

    n_legs = per_market * len(MARKETS)
    print(f"\n{'=' * 66}\nCOMBINED — top {per_market} of EACH market, "
          f"one {n_legs}-leg ticket/day\n{'=' * 66}")

    rets: list[float] = []
    odds: list[float] = []
    dist: dict[int, int] = defaultdict(int)
    lw = ll = 0
    per_mkt_legs: dict[str, list[int]] = {m: [0, 0] for m in MARKETS}
    skipped = 0
    for d in sorted(by_date):
        day: list[dict] = []
        for mkt in MARKETS:
            day += _top_n(by_date[d].get(mkt, []), per_market)
        if len(day) < n_legs:
            continue
        live = [e for e in day if e["outcome"] in ("W", "L")]
        if not live:
            continue
        dec, ok = 1.0, True
        for e in live:
            dv = _dec(e.get("odds"))
            if dv is None:
                ok = False
                break
            dec *= dv
        if not ok:
            skipped += 1
            continue
        odds.append(dec)
        nw = sum(1 for e in live if e["outcome"] == "W")
        dist[nw] += 1
        lw += nw
        ll += len(live) - nw
        for e in live:
            per_mkt_legs[e["market"]][0 if e["outcome"] == "W" else 1] += 1
        rets.append(1.0 + (dec - 1.0) * (1.0 + BOOST)
                    if nw == len(live) else 0.0)

    n = len(rets)
    if not n:
        print("\n  no fully-priced tickets")
        return
    won = sum(1 for r in rets if r > 0)
    lo, hi = _wilson(won, n)
    leg_rate = lw / (lw + ll) if (lw + ll) else 0.0
    random.seed(7)
    boot = sorted(sum(random.choice(rets) for _ in range(n)) / n - 1.0
                  for _ in range(BOOT))
    p_pos = sum(1 for b in boot if b > 0) / BOOT
    print(f"\n  tickets {n} (+{skipped} unpriced)   hit {won} = {won/n*100:.1f}%"
          f"  [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"  leg rate {lw}-{ll} = {leg_rate*100:.1f}%   "
          f"independent expectation {leg_rate**n_legs*100:.1f}%")
    for mkt, label in MARKETS.items():
        w, l = per_mkt_legs[mkt]
        if w + l:
            print(f"    {label:12s} legs {w}-{l} = {w/(w+l)*100:.1f}%")
    print(f"  median ticket odds  +{(st.median(odds)-1)*100:.0f}")
    print(f"  ROI with {int(BOOST*100)}% boost  {sum(rets)/n-1:+.1%}   "
          f"95% CI [{boot[int(.025*BOOT)]:+.1%}, {boot[int(.975*BOOT)]:+.1%}]"
          f"   P(>0) {p_pos:.2f}")
    print("  legs won per ticket: " + "  ".join(
        f"{k}/{n_legs}:{dist[k]}" for k in sorted(dist, reverse=True)))


def main(n_legs: int = 6) -> None:
    allrows = [e for e in json.load(open(PICKS, encoding="utf-8"))
               if e.get("outcome") in ("W", "L", "P")]
    for mkt, label in MARKETS.items():
        _report(label, [e for e in allrows if e.get("market") == mkt], n_legs)
    if n_legs % len(MARKETS) == 0:
        _report_combined(allrows, n_legs // len(MARKETS))
    print("\nNote: every number here predates the 2026-08-10-hit-mix sim "
          "rework.\nIt grades the picks the OLD simulator produced.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
