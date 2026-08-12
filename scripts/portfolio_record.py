"""Running several daily tickets at once: what the PORTFOLIO returns.

Three tickets a day, one unit each: top-N pitcher K, top-N total bases, and the
top-(N/2) of each combined.

The three overlap heavily — the 3+3 ticket is built from legs that also sit in
both top-6 boards — so they are NOT three independent bets. A good slate carries
all three and a bad one kills all three. The bootstrap therefore resamples whole
DAYS, keeping a day's three tickets together; resampling individual tickets
would treat correlated bets as independent and make the interval far too tight.

Only days where all three tickets are fully priced are counted, so every build
is graded on the same slate.

Usage: python -m scripts.portfolio_record [n_legs]
"""
from __future__ import annotations
import json
import math
import random
import sys
from collections import defaultdict

PICKS = "data/bets/sim_picks.json"
K, TB = "prop_pitcher_k", "prop_tb"
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

def _dec(odds):
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o)) if o else None


def _ticket(legs: list[dict]) -> float | None:
    """Return per 1 unit staked, or None if any leg lacks a price."""
    live = [e for e in legs if e["outcome"] in ("W", "L")]
    if not live:
        return None
    dec = 1.0
    for e in live:
        dv = _dec(e.get("odds"))
        if dv is None:
            return None
        dec *= dv
    if all(e["outcome"] == "W" for e in live):
        return 1.0 + (dec - 1.0) * (1.0 + BOOST)
    return 0.0


def main(n: int = 6) -> None:
    rows = [e for e in json.load(open(PICKS, encoding="utf-8"))
            if e.get("outcome") in ("W", "L", "P") and e.get("market") in (K, TB)]
    by: dict = defaultdict(lambda: defaultdict(list))
    for e in rows:
        by[e["date"]][e["market"]].append(e)

    half = n // 2
    days: list[tuple[str, dict]] = []
    for d in sorted(by):
        ks = _top_n(by[d].get(K, []), 99)
        ts = _top_n(by[d].get(TB, []), 99)
        if len(ks) < n or len(ts) < n:
            continue
        tick = {
            f"K top {n}": _ticket(ks[:n]),
            f"TB top {n}": _ticket(ts[:n]),
            f"{half}+{half} combined": _ticket(ks[:half] + ts[:half]),
        }
        if any(v is None for v in tick.values()):
            continue
        days.append((d, tick))

    if not days:
        print("no days with all three tickets priced")
        return
    names = list(days[0][1].keys())

    print(f"PORTFOLIO — {len(names)} tickets/day, 1 unit each, "
          f"{len(days)} qualifying days\n")
    print(f"{'ticket':>20s} {'hit':>5s} {'hit%':>7s} {'ROI':>9s}")
    for nm in names:
        rs = [t[nm] for _, t in days]
        w = sum(1 for r in rs if r > 0)
        print(f"{nm:>20s} {w:5d} {w/len(rs)*100:6.1f}% "
              f"{sum(rs)/len(rs)-1:+8.1%}")

    per_day = [sum(t[nm] for nm in names) for _, t in days]
    stake = float(len(names))
    total_ret = sum(per_day)
    total_stake = stake * len(days)
    roi = total_ret / total_stake - 1.0

    random.seed(7)
    boot = sorted(
        sum(random.choice(per_day) for _ in range(len(days)))
        / (stake * len(days)) - 1.0
        for _ in range(BOOT))
    p_pos = sum(1 for b in boot if b > 0) / BOOT

    print(f"\n  staked {total_stake:.0f} units over {len(days)} days "
          f"({len(names)}/day)")
    print(f"  returned {total_ret:.2f}   PORTFOLIO ROI {roi:+.1%}")
    print(f"  95% CI (resampled by DAY) [{boot[int(.025*BOOT)]:+.1%}, "
          f"{boot[int(.975*BOOT)]:+.1%}]   P(>0) {p_pos:.2f}")
    print(f"  profit {total_ret - total_stake:+.2f} units "
          f"= {(total_ret-total_stake)/len(days):+.3f}/day")

    hits = defaultdict(int)
    for _, t in days:
        hits[sum(1 for nm in names if t[nm] > 0)] += 1
    print("\n  tickets hitting on the same day: " + "  ".join(
        f"{k}/{len(names)}:{hits[k]}" for k in sorted(hits, reverse=True)))
    blank = hits[0]
    print(f"  blank days {blank}/{len(days)} = {blank/len(days)*100:.0f}%"
          f"   (-{stake:.0f} units each)")

    # worst drawdown on a 1-unit-per-ticket flat stake
    bal = run = 0.0
    for r in per_day:
        bal += r - stake
        run = min(run, bal)
    print(f"  worst cumulative drawdown {run:+.2f} units")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
