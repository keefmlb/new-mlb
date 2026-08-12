"""Every top-N-K x top-M-TB daily parlay combination, graded.

Builds one ticket per day from the top `n_k` pitcher-strikeout picks and the
top `n_tb` total-bases picks on the simulation leaderboard, for every
combination up to 6 of each, and reports what each returned.

Only fully-priced tickets are staked — counting a ticket as stake while unable
to pay it out understates ROI. Pushes void the leg and shrink the parlay.
ROI includes the daily boost. P(>0) is a bootstrap over tickets.

Read the sample size before the ROI: these all share ~25 overlapping days, so
neighbouring cells are not independent evidence and the spread across the grid
is mostly ticket-level clustering, not a real ranking of builds.

Usage: python -m scripts.combo_grid [max_per_market]
"""
from __future__ import annotations
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict

PICKS = "data/bets/sim_picks.json"
K, TB = "prop_pitcher_k", "prop_tb"
BOOST = 0.20
BOOT = 10000



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

def _dec(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o)) if o else None


def _wilson(w: int, n: int):
    if not n:
        return (0.0, 0.0)
    p, z = w / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def evaluate(by_date, n_k: int, n_tb: int):
    rets, odds = [], []
    for d in sorted(by_date):
        day = (_top_n(by_date[d].get(K, []), n_k)
               + _top_n(by_date[d].get(TB, []), n_tb))
        if len(day) < n_k + n_tb:
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
            continue
        odds.append(dec)
        rets.append(1.0 + (dec - 1.0) * (1.0 + BOOST)
                    if all(e["outcome"] == "W" for e in live) else 0.0)
    return rets, odds


def main(mx: int = 6) -> None:
    rows = [e for e in json.load(open(PICKS, encoding="utf-8"))
            if e.get("outcome") in ("W", "L", "P") and e.get("market") in (K, TB)]
    by: dict = defaultdict(lambda: defaultdict(list))
    for e in rows:
        by[e["date"]][e["market"]].append(e)

    out = []
    for n_k in range(0, mx + 1):
        for n_tb in range(0, mx + 1):
            if n_k + n_tb == 0:
                continue
            rets, odds = evaluate(by, n_k, n_tb)
            n = len(rets)
            if not n:
                continue
            won = sum(1 for r in rets if r > 0)
            roi = sum(rets) / n - 1.0
            random.seed(7)
            boot = sorted(sum(random.choice(rets) for _ in range(n)) / n - 1.0
                          for _ in range(BOOT))
            lo, hi = _wilson(won, n)
            out.append({
                "k": n_k, "tb": n_tb, "legs": n_k + n_tb, "tix": n, "won": won,
                "hit": won / n, "cilo": lo, "cihi": hi,
                "med": st.median(odds), "roi": roi,
                "ppos": sum(1 for b in boot if b > 0) / BOOT,
                "profit": sum(rets) - n,
            })

    hdr = (f"{'build':>9s} {'legs':>4s} {'tix':>4s} {'W':>3s} {'hit%':>6s} "
           f"{'hit 95% CI':>14s} {'median':>8s} {'ROI':>9s} {'P(>0)':>6s} {'units':>7s}")

    print(f"ALL COMBINATIONS — top N pitcher K x top M total bases, "
          f"1 ticket/day\n{'=' * 92}\n")
    print("BY TOTAL LEGS\n")
    print(hdr)
    for legs in range(1, 2 * mx + 1):
        grp = [r for r in out if r["legs"] == legs]
        if not grp:
            continue
        print(f"  -- {legs} leg{'s' if legs > 1 else ''} --")
        for r in sorted(grp, key=lambda r: -r["roi"]):
            print(f"{f'{r[chr(107)]}K+{r[chr(116)+chr(98)]}TB':>9s} {r['legs']:4d} "
                  f"{r['tix']:4d} {r['won']:3d} {r['hit']*100:5.1f}% "
                  f"[{r['cilo']*100:4.0f}%,{r['cihi']*100:4.0f}%] "
                  f"{'+' + str(int((r['med']-1)*100)):>8s} {r['roi']:+8.1%} "
                  f"{r['ppos']:6.2f} {r['profit']:+7.2f}")

    print(f"\n{'=' * 92}\nTOP 12 BY ROI\n")
    print(hdr)
    for r in sorted(out, key=lambda r: -r["roi"])[:12]:
        print(f"{f'{r[chr(107)]}K+{r[chr(116)+chr(98)]}TB':>9s} {r['legs']:4d} "
              f"{r['tix']:4d} {r['won']:3d} {r['hit']*100:5.1f}% "
              f"[{r['cilo']*100:4.0f}%,{r['cihi']*100:4.0f}%] "
              f"{'+' + str(int((r['med']-1)*100)):>8s} {r['roi']:+8.1%} "
              f"{r['ppos']:6.2f} {r['profit']:+7.2f}")

    print(f"\n{'=' * 92}\nROI GRID  (rows = K legs, cols = TB legs)\n")
    print("      " + "".join(f"{m:>9d}TB" for m in range(0, mx + 1)))
    for n_k in range(0, mx + 1):
        cells = []
        for n_tb in range(0, mx + 1):
            r = next((x for x in out if x["k"] == n_k and x["tb"] == n_tb), None)
            cells.append(f"{r['roi']*100:+10.0f}%" if r else f"{'-':>11s}")
        print(f"{n_k:3d}K  " + "".join(cells))

    n_tix = out[0]["tix"] if out else 0
    print(f"\n  Every cell is ~{n_tix} overlapping days. Neighbouring cells share "
          f"most of their\n  legs, so this grid is ONE sample viewed many ways — "
          f"not {len(out)} independent tests.\n  Picking the max here is fitting "
          f"to noise; use it to see the shape, not to choose.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
