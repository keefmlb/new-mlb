"""How to split a fixed-size daily parlay between pitcher K and total bases.

Holds the leg count constant and slides the mix, so the comparison is about
COMPOSITION rather than ticket length. Both effects that matter pull in
opposite directions:

  - K legs hit more often AND move together (a high-strikeout slate lifts every
    pitcher on it), so K-heavy tickets beat their independent-leg price.
  - But each board degrades with depth, so a K-heavy ticket has to reach down
    to worse K ranks, while a balanced one buys only the top of both.

Only fully-priced tickets are staked. Pushes void the leg.

Usage: python -m scripts.split_sweep [total_legs]
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


def _wilson(w: int, n: int):
    if not n:
        return (0.0, 0.0)
    p, z = w / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def evaluate(by_date, n_k: int, n_tb: int):
    rets, odds, kw, kl, tw, tl = [], [], 0, 0, 0, 0
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
        for e in live:
            win = e["outcome"] == "W"
            if e["market"] == K:
                kw, kl = kw + win, kl + (not win)
            else:
                tw, tl = tw + win, tl + (not win)
        rets.append(1.0 + (dec - 1.0) * (1.0 + BOOST)
                    if all(e["outcome"] == "W" for e in live) else 0.0)
    return rets, odds, (kw, kl), (tw, tl)


def main(total: int = 6) -> None:
    rows = [e for e in json.load(open(PICKS, encoding="utf-8"))
            if e.get("outcome") in ("W", "L", "P") and e.get("market") in (K, TB)]
    by_date: dict = defaultdict(lambda: defaultdict(list))
    for e in rows:
        by_date[e["date"]][e["market"]].append(e)

    print(f"SPLIT SWEEP — {total} legs total, one ticket/day\n")
    print(f"{'build':>10s} {'tix':>4s} {'hit':>4s} {'hit%':>6s} {'exp%':>6s} "
          f"{'legs%':>6s} {'median':>7s} {'ROI':>9s} {'P(>0)':>6s}  {'95% CI on ROI':>22s}")
    for n_k in range(total, -1, -1):
        n_tb = total - n_k
        rets, odds, (kw, kl), (tw, tl) = evaluate(by_date, n_k, n_tb)
        n = len(rets)
        if not n:
            continue
        won = sum(1 for r in rets if r > 0)
        legs_w, legs_n = kw + tw, kw + kl + tw + tl
        lr = legs_w / legs_n if legs_n else 0.0
        random.seed(7)
        boot = sorted(sum(random.choice(rets) for _ in range(n)) / n - 1.0
                      for _ in range(BOOT))
        p_pos = sum(1 for b in boot if b > 0) / BOOT
        roi = sum(rets) / n - 1.0
        print(f"{f'{n_k}K+{n_tb}TB':>10s} {n:4d} {won:4d} {won/n*100:5.1f}% "
              f"{lr**total*100:5.1f}% {lr*100:5.1f}% "
              f"{'+' + str(int((st.median(odds)-1)*100)):>7s} {roi:+8.1%} "
              f"{p_pos:6.2f}  [{boot[int(.025*BOOT)]:+.0%}, {boot[int(.975*BOOT)]:+.0%}]"
              f"   K {kw}-{kl} TB {tw}-{tl}")

    print("\n  exp% = what the observed leg rate would give if legs were "
          "INDEPENDENT.\n  hit% above exp% means the legs move together, which "
          "books do not price.")
    print("\n  READ THE legs% COLUMN FIRST. If it barely moves across builds, "
          "every split is\n  buying the same quality of legs and the ROI spread "
          "is ticket-level clustering\n  on a small sample, NOT evidence that one "
          "mix is better. That is the case at\n  6 legs: the marginal legs being "
          "swapped (K ranks 4-6 avg 74.7%, TB ranks 1-3\n  avg 73.7%) hit at "
          "nearly the same rate, so the leg pool is unchanged.")
    # leg quality by depth, to show what a heavier split has to buy
    print("\n  leg hit rate by rank (what each extra leg costs you):")
    for mkt, lab in ((K, "K "), (TB, "TB")):
        cells = []
        for rk in range(1, total + 1):
            sel = [e for e in rows if e.get("market") == mkt
                   and e.get("rank") == rk and e["outcome"] in ("W", "L")]
            if sel:
                w = sum(1 for e in sel if e["outcome"] == "W")
                cells.append(f"r{rk} {w/len(sel)*100:.0f}%")
        print(f"    {lab}  " + "  ".join(cells))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
