"""Which ranking rule actually puts the best bets in the top 10?

The sim leaderboard currently ranks purely by `sim_hit`, which by construction
surfaces the heaviest chalk — not necessarily the most profitable picks. The
graded record (data/bets/sim_picks.json, post-purge) is now large enough to
test alternatives directly: re-rank each day's pool by a candidate rule, take
the daily top-N per market, and measure realized hit rate and ROI.

This is the sim-board counterpart of scripts/score_formula_search.py.

Signals available per pick: sim_hit (model prob), implied prob from the offered
price, their difference (edge), and the payout. Candidates test the three real
hypotheses: rank by confidence, rank by edge vs the market, or rank by a blend.

Honest caveats printed with the results — the same pick pool is used to both
choose and evaluate, so a winner needs a clear margin, not a hair.

Run:  python -m scripts.sim_rank_search
"""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bet_tracker import roi_ci  # noqa: E402

PICKS = ROOT / "data" / "bets" / "sim_picks.json"
MARKETS = ("prop_tb", "prop_pitcher_k")


def _prob(a):
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)


def _dec(a):
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))


def _usable(r):
    o = r.get("odds")
    return (r.get("outcome") in ("W", "L") and r.get("sim_hit") is not None
            and isinstance(o, (int, float)) and o and not (-100 < o < 100))


# ---- candidate ranking rules (higher = bet first) ----
def _conf(r):        return r["sim_hit"]                     # current rule
def _edge(r):        return r["sim_hit"] - r["imp"]          # model vs price
def _edge_x_conf(r): return (r["sim_hit"] - r["imp"]) * r["sim_hit"]
def _ev(r):          return r["sim_hit"] * _dec(r["odds"]) - 1.0
def _kelly(r):
    b = _dec(r["odds"]) - 1.0
    p = r["sim_hit"]
    return max(0.0, (p * b - (1 - p)) / b) if b > 0 else 0.0
def _chalk(r):       return r["imp"]                         # follow the price
def _conf_shrunk(r):
    # confidence with the measured ~9pp overconfidence removed, then edge
    return (r["sim_hit"] - 0.09) - r["imp"]


CANDIDATES = {
    "sim_hit (CURRENT)":   _conf,
    "edge = sim - implied": _edge,
    "edge x conf":          _edge_x_conf,
    "EV per $":             _ev,
    "Kelly":                _kelly,
    "calibrated edge (-9p)": _conf_shrunk,
    "price only (chalk)":   _chalk,
}


def _eval(rows, rank_fn, top_n):
    by_day = defaultdict(list)
    for r in rows:
        by_day[(r["date"], r["market"])].append(r)
    profits, wins, n = [], 0, 0
    for key, pool in by_day.items():
        sel = sorted(pool, key=rank_fn, reverse=True)[:top_n]
        for r in sel:
            n += 1
            if r["outcome"] == "W":
                wins += 1
                profits.append(_dec(r["odds"]) - 1.0)
            else:
                profits.append(-1.0)
    if not n:
        return None
    return {"n": n, "hit": wins / n, "roi": sum(profits) / n,
            "ci": roi_ci(profits)}


def main():
    data = json.loads(PICKS.read_text(encoding="utf-8"))
    rows = []
    for r in data:
        if not _usable(r) or r.get("market") not in MARKETS:
            continue
        r = dict(r)
        r["imp"] = _prob(r["odds"])
        rows.append(r)
    print(f"Graded, priced {'/'.join(MARKETS)} picks: {len(rows)}")

    for top_n in (3, 5, 10):
        print("\n" + "=" * 88)
        print(f"DAILY TOP-{top_n} PER MARKET, BY RANKING RULE")
        print("=" * 88)
        res = []
        for name, fn in CANDIDATES.items():
            out = _eval(rows, fn, top_n)
            if out:
                res.append((name, out))
        res.sort(key=lambda x: -x[1]["roi"])
        for name, o in res:
            ci = o["ci"]
            cis = f"[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
            star = "  <--" if name.startswith("sim_hit") else ""
            print(f"  {name:24s} n={o['n']:4d}  hit {o['hit']:5.1%}  "
                  f"ROI {o['roi']:+7.1%}  {cis}{star}")

    print("\nCAVEAT: rules are chosen and scored on the same pool, so the top "
          "line is\noptimistic. Only adopt a rule that wins at MULTIPLE top-N "
          "levels by a clear\nmargin — a narrow win is noise.")


if __name__ == "__main__":
    main()
