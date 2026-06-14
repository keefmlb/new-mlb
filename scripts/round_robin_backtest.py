"""Does betting our top plays as a ROUND ROBIN lead to profit?

A round robin of N selections "by k" places all C(N,k) k-leg parlays. The
honest prior from probability theory: a parlay's EV is the product of its
legs' EV multipliers, so round-robining legs that are individually −EV is
MORE −EV than singles, not less — round robins spread variance, they don't
manufacture edge. Correlated legs (same game/day) make it worse. This script
checks that against our ACTUAL graded history rather than asserting it.

Data: data/odds/replay_archive.csv — every floor-clearing full-board bet,
graded W/L/P, with odds. For each day we take the top-N plays by score (our
ranking) and compare, on realized outcomes:
  - singles  (flat $1 each)
  - round robin by 2  (all pairs as 2-leg parlays, $1 each)
  - round robin by 3  (all triples, $1 each)
Parlay grading: any leg L ⇒ parlay loses; push legs drop out (decimal 1.0);
all-push ⇒ stake returned. ROI = profit / total staked, with a bootstrap CI.

Also runs the skill-backed subset (runs/rbi only) since that's where the
forecasting scorecard shows any signal.

Caveat: the archive lacks game_pk, so we can't enforce different-game legs —
same-day top plays can share a game (positively correlated), which INFLATES
round-robin variance and, for −EV legs, the loss rate. Read accordingly.

Run:  python -m scripts.round_robin_backtest
"""
from __future__ import annotations
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import value
from src.bet_tracker import roi_ci

ARCHIVE = ROOT / "data" / "odds" / "replay_archive.csv"


def _parlay_profit(legs: list[dict]) -> float | None:
    """Per-$1 profit for one parlay. None if ungradable."""
    outs = [l["outcome"] for l in legs]
    if any(o not in ("W", "L", "P") for o in outs):
        return None
    if "L" in outs:
        return -1.0
    win_legs = [l for l in legs if l["outcome"] == "W"]
    if not win_legs:
        return 0.0  # all push -> stake back
    dec = 1.0
    for l in win_legs:
        dec *= value.american_to_decimal(int(l["odds"]))
    return dec - 1.0


def _eval(day_groups: dict[str, list[dict]], top_n: int, by_k: int) -> list[float]:
    """Return the per-bet profit vector for 'top-N by score, round robin by k'
    across all days."""
    profits: list[float] = []
    for day, bets in day_groups.items():
        sel = sorted(bets, key=lambda b: -b["score"])[:top_n]
        if len(sel) < by_k:
            continue
        for combo in combinations(sel, by_k):
            p = _parlay_profit(list(combo))
            if p is not None:
                profits.append(p)
    return profits


def _line(label: str, profits: list[float]) -> str:
    if not profits:
        return f"  {label:28s} (no bets)"
    n = len(profits)
    roi = sum(profits) / n
    ci = roi_ci(profits)
    ci_s = f"  roiCI[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
    wins = sum(1 for p in profits if p > 0)
    return f"  {label:28s} n={n:5d}  hit {wins/n:4.0%}  ROI {roi:+7.1%}{ci_s}"


def _run(df: pd.DataFrame, title: str) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in df.itertuples():
        groups[r.date].append({"score": float(r.score), "odds": int(r.odds),
                               "outcome": r.outcome})
    print("\n" + "=" * 72)
    print(title + f"   ({len(groups)} days, {len(df)} graded bets)")
    print("=" * 72)
    for top_n in (3, 4, 5):
        print(f"  -- top-{top_n} plays/day " + "-" * 30)
        print(_line(f"singles (by 1)", _eval(groups, top_n, 1)))
        print(_line(f"round robin by 2", _eval(groups, top_n, 2)))
        print(_line(f"round robin by 3", _eval(groups, top_n, 3)))


def main():
    if not ARCHIVE.exists():
        print("No replay_archive.csv yet — run scripts.replay_board first.")
        return
    df = pd.read_csv(ARCHIVE)
    df = df[df["outcome"].isin(["W", "L", "P"])].copy()
    if df.empty:
        print("No graded bets in archive.")
        return
    _run(df, "ALL top plays")
    sb = df[df["market"].isin(["prop_runs", "prop_rbi"])]
    if len(sb) >= 20:
        _run(sb, "SKILL-BACKED only (runs/rbi)")

    # Per-market round robin: do the singles-vs-RR conclusions hold up the
    # same way for every prop market? Skipped when n<30 (too noisy).
    print("\n" + "=" * 72)
    print("PER-MARKET — round robin breakdown for every market with n>=30")
    print("=" * 72)
    for mkt, sub in df.groupby("market"):
        if len(sub) < 30 or sub["date"].nunique() < 3:
            continue
        _run(sub, f"  {mkt}")

    print("\n  Round robin does not create edge: if singles are −EV, every")
    print("  k-leg combination of them is more −EV (and higher variance).")
    print("  It only helps when the legs are genuinely +EV and you want to")
    print("  trade some hit-rate for a higher ceiling. Compare the ROI rows.")


if __name__ == "__main__":
    main()
