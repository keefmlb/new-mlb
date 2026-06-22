"""Search for a better ranking formula than the current Score.

Background: the current Score formula is
    score = shrunk_edge × reliability / sqrt(p × (1 − p))
and on the replay archive its daily top-10 subset has underperformed the
full board on a growing sample:
    -6.3% (n=40)  →  -15.5% (n=57)  →  -19.2% (n=67)
across the last three retrains. That's monotonic decline on more data — not
variance. The `1 / sqrt(p·(1−p))` denominator rewards extreme model
probabilities, but the calibration tests show the model is overconfident at
extremes, so this term amplifies exactly the bets the model is most likely
to be wrong about.

This script tests candidate ranking formulas on `replay_archive.csv` (every
floor-clearing bet across snapshot days, graded). For each candidate it
ranks the day's bets by that formula, takes the daily top-N, and computes
the realized ROI with a bootstrap CI. The full-board ROI is the "do
nothing" baseline — a candidate needs to beat both the current Score AND
the full board to be worth deploying.

Honest caveats:
  - Sample is still modest (≈300 bets); we report bootstrap CIs and
    daily-top-N stability across multiple Ns so we don't overfit to one N.
  - Some archive rows predate the extended-fields capture and have no
    model_prob / ev_per_dollar etc.; those rows are skipped.
  - The current Score is graded on the SAME data it ranked live, so on
    days the live system picked those bets the comparison is honest;
    rows from days with no live exposure are pure replay.

Run:  python -m scripts.score_formula_search
"""
from __future__ import annotations
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import value
from src.bet_tracker import roi_ci

ARCHIVE = ROOT / "data" / "odds" / "replay_archive.csv"

# Per-market reliability snapshot (read at script-load so the search reflects
# what the live ranker would see). Falls back to 0.40 default.
_REL = value._get_stat_reliability()
SKILL_BACKED = ("prop_runs", "prop_rbi")


def _profit(row: pd.Series) -> float | None:
    o = row["outcome"]
    if o not in ("W", "L"):
        return None
    return float(row["decimal_odds"] - 1.0) if o == "W" else -1.0


def _shrunk(edge_pct: float) -> float:
    if edge_pct <= 5.0:
        return float(edge_pct)
    return 5.0 + math.sqrt(max(0.0, edge_pct - 5.0) * 5.0)


def _rel(market: str) -> float:
    return _REL.get(market, 0.40)


def _safe(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, float(p)))


# ---------- Candidate ranking formulas ----------
# Each takes a row dict and returns a scalar (higher = bet first). NaN rows
# are filtered upstream.

def _current_score(r):
    p = _safe(r["model_prob"])
    return _shrunk(r["edge_pct"]) * _rel(r["market"]) / math.sqrt(p * (1 - p))


def _raw_edge(r):
    return float(r["edge_pct"])


def _ev_per_dollar(r):
    return float(r["ev_per_dollar"])


def _kelly(r):
    return float(r["kelly"])


def _edge_x_rel(r):
    return float(r["edge_pct"]) * _rel(r["market"])


def _shrunk_edge_x_rel(r):
    return _shrunk(r["edge_pct"]) * _rel(r["market"])


def _conviction_x_edge(r):
    return float(r["model_prob"]) * float(r["edge_pct"])


def _raw_conviction_x_edge(r):
    raw = r.get("model_prob_raw")
    if raw is None or pd.isna(raw):
        raw = r["model_prob"]
    return float(raw) * float(r["edge_pct"])


def _anti_sharpe(r):
    # Opposite of the current Sharpe-like term: PENALIZE extreme p (where
    # the model is overconfident) instead of rewarding it. 4·p·(1−p) ∈ [0,1].
    p = _safe(r["model_prob"])
    return _shrunk(r["edge_pct"]) * _rel(r["market"]) * 4.0 * p * (1 - p)


def _ev_x_rel(r):
    return float(r["ev_per_dollar"]) * _rel(r["market"])


def _kelly_x_rel(r):
    return float(r["kelly"]) * _rel(r["market"])


def _edge_juice_adj(r):
    # Per-unit-risk edge: divide by (decimal_odds − 1) so high payouts don't
    # crowd in.
    d = float(r["decimal_odds"])
    return float(r["edge_pct"]) / max(0.01, d - 1.0)


def _edge_1sided_disc(r):
    """edge_pct, but one-sided 'Yes' props are halved — their edge rests on
    the unreliable 8% juice-strip estimate and floods the live leaderboard."""
    e = float(r["edge_pct"])
    if "1-sided" in str(r.get("description", "")):
        e *= 0.5
    return e


def _skill_backed_only(r):
    """Edge × reliability, but ONLY for skill-backed markets — everything
    else is sent to the back of the pack."""
    if r["market"] not in SKILL_BACKED:
        return -1e6
    return float(r["edge_pct"]) * _rel(r["market"])


CANDIDATES = {
    "current_score (live)":      _current_score,
    "raw edge_pct":              _raw_edge,
    "edge_pct 1-sided×0.5":      _edge_1sided_disc,
    "EV/$":                      _ev_per_dollar,
    "Kelly":                     _kelly,
    "edge × rel":                _edge_x_rel,
    "shrunk_edge × rel":         _shrunk_edge_x_rel,
    "conviction × edge":         _conviction_x_edge,
    "raw_conviction × edge":     _raw_conviction_x_edge,
    "anti-Sharpe (4p(1-p))":     _anti_sharpe,
    "EV × rel":                  _ev_x_rel,
    "Kelly × rel":               _kelly_x_rel,
    "edge / (decimal-1)":        _edge_juice_adj,
    "skill-backed edge×rel":     _skill_backed_only,
}


def _evaluate(df: pd.DataFrame, ranker, top_n: int) -> list[float]:
    """Return per-bet profit for daily top-N ranked by `ranker`."""
    profits: list[float] = []
    for day, sub in df.groupby("date"):
        sub = sub.copy()
        sub["_rank"] = sub.apply(ranker, axis=1)
        sel = sub.sort_values("_rank", ascending=False).head(top_n)
        for _, r in sel.iterrows():
            p = _profit(r)
            if p is not None:
                profits.append(p)
    return profits


def _fmt(label: str, profits: list[float], baseline: float | None = None) -> str:
    n = len(profits)
    if n == 0:
        return f"  {label:30s} (no bets)"
    roi = sum(profits) / n
    ci = roi_ci(profits)
    wins = sum(1 for p in profits if p > 0)
    delta = ""
    if baseline is not None:
        delta = f"  Δ {(roi - baseline) * 100:+5.1f}pp"
    ci_s = f"  roiCI[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
    return f"  {label:30s} n={n:4d}  hit {wins/n:4.0%}  ROI {roi:+6.1%}{ci_s}{delta}"


def main():
    if not ARCHIVE.exists():
        print("No replay_archive.csv yet.")
        return
    df = pd.read_csv(ARCHIVE)
    needed = ["model_prob", "novig_prob", "ev_per_dollar",
              "kelly", "decimal_odds", "edge_pct", "odds", "market"]
    pre = len(df)
    for col in needed:
        df = df[df[col].notna()]
    df = df[df["outcome"].isin(["W", "L", "P"])].copy()
    print(f"Archive rows with full ranking fields + graded outcome: "
          f"{len(df)} / {pre}")
    if df.empty:
        print("No rows with the extended fields yet — re-run replay_board to "
              "populate them.")
        return

    # Full board ROI = the "do nothing, bet every floor-clearer" baseline.
    full = []
    for _, r in df.iterrows():
        p = _profit(r)
        if p is not None:
            full.append(p)
    full_roi = sum(full) / len(full) if full else 0.0
    print(_fmt("FULL BOARD baseline", full, baseline=full_roi))

    for top_n in (5, 10, 20):
        print("\n" + "=" * 80)
        print(f"DAILY TOP-{top_n} BY EACH FORMULA   (Δ vs full-board baseline above)")
        print("=" * 80)
        results = []
        for name, fn in CANDIDATES.items():
            profs = _evaluate(df, fn, top_n)
            roi = sum(profs) / len(profs) if profs else float("nan")
            results.append((name, profs, roi))
        # rank by ROI desc
        results.sort(key=lambda x: -x[2] if not math.isnan(x[2]) else 1)
        for name, profs, _ in results:
            print(_fmt(name, profs, baseline=full_roi))

    print("\n" + "=" * 80)
    print("BY MARKET — how does the WINNER's top-10 break down by market?")
    print("=" * 80)
    best_name, best_fn, _ = max(
        ((n, f, sum(_evaluate(df, f, 10)) / max(1, len(_evaluate(df, f, 10))))
         for n, f in CANDIDATES.items()),
        key=lambda x: x[2])
    df = df.copy()
    df["_rank"] = df.apply(best_fn, axis=1)
    top = (df.sort_values(["date", "_rank"], ascending=[True, False])
              .groupby("date").head(10))
    for mkt, sub in top.groupby("market"):
        profs = [_profit(r) for _, r in sub.iterrows()]
        profs = [p for p in profs if p is not None]
        if not profs:
            continue
        print(_fmt(f"[{best_name}] {mkt}", profs))


if __name__ == "__main__":
    main()
