"""Fit a calibration map for the simulator's hit probabilities.

The sim's `sim_hit` is systematically overconfident: across 3,349 graded picks
it claimed 59.6% and delivered 50.8% (-8.9pp), and the gap holds in every
market. For singles that mostly distorts the display; for parlays it compounds
multiplicatively (a 5-leg ticket at a displayed 85%/leg is really ~23%, not
44%), which is exactly how the tickets are being built.

This fits p_cal = sigmoid(a + b * logit(p_raw)) per market on the graded
record, validated on a TEMPORAL holdout. A market's fit is only saved if it
beats the raw probability out-of-sample on Brier score; otherwise that market
falls back to the pooled fit, and failing that to identity.

DOUBLE-SHRINK GUARD: fitting must always use the RAW simulator output. The
leaderboard stores `sim_hit_raw` alongside the calibrated `sim_hit`, and this
script reads `sim_hit_raw` when present. Never fit on calibrated values — this
project has shipped that bug before (see the game-line double-shrink note in
CLAUDE.md).

Run:  python -m scripts.fit_sim_calibration
Out:  data/models/sim_calibration.json
"""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PICKS = ROOT / "data" / "bets" / "sim_picks.json"
OUT = ROOT / "data" / "models" / "sim_calibration.json"

_EPS = 1e-6
_MIN_ROWS = 120          # per-market minimum to attempt a fit
_HOLDOUT_FRAC = 0.30


def _logit(p):
    p = min(max(float(p), _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _rows():
    data = json.loads(PICKS.read_text(encoding="utf-8"))
    out = []
    for r in data:
        if r.get("outcome") not in ("W", "L"):
            continue
        # ALWAYS prefer the raw value; calibrated rows must never be refit.
        p = r.get("sim_hit_raw")
        if p is None:
            p = r.get("sim_hit")
        if p is None:
            continue
        p = float(p)
        if not (0.0 < p < 1.0):
            continue
        out.append({"date": r.get("date", ""), "market": r.get("market", "?"),
                    "p": p, "rank": r.get("rank") or 10,
                    "line": r.get("line") if r.get("line") is not None else 0.5,
                    "y": 1 if r["outcome"] == "W" else 0})
    out.sort(key=lambda r: r["date"])
    return out


def _rank_feat(rank) -> float:
    """log(rank) — RANK CARRIES SIGNAL BEYOND sim_hit. Measured: a TB pick at
    raw 85% hits 77% when it's rank<=3 but a rank 6-20 pick at raw 80% hits
    only 63%. A calibration fit on probability alone is dominated by the many
    mid-rank rows and therefore UNDER-states the few top ones (-6.8pp on TB
    top-3), which is exactly the population being bet."""
    try:
        r = float(rank)
    except (TypeError, ValueError):
        r = 10.0
    return math.log(max(1.0, min(r, 25.0)))


def _line_feat(line) -> float:
    """The LINE the pick sits on. Ranking by probability used to fill every
    batter board with 'over 0.5' only; once line diversification opened up the
    1.5+ lines they exposed a distribution-shape error the 0.5-only fit could
    never see — on TB 1.5+ the sim claims ~50% and delivers 27.5%. A single
    probability curve cannot correct that, because the bias depends on WHERE in
    the tail the line sits, so the line enters as its own term."""
    try:
        return float(line)
    except (TypeError, ValueError):
        return 0.5


def _design(rows, use_rank: bool, use_line: bool):
    cols = []
    for r in rows:
        row = [_logit(r["p"])]
        if use_rank:
            row.append(_rank_feat(r["rank"]))
        if use_line:
            row.append(_line_feat(r["line"]))
        cols.append(row)
    return np.array(cols, dtype=float)


def _fit(rows, use_rank: bool, use_line: bool = False):
    """Logistic fit of outcome on logit(p) [+ log(rank)] [+ line]."""
    if len(rows) < 30:
        return None
    X = _design(rows, use_rank, use_line)
    y = np.array([r["y"] for r in rows], dtype=int)
    if y.min() == y.max():
        return None
    # A line term is meaningless if every row sits on the same line.
    if use_line and len({round(_line_feat(r["line"]), 2) for r in rows}) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        m.fit(X, y)
        c = [float(v) for v in m.coef_[0]]
        out = {"a": float(m.intercept_[0]), "b": c[0]}
        i = 1
        if use_rank:
            out["c_rank"] = c[i]; i += 1
        if use_line:
            out["c_line"] = c[i]
        return out
    except Exception:
        return None


def _apply(fit, p, rank, line=0.5):
    z = fit["a"] + fit["b"] * _logit(p)
    if "c_rank" in fit:
        z += fit["c_rank"] * _rank_feat(rank)
    if "c_line" in fit:
        z += fit["c_line"] * _line_feat(line)
    return _sigmoid(z)


def _brier2(rows, fn):
    """Brier over rows; `fn` takes the whole row (needs p AND rank)."""
    return sum((fn(r) - r["y"]) ** 2 for r in rows) / len(rows)


def main():
    rows = _rows()
    if not rows:
        print("No graded sim picks with probabilities yet.")
        return
    print(f"Graded sim picks: {len(rows)}  "
          f"({rows[0]['date']} .. {rows[-1]['date']})")

    def _eval(sub, label):
        """Fit on the temporal train split, score on the holdout. Tries the
        rank-aware form and the probability-only form; keeps whichever wins
        OUT OF SAMPLE, and only if it beats the raw probability."""
        cut = int(len(sub) * (1 - _HOLDOUT_FRAC))
        tr, te = sub[:cut], sub[cut:]
        if len(tr) < 30 or len(te) < 20:
            return None
        raw = _brier2(te, lambda r: r["p"])
        best, best_b, best_lbl = None, raw, "raw"
        for use_rank, use_line, lbl in (
                (True, True, "prob+rank+line"), (False, True, "prob+line"),
                (True, False, "prob+rank"), (False, False, "prob")):
            fit = _fit(tr, use_rank, use_line)
            if fit is None:
                continue
            b = _brier2(te, lambda r, f=fit: _apply(f, r["p"], r["rank"],
                                                    r["line"]))
            if b < best_b:
                best, best_b, best_lbl = fit, b, lbl
        mean_p = sum(r["p"] for r in sub) / len(sub)
        mean_y = sum(r["y"] for r in sub) / len(sub)
        gain = (raw - best_b) / raw * 100 if raw else 0.0
        tag = (f"kept {best_lbl}" if best else "REJECTED (no OOS gain)")
        rk = "".join(f" {k}={best[k]:+.3f}" for k in ("c_rank","c_line") if best and k in best)
        print(f"  {label:22s} n={len(sub):5d}  claims {mean_p:5.1%} -> actual "
              f"{mean_y:5.1%}  OOS Brier {raw:.4f}->{best_b:.4f} "
              f"({gain:+.1f}%)  {tag}{rk}")
        if not best:
            return None
        # Record which lines the fit actually SAW in training. A calibration
        # learned only on 0.5-line picks says nothing about a 2.5-line pick,
        # and applying it there silently produces a confident wrong number —
        # which is how TB 1.5+ came to display ~38% while delivering 27.5%.
        # game_sim uses this to flag out-of-range picks as uncalibrated.
        best.update({"n": len(sub), "brier_raw": raw, "brier_cal": best_b,
                     "form": best_lbl,
                     "train_lines": sorted({round(_line_feat(r["line"]), 1)
                                            for r in tr})})
        return best

    print("\nPOOLED")
    pooled = _eval(rows, "ALL MARKETS")

    print("\nPER MARKET")
    by = defaultdict(list)
    for r in rows:
        by[r["market"]].append(r)
    per = {}
    for mkt, sub in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(sub) < _MIN_ROWS:
            print(f"  {mkt:22s} n={len(sub):5d}  (below {_MIN_ROWS}, "
                  "will use pooled)")
            continue
        got = _eval(sub, mkt)
        if got:
            per[mkt] = got

    if not pooled and not per:
        print("\nNothing beat the raw probability out-of-sample; not writing.")
        return
    payload = {"pooled": pooled, "markets": per,
               "note": "p_cal = sigmoid(a + b*logit(p_raw) + c_rank*log(rank) + c_line*line). Fit on RAW sim "
                       "output only. Markets absent here fall back to pooled, "
                       "then identity."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")
    print(f"  per-market fits: {sorted(per)}   pooled: "
          f"{'yes' if pooled else 'no'}")


if __name__ == "__main__":
    main()
