"""Fit per-market logistic recalibration for player-prop probabilities.

The prop pipeline converts a projected mean to P(over) via a Negative
Binomial tail (value.prob_over_count). Until now those raw probabilities got
a single hand-set logit shrink (CALIBRATION_LOGIT_B = 0.70) for every market
— a guess. The game-line markets got a FITTED shrink in June 2026 and the
fitted slopes (b≈0.46-0.69) came back far more aggressive than 0.70, which
strongly suggests the prop markets are still over-confident too.

This script fits `logit(p_cal) = a + b * logit(p_raw)` per prop market the
same way fit_winprob_calibration does for game lines, using the analytical
projection-vs-actual datasets (props_{bat,pit}_{2025,2026}.csv built by
build_props_2025.py / train_props.py). Those projections are leak-free by
construction — they're built from prior-Monday stat snapshots — so the rows
are honest out-of-sample samples without needing fold refits.

Methodology notes:
  - Each row is evaluated at a half-point line grid around the projection
    (nearest half-line and ±1, floored at 0.5) so the fit sees raw
    probabilities across the whole range, not just near 0.5. Lines within a
    row are correlated; fine for a 2-parameter point estimate, but don't
    read n as independent samples.
  - Honest evaluation: (a, b) are fit on the FIRST 80% of dates and the
    Brier improvement is reported on the LAST 20% (out-of-sample). The
    saved parameters are then refit on ALL rows.
  - Limitation: these are ANALYTICAL projections; the live pipeline blends
    analytical + ML per StatModel.blend_weight. The calibration target
    (NegBin tail overconfidence) is dominated by the distributional
    assumption, not the point estimate, so the fit transfers.

value.evaluate_prop applies the result to RAW prop probabilities (before
the market blend), falling back to the 0.70 logit shrink for markets
without a fitted entry. Re-run after build_props_2025 / train_props.

Run:  python -m scripts.fit_prop_calibration
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import value

OUT = ROOT / "data" / "models" / "prop_calibration.json"

BAT_CSVS = [ROOT / "data" / "games" / f"props_bat_{y}.csv" for y in (2025, 2026)]
PIT_CSVS = [ROOT / "data" / "games" / f"props_pit_{y}.csv" for y in (2025, 2026)]

# (projection column, actual column, market key as used by evaluate_prop)
BAT_MARKETS = [
    ("proj_h",    "h",      "hits"),
    ("proj_hr",   "hr",     "hr"),
    ("proj_tb",   "tb",     "tb"),
    ("proj_rbi",  "rbi",    "rbi"),
    ("proj_runs", "runs_b", "runs"),
    ("proj_k",    "k_b",    "k"),
    ("proj_bb",   "bb_b",   "bb"),
    ("proj_hrr",  "hrr",    "hrr"),
]
PIT_MARKETS = [
    ("proj_k",        "k_p",  "pitcher_k"),
    ("proj_bb",       "bb_p", "pitcher_bb"),
    ("proj_h",        "h_p",  "pitcher_h"),
    ("proj_er",       "er",   "pitcher_er"),
    ("proj_hr",       "hr_p", "pitcher_hr"),
    ("expected_outs", "outs", "pitcher_outs"),
]

_EPS = 1e-6
_LINE_OFFSETS = (-1.0, 0.0, 1.0)
_HOLDOUT_FRAC = 0.20
_MIN_ROWS = 500


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _fit_ab(raw_p: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(_logit(raw_p).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def _apply_ab(raw_p: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a + b * _logit(raw_p)
    return 1.0 / (1.0 + np.exp(-z))


def _load(csvs: list[Path]) -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in csvs if p.exists()]
    for p in csvs:
        if not p.exists():
            print(f"  (no {p.name} — skipping)")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    # Derive Hits+Runs+RBIs (a real book market) so it gets its own fitted
    # calibration instead of falling back to the blanket 0.70 shrink.
    _a = [c for c in ("h", "runs_b", "rbi") if c in df.columns]
    _p = [c for c in ("proj_h", "proj_runs", "proj_rbi") if c in df.columns]
    if len(_a) == 3 and len(_p) == 3:
        df["hrr"] = df[_a].sum(axis=1)
        df["proj_hrr"] = df[_p].sum(axis=1)
    return df.sort_values("date").reset_index(drop=True)


def _build_samples(df: pd.DataFrame, proj_col: str, act_col: str,
                   market: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_p, outcome, date_ordinal) across the half-point line grid."""
    sub = df[[proj_col, act_col, "date"]].dropna()
    sub = sub[sub[proj_col] > 0]
    means = sub[proj_col].to_numpy(float)
    actual = sub[act_col].to_numpy(float)
    date_rank = pd.factorize(sub["date"], sort=True)[0]

    ps, ys, ds = [], [], []
    for mean, act, dr in zip(means, actual, date_rank):
        base = np.floor(mean) + 0.5     # nearest half-line at/under the mean
        disp = value.get_dispersion(market, mean)
        for off in _LINE_OFFSETS:
            line = base + off
            if line < 0.5:
                continue
            ps.append(value.prob_over_count(mean, line, disp))
            ys.append(1 if act > line else 0)
            ds.append(dr)
    return np.asarray(ps), np.asarray(ys, dtype=int), np.asarray(ds)


def _fit_market(df: pd.DataFrame, proj_col: str, act_col: str,
                market: str) -> dict | None:
    raw_p, y, date_rank = _build_samples(df, proj_col, act_col, market)
    if len(y) < _MIN_ROWS:
        print(f"  {market:14s} skipped (n={len(y)} < {_MIN_ROWS})")
        return None

    # Temporal 80/20 split for the honest out-of-sample Brier numbers.
    cut = np.quantile(date_rank, 1.0 - _HOLDOUT_FRAC)
    tr, te = date_rank <= cut, date_rank > cut
    oos = {}
    if te.sum() >= 100 and tr.sum() >= _MIN_ROWS:
        ab = _fit_ab(raw_p[tr], y[tr])
        if ab:
            p_cal = _apply_ab(raw_p[te], *ab)
            # The hand-set 0.70 shrink is the incumbent — beat IT, not raw.
            p_incumbent = _apply_ab(raw_p[te], 0.0, value.CALIBRATION_LOGIT_B)
            oos = {
                "brier_raw_oos":       float(((raw_p[te] - y[te]) ** 2).mean()),
                "brier_incumbent_oos": float(((p_incumbent - y[te]) ** 2).mean()),
                "brier_cal_oos":       float(((p_cal - y[te]) ** 2).mean()),
                "n_oos": int(te.sum()),
            }

    # Gate: a fitted calibration must BEAT the incumbent 0.70 shrink on the
    # out-of-sample slice, or we don't ship it (the market falls back to the
    # incumbent). Catches nonstationary markets — e.g. pitcher_outs, whose
    # 2025 projection distribution differs enough from 2026 that the pooled
    # fit was 14% WORSE on the 2026 holdout.
    inc = oos.get("brier_incumbent_oos")
    cal = oos.get("brier_cal_oos")
    if inc is not None and cal is not None and cal >= inc:
        print(f"  {market:14s} NOT SAVED — fit loses to 0.70-shrink OOS "
              f"({inc:.4f} vs {cal:.4f}); falling back to default shrink")
        return None

    # Final parameters: fit on everything.
    ab = _fit_ab(raw_p, y)
    if not ab:
        print(f"  {market:14s} fit failed")
        return None
    a, b = ab
    p_cal = _apply_ab(raw_p, a, b)
    res = {"a": a, "b": b, "n": int(len(y)),
           "brier_raw": float(((raw_p - y) ** 2).mean()),
           "brier_cal": float(((p_cal - y) ** 2).mean()), **oos}
    vs = (f"  OOS vs 0.70-shrink: {inc:.4f}->{cal:.4f}"
          f" ({(inc - cal) / inc * 100:+.1f}%)" if inc and cal else "")
    print(f"  {market:14s} a={a:+.4f}  b={b:.4f}  n={res['n']}{vs}")
    return res


def main():
    print("=" * 64)
    print("Fitting per-market prop probability calibration")
    print("=" * 64)
    out: dict = {}

    for label, csvs, markets in [("batter", BAT_CSVS, BAT_MARKETS),
                                 ("pitcher", PIT_CSVS, PIT_MARKETS)]:
        print(f"\n{label} props:")
        df = _load(csvs)
        if df is None:
            continue
        for proj_col, act_col, market in markets:
            if proj_col not in df.columns or act_col not in df.columns:
                print(f"  {market:14s} skipped (missing columns)")
                continue
            res = _fit_market(df, proj_col, act_col, market)
            if res:
                out[market] = res

    if not out:
        print("\nNothing fitted — are the props_*_{2025,2026}.csv files present?")
        return
    out["method"] = ("logit recalibration per prop market, analytical "
                     "projections vs actuals, 2025+2026, half-point line grid "
                     "(nearest ±1), 80/20 temporal split for OOS numbers")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")
    print("value.evaluate_prop applies these to RAW prop probs (before the "
          "market blend); markets without an entry fall back to the 0.70 shrink.")


if __name__ == "__main__":
    main()
