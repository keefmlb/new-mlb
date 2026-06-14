"""Forecast-encompassing test: does the model add info the market lacks?

The scorecard compares model vs market Brier scores and differences them.
This asks the sharper question from the forecasting literature: regress the
realized outcome on BOTH forecasts jointly —

    P(over) = sigmoid( b0 + b_mkt·logit(p_market) + b_mdl·logit(p_model) )

and test whether b_mdl is distinguishable from 0. Conditioning on the market
forecast removes all the variance the market already explains, so the test
isolates the model's INCREMENTAL information — a more powerful (tighter) test
than differencing two noisy Brier scores. Bonus: the fitted (b_mkt, b_mdl)
are the optimal log-odds combination weights, i.e. the principled blend, and
their CIs say whether the model weight is real.

Reads:
  b_mdl ≈ 0, CI spans 0  -> model adds nothing beyond the market (market
                            "encompasses" the model). Bet the market.
  b_mdl > 0, CI excludes 0 -> model carries incremental signal; lean on it
                            by the implied weight b_mdl/(b_mkt+b_mdl).
  b_mdl < 0, CI excludes 0 -> model is actively misleading vs the market.

Inference is a GAME-CLUSTERED bootstrap: each resample draws whole games and
refits the logistic, so within-game correlation doesn't fake-narrow the CI.
Uses the full implied-distribution (every informative alt line), like the
multi-line scorecard.

Caveat: p_model uses the prop calibration, fitted on a pool overlapping these
weeks, so the model is mildly favored in-sample. A b_mdl that is still ~0 is
therefore strong evidence of no incremental edge.

Run:  python -m scripts.forecast_encompassing
"""
from __future__ import annotations
import sys
from pathlib import Path

import math

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_score import _collect, _select_snapshots
from src import value

_EPS = 1e-6


def _fit_oos_calibration(cutoff: str) -> dict:
    """Per-market logit (a,b) fit on prop rows STRICTLY BEFORE `cutoff`.

    Same recipe as fit_prop_calibration but date-restricted, so the
    calibration carries no information from the test window — the model's
    p_model becomes a genuine out-of-sample forecast."""
    from scripts.fit_prop_calibration import (
        _load, _build_samples, _fit_ab,
        BAT_CSVS, PIT_CSVS, BAT_MARKETS, PIT_MARKETS)
    cal: dict = {}
    for csvs, markets in [(BAT_CSVS, BAT_MARKETS), (PIT_CSVS, PIT_MARKETS)]:
        df = _load(csvs)
        if df is None:
            continue
        df = df[df["date"].astype(str) < cutoff]
        if df.empty:
            continue
        for proj_col, act_col, market in markets:
            if proj_col not in df.columns or act_col not in df.columns:
                continue
            raw_p, y, _ = _build_samples(df, proj_col, act_col, market)
            if len(y) < 500:
                continue
            ab = _fit_ab(raw_p, y)
            if ab:
                cal[market] = ab
    return cal


def _make_oos_cal_fn(cal: dict):
    """(raw_tail_prob, market) -> calibrated P(over) using the OOS (a,b),
    falling back to the default logit shrink when a market wasn't fit."""
    def fn(raw_p, market):
        ab = cal.get(market)
        if not ab:
            return value.calibrate_prob(raw_p)
        a, b = ab
        p = min(max(float(raw_p), 1e-6), 1 - 1e-6)
        z = a + b * math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-z))
    return fn


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Return [b_mkt, b_mdl] from logistic(over ~ logit_mkt + logit_mdl)."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)
    return lr.coef_[0].copy()


def _cluster_boot_betas(sub, n_boot: int = 1500, seed: int = 0):
    """Game-clustered bootstrap of [b_mkt, b_mdl]. Returns the point fit and
    the (lo, hi) 95% CI for each coefficient, or None."""
    lm = _logit(sub["p_market"].to_numpy())
    ld = _logit(sub["p_model"].to_numpy())
    y = sub["over"].to_numpy().astype(int)
    X = np.column_stack([lm, ld])
    point = _fit(X, y)
    if point is None:
        return None
    games = sub["game_pk"].to_numpy()
    by_game: dict = {}
    for i, g in enumerate(games):
        by_game.setdefault(g, []).append(i)
    keys = list(by_game.keys())
    idx_by_game = [np.array(by_game[k]) for k in keys]
    rng = np.random.default_rng(seed)
    betas = []
    G = len(keys)
    for _ in range(n_boot):
        pick = rng.integers(0, G, size=G)
        rows = np.concatenate([idx_by_game[p] for p in pick])
        b = _fit(X[rows], y[rows])
        if b is not None:
            betas.append(b)
    if len(betas) < 100:
        return None
    betas = np.array(betas)
    ci = np.quantile(betas, [0.025, 0.975], axis=0)
    return point, ci


# Currently skill-backed markets (CI excludes 0). Used only as a visual
# highlight — the report runs over every market with enough data.
_FOCUS = ("runs", "rbi")


def _report(df, title: str) -> None:
    print("\n" + "=" * 90)
    print(title + f"   (offers {len(df)}, games {df['game_pk'].nunique()})")
    print("=" * 90)
    print(f"  {'market':14s} {'n':>5s} {'games':>5s} {'b_mkt':>7s} {'b_mdl':>7s} "
          f"{'b_mdl 95% CI':>20s} {'wt_mdl':>7s}  verdict")
    rows = [(m, s) for m, s in df.groupby("market")]
    rows.append(("ALL", df))
    rows.sort(key=lambda it: (0 if it[0] in _FOCUS else 1, -len(it[1])))
    for mkt, sub in rows:
        if len(sub) < 40 or sub["game_pk"].nunique() < 8:
            continue
        res = _cluster_boot_betas(sub)
        if res is None:
            continue
        (b_mkt, b_mdl), ci = res
        lo, hi = ci[0, 1], ci[1, 1]   # b_mdl CI
        denom = abs(b_mkt) + abs(b_mdl)
        wt = b_mdl / denom if denom > 1e-9 else 0.0
        verdict = ("* model ADDS info" if lo > 0
                   else "x model misleads" if hi < 0
                   else "market encompasses")
        mark = "»" if mkt in _FOCUS else " "
        print(f"{mark} {mkt:14s} {len(sub):5d} {sub['game_pk'].nunique():5d} "
              f"{b_mkt:7.3f} {b_mdl:7.3f} [{lo:+.3f},{hi:+.3f}]  {wt:6.0%}  {verdict}")


def main():
    insample = _collect(nearest_only=False)
    if insample.empty:
        print("No priceable offers collected.")
        return

    # Out-of-sample: refit the prop calibration using ONLY data before the
    # earliest test day, then recompute p_model with it. Strips the in-sample
    # calibration advantage so any surviving b_mdl is real incremental skill.
    snaps = _select_snapshots()
    cutoff = min(snaps) if snaps else "2026-06-05"
    cal = _fit_oos_calibration(cutoff)
    oos = _collect(nearest_only=False, cal_fn=_make_oos_cal_fn(cal))

    print("Forecast-encompassing  P(over) ~ logit(market) + logit(model)")
    print("  b_mkt~1, b_mdl~0 => market encompasses model (model adds nothing)")
    _report(insample, "IN-SAMPLE (production calibration, overlaps test weeks)")
    _report(oos, f"OUT-OF-SAMPLE (calibration refit on dates < {cutoff}; "
                 f"{len(cal)} markets fit)")

    print("\n" + "=" * 90)
    print("READ (Jun 2026): OOS ≈ in-sample to the 3rd decimal. The calibration")
    print("pool (~100k+ rows, all 2025 + 2026 to date) dwarfs the ~6 test days,")
    print("so dropping them barely moves (a,b) — the in-sample caveat is RESOLVED:")
    print("p_model is effectively already out-of-sample; the b_mdl signals are not")
    print("a calibration-overfit artifact.")
    print("  REMAINING confound: one-sided novig is a flat-8% juice-strip ESTIMATE")
    print("  (b_mkt≈0.04 rbi / 0.43 runs vs 1.24 two-sided pitcher_k) — a degraded")
    print("  benchmark, so the rbi/runs positive b_mdl is largely 'beats a noisy")
    print("  market proxy', not proven edge vs a real two-sided line. Clean read")
    print("  (two-sided pitcher_k): b_mdl≈0, no edge. To resolve runs/rbi, the next")
    print("  lever is a CALIBRATED one-sided overround (replace the flat 8%).")


if __name__ == "__main__":
    main()
