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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.forecast_score import _collect

_EPS = 1e-6


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


def main():
    df = _collect(nearest_only=False)
    if df.empty:
        print("No priceable offers collected.")
        return
    # one-sided novig is a juice-strip estimate; keep both but tag.
    print(f"Offers: {len(df)}  games: {df['game_pk'].nunique()}")
    print("\nForecast-encompassing  P(over) ~ logit(market) + logit(model)")
    print("  b_mkt~1, b_mdl~0 => market encompasses model (model adds nothing)")
    print("=" * 86)
    print(f"  {'market':14s} {'n':>5s} {'games':>5s} {'b_mkt':>7s} {'b_mdl':>7s} "
          f"{'b_mdl 95% CI':>20s} {'wt_mdl':>7s}  verdict")
    rows = [(m, s) for m, s in df.groupby("market")]
    rows.append(("ALL", df))
    # focus markets first
    focus = ("runs", "rbi")
    rows.sort(key=lambda it: (0 if it[0] in focus else 1, -len(it[1])))
    for mkt, sub in rows:
        if len(sub) < 40 or sub["game_pk"].nunique() < 8:
            continue
        res = _cluster_boot_betas(sub)
        if res is None:
            continue
        (b_mkt, b_mdl), ci = res
        lo, hi = ci[0, 1], ci[1, 1]   # b_mdl CI
        # implied model weight in the log-odds combination (clamped display)
        denom = abs(b_mkt) + abs(b_mdl)
        wt = b_mdl / denom if denom > 1e-9 else 0.0
        if lo > 0:
            verdict = "* model ADDS info"
        elif hi < 0:
            verdict = "x model misleads"
        else:
            verdict = "market encompasses"
        mark = "»" if mkt in focus else " "
        print(f"{mark} {mkt:14s} {len(sub):5d} {sub['game_pk'].nunique():5d} "
              f"{b_mkt:7.3f} {b_mdl:7.3f} [{lo:+.3f},{hi:+.3f}]  {wt:6.0%}  {verdict}")
    print("=" * 86)
    print("  wt_mdl = b_mdl/(|b_mkt|+|b_mdl|): share of the log-odds combo the model earns.")
    print("  * = b_mdl CI excludes 0 (incremental edge).")
    print("  READ WITH TWO CAVEATS:")
    print("   1. In-sample: p_model uses calibration fitted on overlapping weeks,")
    print("      so b_mdl is biased UP. A null result is therefore strong; a")
    print("      positive one is an upper bound until refit out-of-sample.")
    print("   2. One-sided benchmark: where b_mkt is small (rbi 0.04, runs 0.43"
          " vs two-sided pitcher_k 1.24),")
    print("      the market prob is a juice-strip ESTIMATE, a degraded forecast —")
    print("      the model 'adding info' there partly just beats a noisy benchmark.")
    print("      The clean read is the TWO-SIDED market (pitcher_k): b_mdl ~ 0, no edge.")


if __name__ == "__main__":
    main()
