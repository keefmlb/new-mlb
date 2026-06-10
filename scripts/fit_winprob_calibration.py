"""Fit logistic recalibration for game-line win probabilities — OUT-OF-SAMPLE.

The team-runs model produces run predictions (lam_home, lam_away) which get
converted to win / cover / over probabilities via the joint-Poisson grid. A
full-season backtest (June 2026) showed these raw probabilities are
systematically over-dispersed: when the model says home win 30%, home actually
wins ~42%; when it says 60%, actual is ~54%. This overconfidence inflates the
apparent edge on exactly the bets the model is most wrong about.

Fix: fit a logistic recalibration `logit(p_cal) = a + b * logit(p_raw)` per
market on actual outcomes. b < 1 shrinks probabilities toward 0.5.

Methodology (reworked Jun 9 2026):
  - WALK-FORWARD, not in-sample. The old version scored the production model
    on its own training games; in-sample probabilities are better calibrated
    than live ones, so the fitted shrink understated the real overconfidence.
    Here each 2026 month is predicted by a model trained only on data BEFORE
    it (all of 2025 + earlier 2026 months), exactly like live use.
  - Fits THREE markets: moneyline, run_line, and total. Totals are evaluated
    at a half-point line grid (7.5 / 8.5 / 9.5) per game so the fit sees raw
    probabilities across the whole range, not just near 0.5.
  - Fold models are GLM-only (use_gbt=False) for speed/determinism — an
    approximation of the production GLM+GBT ensemble. The calibration target
    (overdispersion from the independent-Poisson assumption) is dominated by
    the probability conversion, not the regressor choice.

value.calibrate_winprob() applies the result to RAW model probabilities only
(the market blend happens after calibration, in probability space).
Re-run after each train_combined.

Run:  python -m scripts.fit_winprob_calibration
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

from src import model as mdl, value

OUT = ROOT / "data" / "models" / "winprob_calibration.json"

_EPS = 1e-6
_TOTAL_LINES = (7.5, 8.5, 9.5)   # half-point grid: no pushes, wide prob range
_MIN_TRAIN_GAMES = 300           # don't fit a fold on less than this


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _fit_logistic(raw_p: np.ndarray, outcome: np.ndarray) -> dict:
    """Fit outcome ~ sigmoid(a + b*logit(raw_p)). Returns {a, b, n, brier_raw,
    brier_cal}. Falls back to identity (a=0,b=1) if sklearn missing or degenerate."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"a": 0.0, "b": 1.0, "n": int(len(outcome)),
                "brier_raw": None, "brier_cal": None}
    X = _logit(raw_p).reshape(-1, 1)
    y = outcome.astype(int)
    if len(np.unique(y)) < 2:
        return {"a": 0.0, "b": 1.0, "n": int(len(y)),
                "brier_raw": None, "brier_cal": None}
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(X, y)
    a = float(lr.intercept_[0]); b = float(lr.coef_[0][0])
    p_cal = lr.predict_proba(X)[:, 1]
    brier_raw = float(((raw_p - y) ** 2).mean())
    brier_cal = float(((p_cal - y) ** 2).mean())
    return {"a": a, "b": b, "n": int(len(y)),
            "brier_raw": brier_raw, "brier_cal": brier_cal}


def _predict_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame | None:
    """Train a GLM-only model on train_df, return per-game (home_pred,
    away_pred, home_score, away_score) for test_df."""
    try:
        model, _ = mdl.fit(train_df, holdout_days=2, use_gbt=False)
    except Exception as exc:
        print(f"    fold fit failed: {exc}")
        return None
    long = mdl.long_form(test_df)
    long["pred"] = model.predict_runs(long)
    g = long.pivot_table(index="game_pk", columns="is_home", values="pred")
    g.columns = ["away_pred", "home_pred"]
    scores = test_df.set_index("game_pk")[["home_score", "away_score"]]
    return g.join(scores).dropna()


def main():
    print("=" * 60)
    print("Fitting game-line win-probability calibration (walk-forward)")
    print("=" * 60)

    from scripts.train_combined import load_combined_games
    df = load_combined_games()
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["_dt"])

    # Test folds = 2026 calendar months; train = all 2025 + 2026 months before.
    df26 = df[df["_dt"].dt.year == 2026]
    months = sorted(df26["_dt"].dt.to_period("M").unique())

    oos_frames: list[pd.DataFrame] = []
    for m in months:
        test = df26[df26["_dt"].dt.to_period("M") == m]
        train = df[df["_dt"] < m.to_timestamp()]
        if len(test) < 30 or len(train) < _MIN_TRAIN_GAMES:
            print(f"  {m}: skipped (train={len(train)}, test={len(test)})")
            continue
        print(f"  {m}: train={len(train)} games, test={len(test)} games")
        fold = _predict_fold(train.drop(columns=["_dt"]),
                             test.drop(columns=["_dt"]))
        if fold is not None and len(fold):
            oos_frames.append(fold)

    if not oos_frames:
        print("No out-of-sample folds could be built — nothing fitted.")
        return
    g = pd.concat(oos_frames)
    print(f"\nPooled out-of-sample games: {len(g)}")

    # Raw probabilities from the joint-Poisson grid (pre-calibration)
    g["p_home"] = [value.home_win_prob(h, a) for h, a in zip(g["home_pred"], g["away_pred"])]
    g["p_home_rl"] = [value.run_line_cover_prob(h, a, -1.5)
                      for h, a in zip(g["home_pred"], g["away_pred"])]
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)
    g["home_cover"] = ((g["home_score"] - g["away_score"]) >= 2).astype(int)

    ml_cal = _fit_logistic(g["p_home"].values, g["home_won"].values)
    rl_cal = _fit_logistic(g["p_home_rl"].values, g["home_cover"].values)

    # Totals: pool P(total > line) across a half-point line grid per game.
    # (Outcomes are correlated within a game; fine for a 2-parameter point
    # estimate, just don't read the reported n as independent samples.)
    tot_p, tot_y = [], []
    actual_total = (g["home_score"] + g["away_score"]).values
    for line in _TOTAL_LINES:
        tot_p.extend(value.total_over_prob(h, a, line)
                     for h, a in zip(g["home_pred"], g["away_pred"]))
        tot_y.extend((actual_total > line).astype(int))
    tot_cal = _fit_logistic(np.array(tot_p), np.array(tot_y))

    for name, c in [("moneyline", ml_cal), ("run_line", rl_cal), ("total", tot_cal)]:
        br = c.get("brier_raw"); bc = c.get("brier_cal")
        impr = f"{(br - bc) / br * 100:+.1f}%" if (br and bc) else "n/a"
        print(f"  {name:10s}  a={c['a']:+.4f}  b={c['b']:.4f}  n={c['n']}  "
              f"Brier {br:.4f}->{bc:.4f} ({impr})" if br else
              f"  {name:10s}  a={c['a']:+.4f}  b={c['b']:.4f}  n={c['n']}")

    out = {"moneyline": ml_cal, "run_line": rl_cal, "total": tot_cal,
           "method": "walk-forward monthly folds, GLM-only, 2025+2026 train"}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")
    print("value.calibrate_winprob() applies these to RAW ML / run-line / total probs.")


if __name__ == "__main__":
    main()
