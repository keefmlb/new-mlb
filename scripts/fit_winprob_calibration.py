"""Fit logistic recalibration for game-line win probabilities.

The team-runs model produces run predictions (lam_home, lam_away) which get
converted to win / cover probabilities via the joint-Poisson grid. A
full-season backtest (June 2026) showed these probabilities are systematically
OVER-DISPERSED: when the model says home win 30%, home actually wins ~42%;
when it says 60%, actual is ~54%. This overconfidence inflates the apparent
edge on exactly the bets the model is most wrong about — and those bets lose
(high-score/high-edge buckets ran -34% ROI in the bet log).

Fix: fit a logistic recalibration `logit(p_cal) = a + b * logit(p_raw)` per
market on actual outcomes. b < 1 shrinks probabilities toward 0.5, killing the
fake edges. We fit two:
  - moneyline:  outcome = home won
  - run_line:   outcome = home covered -1.5 (won by 2+), using raw p(home -1.5)

Saved to data/models/winprob_calibration.json. value.calibrate_winprob()
loads and applies it. Re-run after each train_combined.

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

GAMES = ROOT / "data" / "games" / "games_2026.csv"
OUT   = ROOT / "data" / "models" / "winprob_calibration.json"

_EPS = 1e-6


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


def main():
    print("=" * 60)
    print("Fitting game-line win-probability calibration")
    print("=" * 60)

    games = pd.read_csv(GAMES)
    games = games[games["is_final"] == True].copy()
    games = games.dropna(subset=["home_score", "away_score"])
    print(f"Games: {len(games)}")

    model = mdl.TeamScoreModel.load(ROOT / "data" / "models" / "team_runs.joblib")
    boot = mdl.load_bootstrap_ensemble(ROOT / "data" / "models")
    allm = [model] + boot

    long = mdl.long_form(games)
    long["pred"] = mdl.predict_ensemble(allm, long)
    g = long.pivot_table(index="game_pk", columns="is_home", values="pred")
    g.columns = ["away_pred", "home_pred"]
    g = g.join(games.set_index("game_pk")[["home_score", "away_score"]]).dropna()

    # Raw probabilities from the joint-Poisson grid (pre-calibration)
    g["p_home"] = [value.home_win_prob(h, a) for h, a in zip(g["home_pred"], g["away_pred"])]
    g["p_home_rl"] = [value.run_line_cover_prob(h, a, -1.5)
                      for h, a in zip(g["home_pred"], g["away_pred"])]
    g["home_won"] = (g["home_score"] > g["away_score"]).astype(int)
    g["home_cover"] = ((g["home_score"] - g["away_score"]) >= 2).astype(int)

    ml_cal = _fit_logistic(g["p_home"].values, g["home_won"].values)
    rl_cal = _fit_logistic(g["p_home_rl"].values, g["home_cover"].values)

    for name, c in [("moneyline", ml_cal), ("run_line", rl_cal)]:
        br = c.get("brier_raw"); bc = c.get("brier_cal")
        impr = f"{(br - bc) / br * 100:+.1f}%" if (br and bc) else "n/a"
        print(f"  {name:10s}  a={c['a']:+.4f}  b={c['b']:.4f}  n={c['n']}  "
              f"Brier {br:.4f}->{bc:.4f} ({impr})" if br else
              f"  {name:10s}  a={c['a']:+.4f}  b={c['b']:.4f}  n={c['n']}")

    out = {"moneyline": ml_cal, "run_line": rl_cal}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")
    print("value.calibrate_winprob() will apply these to ML / run-line pricing.")


if __name__ == "__main__":
    main()
