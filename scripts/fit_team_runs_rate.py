"""Compute a league-level rate-factor for team-runs predictions.

The team-runs model uses static feature aggregates (season RPG, season OPS,
park, weather, opposing-starter quality). It can't see "league offense has
been cooler the last two weeks." When the league environment shifts, the
model lags — May 4-10 2026 was a cool offensive stretch where the model
over-projected total runs by ~1.0 per game.

Fix: compute `rate = mean(actual_runs) / mean(pred_runs)` over the last N
days (default 14) and persist to `data/models/team_runs_rate.json`. The
predictor multiplies its team-runs output by this factor at predict time.

Clamped to [0.85, 1.15] so a freak window can't whipsaw projections.
Re-run after `build_dataset` + `train_combined` whenever you want to
refresh the league-level correction.

Run:  python -m scripts.fit_team_runs_rate
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model as mdl

GAMES = ROOT / "data" / "games" / "games_2026.csv"
MODEL = ROOT / "data" / "models" / "team_runs.joblib"
OUT   = ROOT / "data" / "models" / "team_runs_rate.json"

WINDOW_DAYS  = 14    # how many recent days to fit on
MIN_GAMES    = 40    # below this, keep rate_factor at 1.0
CLAMP_LO     = 0.85
CLAMP_HI     = 1.15


def main():
    print("=" * 60)
    print("Fitting team-runs league rate factor")
    print("=" * 60)

    games = pd.read_csv(GAMES)
    games = games[games["is_final"] == True].copy()
    games["date"] = pd.to_datetime(games["date"])
    cutoff_dates = sorted(games["date"].unique())[-WINDOW_DAYS:]
    recent = games[games["date"].isin(cutoff_dates)]
    n = len(recent)
    print(f"Window: last {len(cutoff_dates)} dates -> {n} games")

    if n < MIN_GAMES:
        print(f"Too few games (<{MIN_GAMES}); keeping rate_factor=1.0")
        rate_factor = 1.0
    else:
        model = mdl.TeamScoreModel.load(MODEL)
        long  = mdl.long_form(recent)
        long["pred"] = model.predict_runs(long)

        # Per-game totals
        totals = long.groupby("game_pk").agg(pred=("pred", "sum"),
                                              actual=("y_runs", "sum"))
        pred_mean = float(totals["pred"].mean())
        act_mean  = float(totals["actual"].mean())
        raw_rate  = act_mean / pred_mean if pred_mean > 0 else 1.0
        rate_factor = max(CLAMP_LO, min(CLAMP_HI, raw_rate))

        bias = (totals["pred"] - totals["actual"]).mean()
        mae  = (totals["pred"] - totals["actual"]).abs().mean()
        print(f"  pred_mean    = {pred_mean:.3f}")
        print(f"  actual_mean  = {act_mean:.3f}")
        print(f"  raw rate     = {raw_rate:.4f}")
        print(f"  clamped rate = {rate_factor:.4f}  (clamp [{CLAMP_LO}, {CLAMP_HI}])")
        print(f"  MAE          = {mae:.3f}")
        print(f"  bias         = {bias:+.3f}")

    out = {
        "rate_factor": rate_factor,
        "window_days": WINDOW_DAYS,
        "n_games":     int(n),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
