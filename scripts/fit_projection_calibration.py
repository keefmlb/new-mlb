"""Fit per-stat isotonic calibrators on backtest data.

Usage:
  python -m scripts.fit_projection_calibration

The 21-day backtest showed every top-decile counting stat under-projects
and every bottom-decile over-projects. We fit a monotonic curve
`actual = f(projected)` per stat and save the calibrators to disk for
runtime use.

Caveats:
  - We fit and save UNCONDITIONALLY off the most-recent backtest, which
    means evaluating the model on the same window has leakage. For honest
    metrics on the calibration impact, run a walk-forward backtest after
    deploying. The calibration's purpose is to improve future projections.
  - We require >= 200 samples per stat. Stats with insufficient data are
    skipped (no calibrator written) and apply() falls back to passthrough.
  - We turn off calibration during the FIT pass (via reload+empty cache)
    so we're regressing actuals against RAW projections, not already-
    calibrated ones.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import backtest as bt, projection_cal

GAMES = ROOT / "data/games/games_2026.csv"
BOX   = ROOT / "data/games/box_2026.csv"
SNAP  = ROOT / "data/games/snapshot_2026.json"
MODEL = ROOT / "data/models/team_runs.joblib"
HOLDOUT_DAYS = 28          # 4 weeks gives ~6700 batter rows / ~745 starts
MIN_SAMPLES = 200          # below this, skip (would over-fit)
OUT_PATH = ROOT / "data/models/projection_calibration.joblib"


def _force_no_existing_calibration():
    """Turn off existing calibrators during the fit pass so we regress
    against RAW projections (not already-calibrated ones)."""
    # Move any existing file out of the way temporarily
    if OUT_PATH.exists():
        backup = OUT_PATH.with_suffix(".joblib.refit_backup")
        OUT_PATH.replace(backup)
        return backup
    return None


def _restore_or_remove(backup):
    if backup is not None and backup.exists():
        backup.unlink()


def main():
    print("=" * 60)
    print("Fitting projection calibrators")
    print("=" * 60)

    backup = _force_no_existing_calibration()
    projection_cal.reload()    # ensure fresh empty cache during fit

    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        print("ERROR: scikit-learn IsotonicRegression unavailable.")
        sys.exit(1)

    print(f"\nRunning {HOLDOUT_DAYS}-day backtest for fit data...")
    res = bt.backtest_players(GAMES, BOX, SNAP, MODEL, holdout_days=HOLDOUT_DAYS)
    bat = res["batters"]
    pit = res["pitchers"]
    print(f"  {len(bat)} batter-games, {len(pit)} pitcher starts")

    cals: dict = {}

    # (stat_name, df, proj_col, actual_col)
    targets = [
        ("batter_h",     bat, "proj_h",    "actual_h"),
        ("batter_hr",    bat, "proj_hr",   "actual_hr"),
        ("batter_tb",    bat, "proj_tb",   "actual_tb"),
        ("batter_rbi",   bat, "proj_rbi",  "actual_rbi"),
        ("batter_runs",  bat, "proj_runs", "actual_runs"),
        ("batter_k",     bat, "proj_k",    "actual_k"),
        ("batter_bb",    bat, "proj_bb",   "actual_bb"),
        ("pitcher_k",    pit, "proj_k",    "actual_k"),
        ("pitcher_bb",   pit, "proj_bb",   "actual_bb"),
        ("pitcher_h",    pit, "proj_h",    "actual_h"),
        ("pitcher_er",   pit, "proj_er",   "actual_er"),
        ("pitcher_hr",   pit, "proj_hr",   "actual_hr"),
        ("pitcher_outs", pit, "proj_outs", "actual_outs"),
    ]

    print(f"\n{'stat':<15}  {'n':>4}  {'rate':>5}  {'pre_top':>8}  {'pre_bot':>8}  "
          f"{'cal_top':>8}  {'cal_bot':>8}")
    for name, df, pc, ac in targets:
        if pc not in df.columns or ac not in df.columns:
            continue
        sub = df[[pc, ac]].dropna()
        sub = sub[sub[pc] > 0]
        if len(sub) < MIN_SAMPLES:
            print(f"  {name:<13s}  {len(sub):>4}  (skipped — fewer than {MIN_SAMPLES})")
            continue

        # Rate factor: simple global mean-actual / mean-proj. Captures
        # systematic level shifts (cool pitching weeks, run-environment
        # drift) that isotonic regression handles poorly at the tails.
        # Clamp to [0.80, 1.20] so a freak window can't whipsaw projections.
        proj_mean = float(sub[pc].mean())
        act_mean  = float(sub[ac].mean())
        rate = act_mean / proj_mean if proj_mean > 0 else 1.0
        rate_factor = max(0.80, min(1.20, rate))

        # Isotonic fit on rate-corrected projections — so iso learns only the
        # residual shape after the global level has been corrected.
        x = sub[pc].values * rate_factor
        y = sub[ac].values

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x, y)
        cals[name] = {"iso": iso, "rate_factor": rate_factor}

        # Diagnostics: top/bottom decile bias before vs after
        sub2 = sub.sort_values(pc).reset_index(drop=True)
        n = len(sub2)
        bot = sub2.iloc[: n // 10]
        top = sub2.iloc[-n // 10:]
        pre_top = top[pc].mean() - top[ac].mean()
        pre_bot = bot[pc].mean() - bot[ac].mean()
        # Post-cal: apply rate_factor then iso prediction
        top_rate = top[pc].values * rate_factor
        bot_rate = bot[pc].values * rate_factor
        cal_top = float(np.mean(iso.predict(top_rate))) - top[ac].mean()
        cal_bot = float(np.mean(iso.predict(bot_rate))) - bot[ac].mean()
        print(f"  {name:<13s}  {n:>4}  {rate_factor:>5.2f}  {pre_top:>+8.3f}  {pre_bot:>+8.3f}"
              f"  {cal_top:>+8.3f}  {cal_bot:>+8.3f}")

    # Persist
    import joblib
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cals, OUT_PATH)
    print(f"\nSaved {len(cals)} calibrators -> {OUT_PATH}")
    print("Run-time projections will now apply these via src.projection_cal.")


if __name__ == "__main__":
    main()
