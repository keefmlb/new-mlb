"""Train the team-runs model on 2025 + 2026 combined.

The 2025 dataset is ~6x larger and gives the GLM enough data to fit reliable
coefficients without over-regularization. We hold out the last 7 days of
2026 for evaluation (same as `scripts.train`) so the final-MAE number is
directly comparable.

Run: python -m scripts.train_combined
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import model as mdl

GAMES_2026 = ROOT / "data" / "games" / "games_2026.csv"
# Prior seasons to fold in for training, oldest first. Any that don't exist on
# disk are skipped — pull them with `python -m scripts.build_dataset_history <yr>`.
PRIOR_SEASONS = [2023, 2024, 2025]
MODEL_OUT = ROOT / "data" / "models" / "team_runs.joblib"


def _align(df_old: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    """Fill columns present in the current-schema reference (df_ref = 2026) but
    missing from a prior-season frame, using leak-/era-safe defaults so the GLM
    doesn't learn a spurious season contrast from the fills.
    """
    missing = sorted(set(df_ref.columns) - set(df_old.columns))
    for c in missing:
        # Recent-form gaps -> backfill to the season-equivalent so the
        # (recent - season) gap is 0 (no era-bias signal).
        if c.endswith("_recent"):
            base = c.replace("_recent", "")
            if base in df_old.columns:
                df_old[c] = df_old[base]; continue
        # Roof flags: derive from park_roof (present every season).
        if c in ("home_park_is_dome", "away_park_is_dome", "park_is_dome"):
            df_old[c] = (df_old["park_roof"].astype(str).str.lower() == "dome").astype(int); continue
        if c in ("home_park_is_retractable", "away_park_is_retractable", "park_is_retractable"):
            df_old[c] = (df_old["park_roof"].astype(str).str.lower() == "retractable").astype(int); continue
        if c == "park_pf_h":
            from src import parks as _parks
            df_old[c] = df_old["venue"].astype(str).map(
                lambda n: (_parks.PARKS_BY_NAME.get(n).pf_h if _parks.PARKS_BY_NAME.get(n) else 1.0))
            continue
        if c.endswith("sp_ppi"):
            df_old[c] = 15.5; continue
        if c.endswith("sp_fip_recent") or c.endswith("bp_era_recent"):
            base = c.replace("_recent", "")
            if base in df_old.columns:
                df_old[c] = df_old[base]; continue
        neutral_defaults = {
            "off_sb_pg": 0.60, "off_sf_pg": 0.30, "off_gidp_pg": 0.75,
            "off_sb_net_pg": 0.30, "sp_days_rest": 5.0, "def_oaa": 0.0,
            "is_day_game": 0, "bp_ip_72h": 2.0, "bp_top_rest": 2.5,
            "catcher_framing": 0.0, "catcher_id": 0, "sprint_speed": 27.0,
        }
        matched = False
        for suffix, val in neutral_defaults.items():
            if c.endswith(suffix):
                df_old[c] = val; matched = True; break
        if matched:
            continue
        df_old[c] = df_ref[c].mean() if df_ref[c].dtype != "O" else df_ref[c].iloc[0]
    return df_old


def load_combined_games() -> pd.DataFrame:
    """Load all available prior seasons (PRIOR_SEASONS) + 2026, with prior-
    season columns aligned/backfilled to the 2026 schema (neutral defaults,
    no era-bias contrasts). Shared with scripts/fit_winprob_calibration.py so
    the calibration's walk-forward folds train on exactly the same data as the
    production model."""
    df26 = pd.read_csv(GAMES_2026)
    frames = []
    for yr in PRIOR_SEASONS:
        path = ROOT / "data" / "games" / f"games_{yr}.csv"
        if not path.exists():
            print(f"  {yr}: (not found, skipping — run build_dataset_history {yr})")
            continue
        dfy = pd.read_csv(path)
        dfy = _align(dfy, df26)
        print(f"  {yr}: {len(dfy)} games")
        frames.append(dfy)
    print(f"  2026: {len(df26)} games")
    frames.append(df26)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["is_final"] == True].copy()
    print(f"Combined finals: {len(df)} games across {len(frames)} seasons")
    return df


def main():
    df = load_combined_games()

    model, eval_df = mdl.fit(df, holdout_days=7, use_gbt=True)

    print("\n=== Combined-train backtest (last 7 days of 2026 holdout) ===")
    print(f"Train games        : {model.train_games}")
    print(f"Train MAE          : {model.train_mae:.3f}")
    print(f"Test MAE (ensemble): {eval_df.attrs['test_mae']:.3f}")
    print(f"  GLM-only         : {eval_df.attrs['test_mae_glm']:.3f}")
    if eval_df.attrs.get("test_mae_gbt") is not None:
        print(f"  GBT-only         : {eval_df.attrs['test_mae_gbt']:.3f}")
    print(f"  Ensemble blend w_glm = {eval_df.attrs['blend_w_glm']:.2f}")

    # Game-total MAE
    eval_df = eval_df.copy()
    gt = eval_df.groupby("game_pk").agg(
        actual=("y_runs", "sum"), pred=("pred_runs", "sum")
    )
    print(f"  Game total MAE   : {(gt['actual'] - gt['pred']).abs().mean():.3f}")
    print(f"  Game total RMSE  : {np.sqrt(((gt['actual'] - gt['pred'])**2).mean()):.3f}")

    # Moneyline accuracy
    h = eval_df[eval_df["is_home"] == 1].set_index("game_pk")[["pred_runs", "y_runs"]]
    a = eval_df[eval_df["is_home"] == 0].set_index("game_pk")[["pred_runs", "y_runs"]]
    j = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    j = j[j["y_runs_h"] != j["y_runs_a"]]
    acc = ((j["pred_runs_h"] > j["pred_runs_a"]) == (j["y_runs_h"] > j["y_runs_a"])).mean()
    print(f"  Moneyline pick accuracy: {acc:.3f} (n={len(j)} games)")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\nSaved combined-trained model -> {MODEL_OUT}")

    # Top-magnitude coefficients
    coefs = sorted(zip(mdl.FEATURES, model.glm.coef_), key=lambda x: -abs(x[1]))
    print("\n=== GLM coefficients (z-scored features) ===")
    for n, c in coefs[:12]:
        print(f"  {n:25s}  {c:+.4f}")
    print(f"  intercept              {model.glm.intercept_:+.4f}  -> base lambda = {np.exp(model.glm.intercept_):.3f}")

    # ---- Bootstrap ensemble (N=7 GLM-only resamples) ----
    print("\n[Bootstrap] Training 7 resamples for variance reduction...")
    for i in range(7):
        boot = df.sample(frac=1.0, replace=True, random_state=i)
        bm, _ = mdl.fit(boot, holdout_days=7, use_gbt=False)
        bm.save(MODEL_OUT.parent / f"team_runs_boot_{i}.joblib")
        print(f"  boot {i}: train_mae={bm.train_mae:.3f}")
    print("  Saved team_runs_boot_0..6.joblib")

    # ---- Temporal ensemble (60d and 14d recent-form windows) ----
    print("\n[Temporal] Training 60d and 14d recent-form models...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    max_date = df["date"].max()

    for days, tag, min_rows in [(60, "60d", 300), (14, "14d", 100)]:
        cutoff = max_date - pd.Timedelta(days=days)
        sub = df[df["date"] >= cutoff].copy()
        if len(sub) < min_rows:
            print(f"  {tag}: only {len(sub)} rows — skipping (need {min_rows})")
            continue
        hd = min(7, max(2, len(sub) // 40))
        tm, te = mdl.fit(sub, holdout_days=hd, use_gbt=False)
        tm.save(MODEL_OUT.parent / f"team_runs_{tag}.joblib")
        print(f"  {tag}: {len(sub)} rows, test_mae={te.attrs['test_mae']:.3f} -> team_runs_{tag}.joblib")

    print("\nDone. Ensemble = main + 7 bootstrap + up to 2 temporal models.")


if __name__ == "__main__":
    main()
