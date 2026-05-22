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

GAMES_2025 = ROOT / "data" / "games" / "games_2025.csv"
GAMES_2026 = ROOT / "data" / "games" / "games_2026.csv"
MODEL_OUT = ROOT / "data" / "models" / "team_runs.joblib"


def main():
    df25 = pd.read_csv(GAMES_2025)
    df26 = pd.read_csv(GAMES_2026)
    print(f"2025: {len(df25)}  |  2026: {len(df26)}")

    # Align columns: take only those in both
    common = sorted(set(df25.columns) & set(df26.columns))
    missing_in_25 = sorted(set(df26.columns) - set(df25.columns))
    if missing_in_25:
        print(f"  Filling missing-in-2025 columns with sensible defaults: {missing_in_25}")
        for c in missing_in_25:
            if c.endswith("_recent"):
                # Fall back to season equivalent. e.g. home_off_rpg_recent -> home_off_rpg
                base = c.replace("_recent", "")
                if base in df25.columns:
                    df25[c] = df25[base]
                    continue
            # Roof flags: derive from park_roof (present in 2025) rather than
            # filling with mean. Otherwise the GLM learns "dome=1 means 2026"
            # via era bias, not the real roof effect.
            if c in ("home_park_is_dome", "away_park_is_dome", "park_is_dome"):
                df25[c] = (df25["park_roof"].astype(str).str.lower() == "dome").astype(int)
                continue
            if c in ("home_park_is_retractable", "away_park_is_retractable", "park_is_retractable"):
                df25[c] = (df25["park_roof"].astype(str).str.lower() == "retractable").astype(int)
                continue
            # park_pf_h: look up per-venue from parks registry
            if c == "park_pf_h":
                from src import parks as _parks
                def _pf_h(name: str) -> float:
                    p = _parks.PARKS_BY_NAME.get(name)
                    return p.pf_h if p else 1.0
                df25[c] = df25["venue"].astype(str).map(_pf_h)
                continue
            # Sequential offense / days_rest / day_game / OAA — fill with
            # NEUTRAL league-average defaults rather than 2026 mean (which
            # would create era contrast). 2025 rows become "neutral baseline"
            # and only 2026 variation drives the coefficient learning.
            neutral_defaults = {
                "off_sb_pg":     0.60, "off_sf_pg":     0.30,
                "off_gidp_pg":   0.75, "off_sb_net_pg": 0.30,
                "sp_days_rest":  5.0,  "def_oaa":       0.0,
                "is_day_game":   0,
                "bp_ip_72h":     2.0,    # league avg ~2 IP/team/day
                "bp_top_rest":   2.5,    # avg top-relief rest
                "catcher_framing": 0.0,  # neutral (league average framing)
                "catcher_id":      0,
            }
            # Recent SP FIP: backfill to season FIP so the recent-gap = 0 for
            # 2025 rows (i.e. no era-bias signal).
            if c.endswith("sp_fip_recent"):
                base = c.replace("_recent", "")
                if base in df25.columns:
                    df25[c] = df25[base]
                    continue
            # Starter pitches-per-inning: 15.5 league average for 2025
            if c.endswith("sp_ppi"):
                df25[c] = 15.5
                continue
            # Bullpen recent ERA: backfill to season bp_era so gap=0 (no era contrast)
            if c.endswith("bp_era_recent"):
                base = c.replace("_recent", "")
                if base in df25.columns:
                    df25[c] = df25[base]
                    continue
            matched = False
            for suffix, val in neutral_defaults.items():
                if c.endswith(suffix):
                    df25[c] = val
                    matched = True
                    break
            if matched:
                continue
            df25[c] = df26[c].mean() if df26[c].dtype != "O" else df26[c].iloc[0]

    df = pd.concat([df25, df26], ignore_index=True)
    df = df[df["is_final"] == True].copy()
    print(f"Combined finals: {len(df)} games")

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
