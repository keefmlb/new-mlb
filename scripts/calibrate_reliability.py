"""Calibrate stat_reliability.json from backtest data.

For each prop market, computes Pearson r between per-game analytical
projections and actuals on a holdout slice (last 20% of dates). Combines
2025 + 2026 data when both CSVs are present so the Pearson estimates are
more stable (~3x the sample). Also evaluates the team-runs model on
games_2026.csv for game-line reliability.

Higher correlation → higher reliability weight → the model's edge on that
market is trusted more when scoring bets.

Run after train_props, train_combined, and (optionally) build_props_2025:
    python -m scripts.calibrate_reliability

Writes updated weights to data/models/stat_reliability.json and prints a
before/after comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.value import write_stat_reliability, _get_stat_reliability

BAT_CSV_2026  = ROOT / "data" / "games" / "props_bat_2026.csv"
BAT_CSV_2025  = ROOT / "data" / "games" / "props_bat_2025.csv"
PIT_CSV_2026  = ROOT / "data" / "games" / "props_pit_2026.csv"
PIT_CSV_2025  = ROOT / "data" / "games" / "props_pit_2025.csv"
GAMES_CSV     = ROOT / "data" / "games" / "games_2026.csv"

# Keep backward-compat names used in the rest of the script
BAT_CSV = BAT_CSV_2026
PIT_CSV = PIT_CSV_2026

HOLDOUT_FRAC = 0.20   # last 20% of dates — never seen by train_props


def _load_combined(csv_2026: Path, csv_2025: Path) -> pd.DataFrame | None:
    """Load 2026 CSV and (if present) prepend 2025 data. Returns None if neither exists."""
    frames = []
    for p, year in [(csv_2025, 2025), (csv_2026, 2026)]:
        if p.exists():
            df = pd.read_csv(p)
            df["_year"] = year
            frames.append(df)
        else:
            print(f"  (no {p.name} — skipping {year})")
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined

# (proj_col, actual_col, market_key)
BATTER_SPECS = [
    ("proj_h",    "h",      "prop_hits"),
    ("proj_hr",   "hr",     "prop_hr"),
    ("proj_tb",   "tb",     "prop_tb"),
    ("proj_rbi",  "rbi",    "prop_rbi"),
    ("proj_runs", "runs_b", "prop_runs"),
    ("proj_k",    "k_b",    "prop_k"),
    ("proj_bb",   "bb_b",   "prop_bb"),
    ("proj_hrr",  "hrr",    "prop_hrr"),
]

PITCHER_SPECS = [
    ("proj_k",        "k_p",  "prop_pitcher_k"),
    ("expected_outs", "outs", "prop_pitcher_outs"),
    ("proj_er",       "er",   "prop_pitcher_er"),
    ("proj_h",        "h_p",  "prop_pitcher_h"),
    ("proj_bb",       "bb_p", "prop_pitcher_bb"),
    ("proj_hr",       "hr_p", "prop_pitcher_hr"),
]


def _holdout(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    return df.iloc[int(len(df) * (1 - HOLDOUT_FRAC)):]


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 20:
        return None
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 20:
        return None
    denom = np.std(a) * np.std(b)
    if denom == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _r_to_reliability(r: float | None) -> float | None:
    """Map Pearson r → [0.05, 0.90] reliability weight.

    Negative r (model anticorrelated with outcome) → treated as 0.
    We use r directly as the weight — it's already on a [0,1] scale
    and empirically matches the intuition that ~0.5 r means the model
    explains ~25% of variance (a decent prop model).
    """
    if r is None:
        return None
    return float(np.clip(max(r, 0.0), 0.05, 0.90))


def _arrow(new: float, old: float | str) -> str:
    if not isinstance(old, float):
        return ""
    if new > old + 0.01:
        return "(+)"
    if new < old - 0.01:
        return "(-)"
    return "  ~"


def main():
    old = dict(_get_stat_reliability())
    new_weights: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Batter props                                                         #
    # ------------------------------------------------------------------ #
    bat_combined = _load_combined(BAT_CSV_2026, BAT_CSV_2025)
    if bat_combined is None:
        print(f"WARNING: no batter props CSVs found — run scripts.train_props first.")
    else:
        # Derive the Hits+Runs+RBIs market (a real book market, summed from
        # three correlated components) so it gets a reliability weight too.
        _a = [c for c in ("h", "runs_b", "rbi") if c in bat_combined.columns]
        _p = [c for c in ("proj_h", "proj_runs", "proj_rbi") if c in bat_combined.columns]
        if len(_a) == 3 and len(_p) == 3:
            bat_combined["hrr"] = bat_combined[_a].sum(axis=1)
            bat_combined["proj_hrr"] = bat_combined[_p].sum(axis=1)
        years = sorted(bat_combined["_year"].unique())
        bat_ho = _holdout(bat_combined)
        print(f"Batter holdout: {len(bat_ho)} player-games  "
              f"({bat_ho['date'].min()} to {bat_ho['date'].max()})  "
              f"[years: {years}]\n")
        print(f"  {'Market':<24}  {'r':>6}  {'old':>6}  {'new':>6}")
        print(f"  {'-'*24}  {'-'*6}  {'-'*6}  {'-'*6}")
        for proj_col, actual_col, market in BATTER_SPECS:
            if proj_col not in bat_ho.columns or actual_col not in bat_ho.columns:
                continue
            r   = _pearson_r(bat_ho[proj_col].values, bat_ho[actual_col].values)
            rel = _r_to_reliability(r)
            if rel is None:
                print(f"  {market:<24}  {'n/a':>6}  {old.get(market, '—')!s:>6}  {'(skip)':>6}")
                continue
            new_weights[market] = rel
            print(f"  {market:<24}  {r:+.3f}  {old.get(market, '—')!s:>6}  "
                  f"{rel:.2f} {_arrow(rel, old.get(market, '—'))}")

    # ------------------------------------------------------------------ #
    # Pitcher props                                                        #
    # ------------------------------------------------------------------ #
    pit_combined = _load_combined(PIT_CSV_2026, PIT_CSV_2025)
    if pit_combined is None:
        print(f"\nWARNING: no pitcher props CSVs found — run scripts.train_props first.")
    else:
        years = sorted(pit_combined["_year"].unique())
        pit_ho = _holdout(pit_combined)
        print(f"\nPitcher holdout: {len(pit_ho)} starter-games  "
              f"({pit_ho['date'].min()} to {pit_ho['date'].max()})  "
              f"[years: {years}]\n")
        print(f"  {'Market':<27}  {'r':>6}  {'old':>6}  {'new':>6}")
        print(f"  {'-'*27}  {'-'*6}  {'-'*6}  {'-'*6}")
        for proj_col, actual_col, market in PITCHER_SPECS:
            if proj_col not in pit_ho.columns or actual_col not in pit_ho.columns:
                continue
            r   = _pearson_r(pit_ho[proj_col].values, pit_ho[actual_col].values)
            rel = _r_to_reliability(r)
            if rel is None:
                print(f"  {market:<27}  {'n/a':>6}  {old.get(market, '—')!s:>6}  {'(skip)':>6}")
                continue
            new_weights[market] = rel
            print(f"  {market:<27}  {r:+.3f}  {old.get(market, '—')!s:>6}  "
                  f"{rel:.2f} {_arrow(rel, old.get(market, '—'))}")

    # ------------------------------------------------------------------ #
    # Game lines — team-runs model                                         #
    # ------------------------------------------------------------------ #
    if not GAMES_CSV.exists():
        print(f"\nWARNING: {GAMES_CSV.name} not found.")
    else:
        try:
            from src import model as mdl
            games   = pd.read_csv(GAMES_CSV)
            final   = games[games["is_final"] == True].copy()
            long    = mdl.long_form(final).dropna(subset=["y_runs"])
            long    = long.sort_values("date").reset_index(drop=True)
            ho      = long.iloc[int(len(long) * (1 - HOLDOUT_FRAC)):]

            team_model = mdl.TeamScoreModel.load(
                ROOT / "data" / "models" / "team_runs.joblib"
            )
            ho = ho.copy()
            ho["pred_runs"] = team_model.predict_runs(ho)

            r_runs = _pearson_r(ho["pred_runs"].values, ho["y_runs"].values)

            # ML accuracy (pick the winning team)
            hm = ho[ho["is_home"] == 1].set_index("game_pk")[["pred_runs", "y_runs"]]
            aw = ho[ho["is_home"] == 0].set_index("game_pk")[["pred_runs", "y_runs"]]
            j  = hm.join(aw, lsuffix="_h", rsuffix="_a").dropna()
            j  = j[j["y_runs_h"] != j["y_runs_a"]]
            ml_acc = float(
                ((j["pred_runs_h"] > j["pred_runs_a"]) ==
                 (j["y_runs_h"]   > j["y_runs_a"])).mean()
            ) if len(j) else 0.5

            # ML reliability: scale [50%→0, 60%→0.4] and clip
            ml_rel  = float(np.clip((ml_acc - 0.50) * 4.0, 0.05, 0.60))
            # Total / runs reliability: r of per-team runs, discounted slightly
            tot_rel = float(np.clip(max(r_runs or 0, 0) * 0.8, 0.05, 0.60))

            new_weights["moneyline"] = ml_rel
            new_weights["total"]     = tot_rel
            new_weights["run_line"]  = ml_rel

            print(f"\nGame lines (n={len(j)} games in holdout):")
            print(f"  Runs Pearson r={r_runs:+.3f}   ML accuracy={ml_acc:.3f}")
            print(f"  {'Market':<12}  {'old':>6}  {'new':>6}")
            print(f"  {'-'*12}  {'-'*6}  {'-'*6}")
            for mkt, val in [("moneyline", ml_rel), ("total", tot_rel), ("run_line", ml_rel)]:
                print(f"  {mkt:<12}  {old.get(mkt, '—')!s:>6}  "
                      f"{val:.2f} {_arrow(val, old.get(mkt, '—'))}")
        except Exception as e:
            print(f"\nGame line calibration skipped: {e}")

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #
    if not new_weights:
        print("\nNothing to write — no data files found.")
        return

    write_stat_reliability(new_weights)
    print(f"\nWrote {len(new_weights)} weights to data/models/stat_reliability.json")
    print("  Re-run 'python -m streamlit run app.py' (or hit Refresh) to apply.")


if __name__ == "__main__":
    main()
