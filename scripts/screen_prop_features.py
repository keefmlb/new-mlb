"""Screen candidate game-context features against prop projection residuals.

The prop feature rows carry player skill, matchup, park, and weather — but
several game-day facts in games_2026.csv never made it in: the plate umpire's
K tendency, catcher framing, team defense (OAA), starter days-rest, bullpen
fatigue, and day/night. Each is knowable before first pitch (no leakage).

For each candidate this script reports the Pearson r against the residual
(actual − analytical projection) of every plausibly-related stat, plus the
top-vs-bottom-quartile residual gap in stat units — the practical effect
size. Candidates only earn plumbing into the feature rows if the correlation
is significant AND the quartile gap is worth chasing.

Run:  python -m scripts.screen_prop_features
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GAMES = ROOT / "data" / "games" / "games_2026.csv"
BAT   = ROOT / "data" / "games" / "props_bat_2026.csv"
PIT   = ROOT / "data" / "games" / "props_pit_2026.csv"


def _side_col(df: pd.DataFrame, home_col: str, away_col: str, own: bool) -> pd.Series:
    """Per-row game column resolved by side. own=True -> the player's own team's
    column; own=False -> the opposing team's column."""
    is_home = df["side"] == "home"
    pick_home = is_home if own else ~is_home
    return np.where(pick_home, df[home_col], df[away_col])


def _screen(df: pd.DataFrame, resid_col: str, cand: pd.Series, label: str,
            min_n: int = 500) -> dict | None:
    ok = df[resid_col].notna() & pd.Series(cand).notna()
    if ok.sum() < min_n:
        return None
    r_ = np.asarray(df.loc[ok, resid_col], dtype=float)
    c_ = np.asarray(pd.Series(cand)[ok], dtype=float)
    if np.std(c_) == 0:
        return None
    r = float(np.corrcoef(c_, r_)[0, 1])
    n = int(ok.sum())
    # ~2-sigma threshold for a correlation: 2/sqrt(n)
    sig = abs(r) > 2.0 / np.sqrt(n)
    q1, q3 = np.quantile(c_, [0.25, 0.75])
    lo = r_[c_ <= q1].mean()
    hi = r_[c_ >= q3].mean()
    return {"label": label, "n": n, "r": r, "sig": sig,
            "q1_resid": lo, "q4_resid": hi, "gap": hi - lo}


def main():
    games = pd.read_csv(GAMES)
    bat = pd.read_csv(BAT).merge(games, on="game_pk", suffixes=("", "_g"))
    pit = pd.read_csv(PIT).merge(games, on="game_pk", suffixes=("", "_g"))
    print(f"batter rows: {len(bat)}   pitcher rows: {len(pit)}")

    # Residuals (actual − analytical projection)
    for c_act, c_proj in [("k_b", "proj_k"), ("h", "proj_h"), ("tb", "proj_tb"),
                          ("hr", "proj_hr"), ("bb_b", "proj_bb"),
                          ("runs_b", "proj_runs"), ("rbi", "proj_rbi")]:
        bat[f"res_{c_act}"] = bat[c_act] - bat[c_proj]
    for c_act, c_proj in [("k_p", "proj_k"), ("outs", "expected_outs"),
                          ("h_p", "proj_h"), ("er", "proj_er"),
                          ("bb_p", "proj_bb"), ("hr_p", "proj_hr")]:
        pit[f"res_{c_act}"] = pit[c_act] - pit[c_proj]

    rows: list[dict] = []

    # ---- Batter candidates ----
    bat_c = {
        "ump_k_mult (batter K)":         ("res_k_b",  bat["ump_k_mult"]),
        "opp_catcher_framing (batter K)": ("res_k_b", _side_col(bat, "home_catcher_framing", "away_catcher_framing", own=False)),
        "opp_def_oaa (batter H)":        ("res_h",    _side_col(bat, "home_def_oaa", "away_def_oaa", own=False)),
        "opp_def_oaa (batter TB)":       ("res_tb",   _side_col(bat, "home_def_oaa", "away_def_oaa", own=False)),
        "is_day_game (batter H)":        ("res_h",    bat["is_day_game"].astype(float)),
        "is_day_game (batter HR)":       ("res_hr",   bat["is_day_game"].astype(float)),
        "opp_sp_days_rest (batter TB)":  ("res_tb",   _side_col(bat, "home_sp_days_rest", "away_sp_days_rest", own=False)),
        "opp_bp_ip_72h (batter RBI)":    ("res_rbi",  _side_col(bat, "home_bp_ip_72h", "away_bp_ip_72h", own=False)),
        "opp_bp_era_recent (batter RBI)": ("res_rbi", _side_col(bat, "home_bp_era_recent", "away_bp_era_recent", own=False)),
        "opp_bp_era_recent (batter R)":  ("res_runs_b", _side_col(bat, "home_bp_era_recent", "away_bp_era_recent", own=False)),
    }
    for label, (res, cand) in bat_c.items():
        out = _screen(bat, res, cand, label)
        if out:
            rows.append(out)

    # ---- Pitcher candidates ----
    pit_c = {
        "ump_k_mult (pitcher K)":        ("res_k_p",  pit["ump_k_mult"]),
        "own_catcher_framing (pitcher K)": ("res_k_p", _side_col(pit, "home_catcher_framing", "away_catcher_framing", own=True)),
        "own_def_oaa (pitcher H)":       ("res_h_p",  _side_col(pit, "home_def_oaa", "away_def_oaa", own=True)),
        "own_def_oaa (pitcher ER)":      ("res_er",   _side_col(pit, "home_def_oaa", "away_def_oaa", own=True)),
        "sp_days_rest (pitcher OUTS)":   ("res_outs", _side_col(pit, "home_sp_days_rest", "away_sp_days_rest", own=True)),
        "sp_days_rest (pitcher K)":      ("res_k_p",  _side_col(pit, "home_sp_days_rest", "away_sp_days_rest", own=True)),
        "own_bp_ip_72h (pitcher OUTS)":  ("res_outs", _side_col(pit, "home_bp_ip_72h", "away_bp_ip_72h", own=True)),
        "own_bp_top_rest (pitcher OUTS)": ("res_outs", _side_col(pit, "home_bp_top_rest", "away_bp_top_rest", own=True)),
        "sp_ppi (pitcher OUTS)":         ("res_outs", _side_col(pit, "home_sp_ppi", "away_sp_ppi", own=True)),
        "is_day_game (pitcher K)":       ("res_k_p",  pit["is_day_game"].astype(float)),
    }
    for label, (res, cand) in pit_c.items():
        out = _screen(pit, res, cand, label)
        if out:
            rows.append(out)

    print(f"\n{'candidate':40s} {'n':>6s} {'r':>7s} {'sig':>4s} "
          f"{'Q1 resid':>9s} {'Q4 resid':>9s} {'Q4-Q1':>7s}")
    print("-" * 90)
    for o in sorted(rows, key=lambda o: -abs(o["r"])):
        print(f"{o['label']:40s} {o['n']:6d} {o['r']:+7.3f} "
              f"{'*' if o['sig'] else '':>4s} "
              f"{o['q1_resid']:+9.3f} {o['q4_resid']:+9.3f} {o['gap']:+7.3f}")
    print("\n* = |r| > 2/sqrt(n) (~95% significance). Positive r means higher")
    print("  candidate values -> projection UNDER-shoots (actual > projected).")


if __name__ == "__main__":
    main()
