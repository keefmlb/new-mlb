"""Train per-stat ML projection models on the 2026 boxscore dataset.

Input:
  data/games/games_2026.csv            (game features — built with leak-free snapshots)
  data/games/box_2026.csv              (per-player per-game stats)
  data/games/snapshots_2026/           (weekly Monday snapshots from build_dataset.py)
  data/models/team_runs.joblib         (already trained)

For each completed game we:
  1) Look up the prior-Monday snapshot for that game date (same snapshots used
     to build games_2026.csv) so player stats reflect only pre-game information.
  2) Compute the team-runs model's predicted runs for both sides.
  3) Build the analytical batter/pitcher projection for each active player.
  4) Assemble a feature row per (player, game) and train a HistGBT per stat.

Run: python -m scripts.train_props
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as feats, model as mdl, parks, prop_models, projections as proj, statcast as sc

GAMES  = ROOT / "data" / "games" / "games_2026.csv"
BOX    = ROOT / "data" / "games" / "box_2026.csv"
SNAPS_DIR = ROOT / "data" / "games" / "snapshots_2026"
TEAM_MODEL = ROOT / "data" / "models" / "team_runs.joblib"
OUT    = ROOT / "data" / "models"


def _int_keys(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def _load_weekly_snapshots() -> dict[str, dict]:
    """Load all weekly snapshots from disk. Returns {iso_date: snap_dict}."""
    snaps: dict[str, dict] = {}
    if not SNAPS_DIR.exists():
        return snaps
    for f in sorted(SNAPS_DIR.glob("*.json")):
        try:
            snaps[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return snaps


def _lookup_snap(snaps: dict[str, dict], game_date: str) -> dict | None:
    """Return the most recent snapshot dated strictly before game_date."""
    keys = sorted(snaps.keys())
    chosen = None
    for k in keys:
        if k < game_date:
            chosen = k
        else:
            break
    return snaps[chosen] if chosen else (snaps[keys[0]] if keys else None)


def main():
    print("Loading data...")
    games = pd.read_csv(GAMES)
    box = pd.read_csv(BOX)

    final = games[games["is_final"] == True].copy().sort_values("date")
    print(f"  {len(final)} final games, {len(box)} player-game rows")

    print("Loading weekly snapshots...")
    snaps = _load_weekly_snapshots()
    if not snaps:
        print("  WARNING: no weekly snapshots found in", SNAPS_DIR)
        print("  Run `python -m scripts.build_dataset` first to generate leak-free snapshots.")
        print("  Falling back to global snapshot (leaky — for debugging only).")
        snap_path = ROOT / "data" / "games" / "snapshot_2026.json"
        fallback = json.loads(snap_path.read_text(encoding="utf-8"))
        snaps = {"0000-01-01": fallback}
    else:
        print(f"  {len(snaps)} snapshots ({min(snaps)} to {max(snaps)})")

    print("Loading team-runs model + computing per-game predicted runs...")
    team_model = mdl.TeamScoreModel.load(TEAM_MODEL)
    long = mdl.long_form(final)
    long["pred_runs"] = team_model.predict_runs(long)
    pred_by_gs: dict[tuple, float] = {}
    for _, r in long.iterrows():
        side = "home" if r["is_home"] == 1 else "away"
        pred_by_gs[(int(r["game_pk"]), side)] = float(r["pred_runs"])

    print("Fetching Statcast batter + pitcher stats (barrel%, hard-hit%, whiff%)...")
    try:
        sc_bat = sc.get_batter_stats(2026)
        sc_pit = sc.get_pitcher_stats(2026)
        sc_arsenal = sc.get_pitcher_arsenal(2026)
        print(f"  {len(sc_bat)} batters, {len(sc_pit)} pitchers, {len(sc_arsenal)} arsenals")
    except Exception as e:
        print(f"  WARNING: Statcast fetch failed ({e}), using league-average defaults")
        sc_bat, sc_pit, sc_arsenal = {}, {}, {}

    print("Building feature rows for batters and starters...")
    bat_rows: list[dict] = []
    pit_rows: list[dict] = []

    for _, g in final.iterrows():
        gpk = int(g["game_pk"])
        gdate = str(g["date"])
        venue = g["venue"]
        park = parks.get_park(venue)
        wadj = {
            "runs_mult":      float(g["runs_mult"]),
            "hr_mult":        float(g["hr_mult"]),
            "wind_to_cf_mph": float(g.get("wind_to_cf_mph", 0.0)),
            "temp_f":         float(g.get("temp_f", 70.0)),
        }

        # Use the pre-game snapshot so player stats don't include this game's outcome
        snap = _lookup_snap(snaps, gdate)
        if snap is None:
            continue

        bat_stats  = _int_keys(snap.get("batter_stats", {}))
        pit_stats  = _int_keys(snap.get("pitcher_stats", {}))
        bat_recent = _int_keys(snap.get("batter_stats_recent", {}))
        pit_recent = _int_keys(snap.get("pitcher_stats_recent", {}))
        bat_vs_l   = _int_keys(snap.get("bat_vs_l", {}))
        bat_vs_r   = _int_keys(snap.get("bat_vs_r", {}))
        pit_vs_l   = _int_keys(snap.get("pit_vs_l", {}))
        pit_vs_r   = _int_keys(snap.get("pit_vs_r", {}))
        bat_sides  = {int(k): v for k, v in snap.get("bat_sides", {}).items()} if "bat_sides" in snap else {}
        pit_throws = {int(k): v for k, v in snap.get("pit_throws", {}).items()} if "pit_throws" in snap else {}
        team_off   = _int_keys(snap.get("team_off", {}))

        for side in ("home", "away"):
            tid   = int(g["home_team_id"]) if side == "home" else int(g["away_team_id"])
            otid  = int(g["away_team_id"]) if side == "home" else int(g["home_team_id"])
            sp_self_raw = g["home_sp_id"] if side == "home" else g["away_sp_id"]
            sp_opp_raw  = g["away_sp_id"] if side == "home" else g["home_sp_id"]
            sp_self = int(sp_self_raw) if pd.notna(sp_self_raw) else None
            sp_opp  = int(sp_opp_raw)  if pd.notna(sp_opp_raw)  else None

            team_pred = pred_by_gs.get((gpk, side))
            opp_pred  = pred_by_gs.get((gpk, "away" if side == "home" else "home"))
            if team_pred is None or opp_pred is None:
                continue

            opp_sp_q   = (
                feats.pitcher_quality_index(pit_stats.get(sp_opp, {}),
                                            sc_stats=sc_pit.get(sp_opp))
                if sp_opp else feats.pitcher_quality_index({})
            )
            opp_off_idx = feats.team_offense_index(team_off.get(otid, {}))

            actual_b = box[(box["game_pk"] == gpk) & (box["side"] == side) & (box["pa"] > 0)]
            actual_b = actual_b.sort_values("pa", ascending=False).head(9).reset_index(drop=True)

            # For pitcher projections we need the OPPOSING lineup's K% (the
            # nine batters this starter actually faced). Mirrors the live
            # predict_core override so historical projections match the live
            # distribution and the dispersion fit is honest.
            other_side = "away" if side == "home" else "home"
            opp_b = box[(box["game_pk"] == gpk) & (box["side"] == other_side) & (box["pa"] > 0)]
            opp_b = opp_b.sort_values("pa", ascending=False).head(9)
            opp_lineup_ids = [int(p) for p in opp_b["player_id"].tolist() if pd.notna(p)]
            if opp_lineup_ids:
                _lk = proj.lineup_k_pct(opp_lineup_ids, bat_stats)
                opp_off_idx = dict(opp_off_idx)
                opp_off_idx["k_pct"] = _lk

            for order_idx, row in actual_b.iterrows():
                pid = int(row["player_id"])
                bs  = bat_stats.get(pid, {"player_id": pid, "name": row["name"], "team_id": tid})
                rs  = bat_recent.get(pid)
                pl  = proj.resolve_platoon(pid, sp_opp, bat_sides, pit_throws, bat_vs_l, bat_vs_r)
                bproj = proj.project_batter(
                    bs, order_idx + 1, team_pred, opp_sp_q, park, wadj,
                    recent_stats=rs, ml_blend=0,
                    bat_side=pl["bat_side"],
                    opp_pit_throws=pl["opp_pit_throws"],
                    bat_split=pl["bat_split"],
                    is_switch=pl.get("is_switch", False),
                )
                feat_row = prop_models.batter_feature_row(bproj, bs, rs, opp_sp_q, park, wadj, team_pred,
                                                          sc_stats=sc_bat.get(pid))
                # batter_feature_row already includes sc fields (xwoba/xba/exit_velo
                # added May 6 2026); the sc_stats kwarg is the single feed.
                feat_row.update({
                    "game_pk": gpk, "date": gdate, "side": side,
                    "player_id": pid, "name": row["name"],
                    "h":      float(row["h"]),
                    "hr":     float(row["hr"]),
                    "doubles": float(row["doubles"]),
                    "tb":     float(row["tb"]),
                    "rbi":    float(row["rbi"]),
                    "runs_b": float(row["runs_b"]),
                    "k_b":    float(row["k_b"]),
                    "bb_b":   float(row["bb_b"]),
                    "pa":     float(row["pa"]),
                })
                bat_rows.append(feat_row)

            actual_p = box[(box["game_pk"] == gpk) & (box["side"] == side) & (box["started"] == True)]
            for _, prow in actual_p.iterrows():
                pid = int(prow["player_id"])
                ps  = pit_stats.get(pid, {"player_id": pid, "name": prow["name"], "team_id": tid})
                rs  = pit_recent.get(pid)
                # Lineup-mix-weighted platoon split for this pitcher vs the
                # specific 9 batters they actually faced. Mirrors live predict.
                from src import lineup_features as _lf
                _pthrows = (pit_throws.get(pid) or "R").upper()
                _split_stats = _lf.pitcher_split_vs_lineup(
                    pid, opp_lineup_ids, bat_sides, _pthrows,
                    pit_vs_l, pit_vs_r,
                ) if opp_lineup_ids else None
                pproj = proj.project_pitcher(ps, tid, opp_off_idx, opp_pred, park, wadj,
                                             recent_stats=rs, ml_blend=0,
                                             sc_stats=sc_pit.get(pid),
                                             split_stats=_split_stats,
                                             arsenal=sc_arsenal.get(pid))
                feat_row = prop_models.pitcher_feature_row(pproj, ps, rs, opp_off_idx, park, wadj, opp_pred,
                                                            sc_stats=sc_pit.get(pid),
                                                            split_stats=_split_stats,
                                                            arsenal=sc_arsenal.get(pid))
                feat_row.update({
                    "game_pk": gpk, "date": gdate, "side": side,
                    "player_id": pid, "name": prow["name"],
                    "k_p":  float(prow["k_p"]),
                    "outs": float(prow["outs"]),
                    "er":   float(prow["er"]),
                    "h_p":  float(prow["h_p"]),
                    "bb_p": float(prow["bb_p"]),
                    "hr_p": float(prow["hr_p"]),
                })
                pit_rows.append(feat_row)

    bat_df = pd.DataFrame(bat_rows)
    pit_df = pd.DataFrame(pit_rows)
    print(f"  batters dataset: {len(bat_df)} rows")
    print(f"  pitchers dataset: {len(pit_df)} rows")

    print("\nTraining models (per stat, GBT-Poisson)...")
    prop_models.train_all(bat_df, pit_df, OUT)

    bat_df.to_csv(ROOT / "data" / "games" / "props_bat_2026.csv", index=False)
    pit_df.to_csv(ROOT / "data" / "games" / "props_pit_2026.csv", index=False)
    print("\nSaved models -> data/models/prop_*.joblib")
    print("Saved feature datasets -> data/games/props_*_2026.csv")


if __name__ == "__main__":
    main()
