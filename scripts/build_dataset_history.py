"""Build a prior-season historical dataset for full-season training/backtest.

Year-parameterized version of build_dataset_2025.py. Pulls every game +
boxscore for the chosen season, using weekly season-to-date stat snapshots so
each game's features come only from data that existed before it was played
(leak-free temporal split). Everything is disk-cached so re-runs are instant.

Run:
    python -m scripts.build_dataset_history 2024
    python -m scripts.build_dataset_history 2023
"""
from __future__ import annotations
import csv
import json
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import mlb_api, features as feats, lineup_features as lf

# Regular-season bounds per year (opening day -> season end).
_SEASON_BOUNDS = {
    2022: (date(2022, 4, 7),  date(2022, 10, 5)),
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 28), date(2024, 9, 29)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
}

_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
if _year not in _SEASON_BOUNDS:
    raise SystemExit(f"No season bounds for {_year}; add to _SEASON_BOUNDS.")
SEASON = _year
SEASON_START, SEASON_END = _SEASON_BOUNDS[_year]


def weekly_snapshots(season: int, start: date, end: date) -> dict[str, dict]:
    """Pull season-to-date team/player stats for every Monday in [start, end].

    Returns {iso_date: {team_off, team_pit, pit, bat}} keyed by snapshot date.
    Each game uses the most recent snapshot BEFORE its game date.
    """
    snaps: dict[str, dict] = {}
    d = start
    while d <= end:
        if d.weekday() == 0 or d == start:        # Mondays + first day
            print(f"  snap {d.isoformat()}...")
            try:
                team_off = mlb_api.team_stats_by_range(season, "hitting", start, d)
                team_pit = mlb_api.team_stats_by_range(season, "pitching", start, d)
                bat = mlb_api.player_stats_by_range(season, "hitting", start, d)
                pit = mlb_api.player_stats_by_range(season, "pitching", start, d)
                snaps[d.isoformat()] = {
                    "team_off": team_off, "team_pit": team_pit,
                    "bat": bat, "pit": pit,
                }
            except Exception as e:
                print(f"    snap failed: {e}")
        d += timedelta(days=1)
    return snaps


def lookup_snapshot(snaps: dict[str, dict], game_date: str) -> dict | None:
    """Find the snapshot for the most recent Monday on or before game_date."""
    keys = sorted(snaps.keys())
    chosen = None
    for k in keys:
        if k < game_date:
            chosen = k
        else:
            break
    return snaps.get(chosen) if chosen else (snaps[keys[0]] if keys else None)


def main():
    print(f"Pulling 2025 schedule {SEASON_START} - {SEASON_END}...")
    games = mlb_api.schedule_range(SEASON_START, SEASON_END)
    final_games = [g for g in games if (g.get("status") or {}).get("codedGameState") == "F"]
    print(f"  total: {len(games)}  final: {len(final_games)}")

    print(f"Pulling weekly stat snapshots ({(SEASON_END - SEASON_START).days // 7 + 1} weeks)...")
    snaps = weekly_snapshots(SEASON, SEASON_START, SEASON_END)
    print(f"  {len(snaps)} snapshots loaded")

    print("Pulling 2025 platoon splits (one-time, cached)...")
    bat_vs_l = mlb_api.player_splits_bulk(SEASON, "hitting", "l")
    bat_vs_r = mlb_api.player_splits_bulk(SEASON, "hitting", "r")
    bat_sides = mlb_api.batter_bats_bulk(SEASON)
    pit_throws = mlb_api.pitcher_throws_bulk(SEASON)
    bat_vs_l = {int(k): v for k, v in bat_vs_l.items()}
    bat_vs_r = {int(k): v for k, v in bat_vs_r.items()}
    bat_sides = {int(k): v for k, v in bat_sides.items()}
    pit_throws = {int(k): v for k, v in pit_throws.items()}
    print(f"  splits: vL={len(bat_vs_l)} vR={len(bat_vs_r)} sides={len(bat_sides)} throws={len(pit_throws)}")

    out_dir = ROOT / "data" / "games"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building feature rows + boxscores ({len(final_games)} final games)...")
    feat_rows: list[dict] = []
    box_rows: list[dict] = []
    for i, g in enumerate(final_games):
        if i % 200 == 0:
            print(f"  ...{i}/{len(final_games)}")
        # Look up the prior-Monday snapshot for this game's stats
        gdate = g.get("officialDate") or g.get("gameDate", "")[:10]
        snap = lookup_snapshot(snaps, gdate)
        if not snap:
            continue
        # Pre-fetch boxscore for lineup recovery (also reused below for stats).
        pre_box: dict | None = None
        h_lu_ids: list[int] = []; a_lu_ids: list[int] = []
        try:
            pre_box = mlb_api.boxscore(g.get("gamePk"))
            h_lu_ids = lf.extract_starting_lineup(pre_box, "home")
            a_lu_ids = lf.extract_starting_lineup(pre_box, "away")
        except Exception:
            pre_box = None

        snap_bat = {int(k): v for k, v in snap.get("bat", {}).items()}
        try:
            f = feats.build_game_features(
                g, snap["team_off"], snap["team_pit"], snap["pit"],
                team_off_recent=None, team_pit_recent=None,
                home_lineup_ids=h_lu_ids,
                away_lineup_ids=a_lu_ids,
                batter_stats=snap_bat,
                bat_vs_l=bat_vs_l, bat_vs_r=bat_vs_r,
                bat_sides=bat_sides, pit_throws=pit_throws,
            )
        except Exception:
            continue
        if f is None:
            continue
        feat_rows.append(asdict(f))

        # Boxscore extract for player-level backtest
        try:
            box = pre_box if pre_box is not None else mlb_api.boxscore(f.game_pk)
            for side in ("home", "away"):
                pdata = box.get("teams", {}).get(side, {}).get("players", {})
                team_id = f.home_team_id if side == "home" else f.away_team_id
                opp_id = f.away_team_id if side == "home" else f.home_team_id
                for pid_str, pl in pdata.items():
                    person = pl.get("person", {})
                    pos = pl.get("position", {}).get("abbreviation", "")
                    s = pl.get("stats", {})
                    bat = s.get("batting") or {}
                    pit = s.get("pitching") or {}
                    if not bat and not pit:
                        continue
                    if (bat.get("plateAppearances") or 0) == 0 and not pit:
                        continue
                    box_rows.append({
                        "game_pk": f.game_pk, "date": f.date, "venue": f.venue,
                        "side": side, "team_id": team_id, "opp_team_id": opp_id,
                        "player_id": person.get("id"), "name": person.get("fullName", ""),
                        "position": pos,
                        "pa": bat.get("plateAppearances", 0) or 0,
                        "ab": bat.get("atBats", 0) or 0, "h":  bat.get("hits", 0) or 0,
                        "doubles": bat.get("doubles", 0) or 0,
                        "triples": bat.get("triples", 0) or 0,
                        "hr": bat.get("homeRuns", 0) or 0,
                        "rbi": bat.get("rbi", 0) or 0,
                        "runs_b": bat.get("runs", 0) or 0,
                        "bb_b": bat.get("baseOnBalls", 0) or 0,
                        "k_b": bat.get("strikeOuts", 0) or 0,
                        "tb": bat.get("totalBases", 0) or 0,
                        "sb": bat.get("stolenBases", 0) or 0,
                        "ip": feats._ip_to_outs(pit.get("inningsPitched")) / 3.0 if pit else 0.0,
                        "outs": feats._ip_to_outs(pit.get("inningsPitched")) if pit else 0,
                        "h_p": pit.get("hits", 0) or 0,
                        "er":  pit.get("earnedRuns", 0) or 0,
                        "k_p": pit.get("strikeOuts", 0) or 0,
                        "bb_p": pit.get("baseOnBalls", 0) or 0,
                        "hr_p": pit.get("homeRuns", 0) or 0,
                        "bf": pit.get("battersFaced", 0) or 0,
                        "started": (pit.get("gamesStarted", 0) or 0) > 0,
                    })
        except Exception:
            pass

    # Persist
    feat_csv = out_dir / f"games_{SEASON}.csv"
    if feat_rows:
        keys = list(feat_rows[0].keys())
        with feat_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader(); wr.writerows(feat_rows)

    box_csv = out_dir / f"box_{SEASON}.csv"
    if box_rows:
        keys = list(box_rows[0].keys())
        with box_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader(); wr.writerows(box_rows)

    print(f"\nDone.")
    print(f"  feature rows: {len(feat_rows)}")
    print(f"  player-game rows: {len(box_rows)}")
    print(f"  -> {feat_csv}")
    print(f"  -> {box_csv}")


if __name__ == "__main__":
    main()
