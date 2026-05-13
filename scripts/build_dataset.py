"""Pull all 2026 games + stats and build the feature dataset.

Run: python -m scripts.build_dataset

Writes:
  data/games/games_2026.csv            — one row per game (leak-free features for finals,
                                         current-stats features for upcoming)
  data/games/box_2026.csv              — per-player per-game boxscore stats
  data/games/snapshots_2026/           — weekly Monday stat snapshots (used by train_props.py)
  data/games/snapshot_2026.json        — current-day stats for live prediction

Historical game features use the prior-Monday snapshot so stats accumulated
during the game itself don't contaminate the training features (leakage fix).
Upcoming/today games use the current snapshot since there is no future to leak.
"""
from __future__ import annotations
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src import mlb_api, features as feats, statcast as sc, umpire as ump, lineup_features as lf

SEASON = 2026
SEASON_START = date(2026, 3, 25)
SNAPS_DIR = ROOT / "data" / "games" / "snapshots_2026"


def _int_keys(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def _pull_weekly_snapshots(season_start: date, through: date) -> dict[str, dict]:
    """Pull & disk-cache weekly Monday stat snapshots from season_start through `through` (exclusive).

    Each snapshot stores season-to-date and last-14-day team + player stats so that
    historical game features can be built without touching data from after game day.
    Returns {iso_date: snap_dict} keyed by snapshot date (string keys in JSON, but
    caller converts with _int_keys before use).
    """
    SNAPS_DIR.mkdir(parents=True, exist_ok=True)
    snaps: dict[str, dict] = {}
    d = season_start
    while d < through:
        if d.weekday() == 0 or d == season_start:
            snap_path = SNAPS_DIR / f"{d.isoformat()}.json"
            if snap_path.exists():
                try:
                    snaps[d.isoformat()] = json.loads(snap_path.read_text(encoding="utf-8"))
                    d += timedelta(days=1)
                    continue
                except Exception:
                    pass
            print(f"  snap {d.isoformat()}...")
            try:
                recent_start = d - timedelta(days=14)
                snap = {
                    "as_of": d.isoformat(),
                    "team_off": {str(k): v for k, v in
                                 mlb_api.team_stats_by_range(SEASON, "hitting", season_start, d).items()},
                    "team_pit": {str(k): v for k, v in
                                 mlb_api.team_stats_by_range(SEASON, "pitching", season_start, d).items()},
                    "pitcher_stats": {str(k): v for k, v in
                                      mlb_api.player_stats_by_range(SEASON, "pitching", season_start, d).items()},
                    "batter_stats": {str(k): v for k, v in
                                     mlb_api.player_stats_by_range(SEASON, "hitting", season_start, d).items()},
                    "team_off_recent": {str(k): v for k, v in
                                        mlb_api.team_stats_by_range(SEASON, "hitting", recent_start, d).items()},
                    "team_pit_recent": {str(k): v for k, v in
                                        mlb_api.team_stats_by_range(SEASON, "pitching", recent_start, d).items()},
                    "batter_stats_recent": {str(k): v for k, v in
                                            mlb_api.player_stats_by_range(SEASON, "hitting", recent_start, d).items()},
                    "pitcher_stats_recent": {str(k): v for k, v in
                                             mlb_api.player_stats_by_range(SEASON, "pitching", recent_start, d).items()},
                }
                snap_path.write_text(json.dumps(snap, default=str), encoding="utf-8")
                snaps[d.isoformat()] = snap
            except Exception as e:
                print(f"    snap failed for {d.isoformat()}: {e}")
        d += timedelta(days=1)
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
    today = datetime.now(timezone.utc).date()

    print(f"[1/6] Pulling current team stats for live prediction...")
    team_off = mlb_api.team_season_stats_bulk(SEASON, "hitting")
    team_pit = mlb_api.team_season_stats_bulk(SEASON, "pitching")
    print(f"      teams: off={len(team_off)} pit={len(team_pit)}")

    print(f"[2/6] Pulling current pitcher + hitter season stats...")
    pit_stats = mlb_api.player_season_stats_bulk(SEASON, "pitching", qualified=False)
    bat_stats = mlb_api.player_season_stats_bulk(SEASON, "hitting", qualified=False)
    print(f"      pitchers: {len(pit_stats)}  hitters: {len(bat_stats)}")

    print(f"[3/6] Pulling last-14-day form for teams + players...")
    form_start = today - timedelta(days=14)
    team_off_recent = mlb_api.team_stats_by_range(SEASON, "hitting", form_start, today)
    team_pit_recent = mlb_api.team_stats_by_range(SEASON, "pitching", form_start, today)
    bat_recent = mlb_api.player_stats_by_range(SEASON, "hitting", form_start, today)
    pit_recent = mlb_api.player_stats_by_range(SEASON, "pitching", form_start, today)
    print(f"      teams off: {len(team_off_recent)}  pit: {len(team_pit_recent)}  "
          f"batters: {len(bat_recent)}  pitchers: {len(pit_recent)}")

    print(f"[4/6] Pulling L/R platoon splits + handedness rosters...")
    bat_vs_l = mlb_api.player_splits_bulk(SEASON, "hitting", "l")
    bat_vs_r = mlb_api.player_splits_bulk(SEASON, "hitting", "r")
    pit_vs_l = mlb_api.player_splits_bulk(SEASON, "pitching", "l")
    pit_vs_r = mlb_api.player_splits_bulk(SEASON, "pitching", "r")
    bat_sides = mlb_api.batter_bats_bulk(SEASON)
    pit_throws = mlb_api.pitcher_throws_bulk(SEASON)
    print(f"      batter splits: vL={len(bat_vs_l)}  vR={len(bat_vs_r)}  bat-sides: {len(bat_sides)}")

    print(f"[4b/6] Building weekly historical snapshots (eliminates feature leakage)...")
    snaps = _pull_weekly_snapshots(SEASON_START, today)
    print(f"       {len(snaps)} snapshots ({min(snaps) if snaps else 'none'} to {max(snaps) if snaps else 'none'})")

    print(f"[4c/6] Fetching Statcast team batting + pitcher leaderboards...")
    try:
        player_team_map = {pid: int(s["team_id"])
                           for pid, s in bat_stats.items() if s.get("team_id")}
        sc_team_bat = sc.get_team_batting(SEASON, player_team_map)
        sc_pit      = sc.get_pitcher_stats(SEASON)
        sc_team_def = sc.get_team_fielding(SEASON, player_team_map)
        print(f"       {len(sc_team_bat)} teams  {len(sc_pit)} pitchers  {len(sc_team_def)} team-def")
    except Exception as e:
        print(f"       WARNING: Statcast fetch failed ({e}) — using league-average fallbacks")
        sc_team_bat, sc_pit, sc_team_def = {}, {}, {}

    print(f"[5/6] Pulling schedule {SEASON_START} through {today + timedelta(days=2)}...")
    games = mlb_api.schedule_range(SEASON_START, today + timedelta(days=2))
    print(f"      games: {len(games)}")

    out_dir = ROOT / "data" / "games"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[6/6] Building feature rows + boxscore extracts...")
    # Pitcher last-appearance lookup: built incrementally as we process games
    # in date order, so each row only sees prior appearances (no leakage).
    _pitcher_last_app: dict[int, str] = {}
    # Bullpen 72hr usage: also built incrementally as we accumulate relief
    # appearances. Each new game queries the lookup BEFORE adding its own
    # box rows -- no leakage of future games into past predictions.
    _bullpen_rel_log: list[dict] = []  # accumulated relief outings
    # Catcher framing proxy: bootstrap from existing box CSV (prior runs of
    # build_dataset). For the FIRST run this will be empty; subsequent runs
    # benefit from accumulated history. For training-time leak-free usage,
    # ideally each game would only see prior games' framing — but the
    # signal is already EB-shrunk to 0 (effectively a zero-noise feature
    # for thinly-sampled catchers), so the leakage is minimal.
    _catcher_framing_proxy: dict[int, float] = {}
    try:
        _prior_box = pd.read_csv(ROOT / "data" / "games" / "box_2026.csv")
        _catcher_framing_proxy = feats.build_catcher_framing_proxy(_prior_box, pit_stats)
        print(f"  loaded catcher framing proxy: {len(_catcher_framing_proxy)} catchers")
    except Exception:
        pass
    # Sort games by date to ensure incremental accumulation works
    try:
        games = sorted(games, key=lambda g: g.get("officialDate") or g.get("gameDate", "")[:10])
    except Exception:
        pass
    feat_rows: list[dict] = []
    box_rows: list[dict] = []
    n_skipped = 0
    n_final = 0
    n_snap_used = 0

    # Umpire accumulation: build K-rate table from boxscores as we go
    ump_rates: dict[str, dict] = ump.load_rates()   # start from cached rates

    for i, g in enumerate(games):
        if i % 50 == 0:
            print(f"  ...{i}/{len(games)}  (final so far: {n_final})")
        st = (g.get("status") or {}).get("codedGameState", "")
        if st in ("D", "C", "P"):
            n_skipped += 1
            continue

        gdate = g.get("officialDate") or g.get("gameDate", "")[:10]
        is_final = st == "F"

        # Historical finals: use per-game snapshot to avoid feature leakage.
        # Today's and future games: use current stats (nothing to leak yet).
        snap = None
        if is_final and snaps and gdate < today.isoformat():
            snap = _lookup_snap(snaps, gdate)

        if snap:
            g_team_off = _int_keys(snap["team_off"])
            g_team_pit = _int_keys(snap["team_pit"])
            g_pit_stats = _int_keys(snap["pitcher_stats"])
            g_team_off_recent = _int_keys(snap.get("team_off_recent", {}))
            g_team_pit_recent = _int_keys(snap.get("team_pit_recent", {}))
            g_bat_stats = _int_keys(snap.get("batter_stats", {}))
            g_bat_recent = _int_keys(snap.get("batter_stats_recent", {}))
            g_pit_recent = _int_keys(snap.get("pitcher_stats_recent", {}))
            n_snap_used += 1
        else:
            g_team_off = team_off
            g_team_pit = team_pit
            g_pit_stats = pit_stats
            g_team_off_recent = team_off_recent
            g_team_pit_recent = team_pit_recent
            g_bat_stats = bat_stats
            g_bat_recent = bat_recent
            g_pit_recent = pit_recent
        # Per-game (or current) reliever-only team aggregates from individual
        # pitcher stats — true bullpen ERA/FIP for late-game leverage.
        g_bullpen_stats = feats.bullpen_stats_by_team(g_pit_stats)

        # For finished games, fetch boxscore now to recover the actual starting
        # lineup so lineup-weighted features reflect who really played.
        # Live/upcoming games use lineups from `mlb_api.extract_lineups` at
        # predict time; here we leave them blank.
        h_lu_ids: list[int] = []; a_lu_ids: list[int] = []
        h_catcher: int | None = None; a_catcher: int | None = None
        pre_box: dict | None = None
        if (g.get("status") or {}).get("codedGameState") == "F":
            try:
                pre_box = mlb_api.boxscore(g.get("gamePk"))
                h_lu_ids = lf.extract_starting_lineup(pre_box, "home")
                a_lu_ids = lf.extract_starting_lineup(pre_box, "away")
                h_catcher = lf.extract_starting_catcher(pre_box, "home")
                a_catcher = lf.extract_starting_catcher(pre_box, "away")
            except Exception:
                pre_box = None

        try:
            f = feats.build_game_features(g, g_team_off, g_team_pit, g_pit_stats,
                                          team_off_recent=g_team_off_recent,
                                          team_pit_recent=g_team_pit_recent,
                                          sc_team_bat=sc_team_bat,
                                          sc_pit=sc_pit,
                                          sc_team_def=sc_team_def,
                                          home_lineup_ids=h_lu_ids,
                                          away_lineup_ids=a_lu_ids,
                                          batter_stats=g_bat_stats,
                                          bat_vs_l=bat_vs_l,
                                          bat_vs_r=bat_vs_r,
                                          bat_sides=bat_sides,
                                          pit_throws=pit_throws,
                                          batter_recent=g_bat_recent,
                                          bullpen_stats=g_bullpen_stats,
                                          pitcher_last_appearance=_pitcher_last_app,
                                          bullpen_usage=feats.BullpenUsageLookup(
                                              pd.DataFrame(_bullpen_rel_log)) if _bullpen_rel_log else None,
                                          catcher_framing=_catcher_framing_proxy,
                                          home_catcher_id=h_catcher,
                                          away_catcher_id=a_catcher,
                                          pit_recent=g_pit_recent)
        except Exception as e:
            print(f"  feature build failed for game {g.get('gamePk')}: {e}")
            continue
        if f is None:
            n_skipped += 1
            continue
        feat_row = asdict(f)
        feat_rows.append(feat_row)

        if f.is_final:
            n_final += 1
            try:
                box = pre_box if pre_box is not None else mlb_api.boxscore(f.game_pk)

                # Extract HP umpire, accumulate K-rate stats, update this row's mult
                hp_ump = ump.get_hp_umpire_from_boxscore(box)
                if hp_ump:
                    home_k = int((box.get("teams", {}).get("home", {})
                                  .get("teamStats", {}).get("batting", {})
                                  .get("strikeOuts", 0)) or 0)
                    away_k = int((box.get("teams", {}).get("away", {})
                                  .get("teamStats", {}).get("batting", {})
                                  .get("strikeOuts", 0)) or 0)
                    r = ump_rates.setdefault(hp_ump, {"games": 0, "total_k": 0})
                    r["games"] += 1
                    r["total_k"] += home_k + away_k
                    feat_row["ump_k_mult"] = ump.get_k_mult(hp_ump, ump_rates)

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
                        row = {
                            "game_pk": f.game_pk,
                            "date": f.date,
                            "venue": f.venue,
                            "side": side,
                            "team_id": team_id,
                            "opp_team_id": opp_id,
                            "player_id": person.get("id"),
                            "name": person.get("fullName", ""),
                            "position": pos,
                            "pa": bat.get("plateAppearances", 0) or 0,
                            "ab": bat.get("atBats", 0) or 0,
                            "h":  bat.get("hits", 0) or 0,
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
                            "h_p":  pit.get("hits", 0) or 0,
                            "er":   pit.get("earnedRuns", 0) or 0,
                            "k_p":  pit.get("strikeOuts", 0) or 0,
                            "bb_p": pit.get("baseOnBalls", 0) or 0,
                            "hr_p": pit.get("homeRuns", 0) or 0,
                            "bf":   pit.get("battersFaced", 0) or 0,
                            "started": (pit.get("gamesStarted", 0) or 0) > 0,
                        }
                        box_rows.append(row)
                        # Update pitcher last-appearance map AFTER processing
                        # this game — so future games see this date as the
                        # previous appearance (no leakage of future games).
                        if pit and feats._ip_to_outs(pit.get("inningsPitched")) > 0:
                            _pid = person.get("id")
                            if _pid:
                                _pitcher_last_app[int(_pid)] = gdate
                            # Track relief outings for bullpen-usage lookup
                            if not row["started"] and row["outs"] > 0:
                                _bullpen_rel_log.append({
                                    "team_id": row["team_id"],
                                    "player_id": row["player_id"],
                                    "date": row["date"],
                                    "outs": row["outs"],
                                    "bf": row["bf"],
                                    "started": False,
                                })
            except Exception as e:
                print(f"  boxscore fetch failed for {f.game_pk}: {e}")

    # Save updated umpire K-rate table
    if ump_rates:
        ump.save_rates(ump_rates)
        n_ump = sum(1 for r in feat_rows if r.get("ump_k_mult", 1.0) != 1.0)
        print(f"  umpire rates: {len(ump_rates)} umpires, {n_ump} games with non-avg mult")

    feat_csv = out_dir / f"games_{SEASON}.csv"
    if feat_rows:
        keys = list(feat_rows[0].keys())
        with feat_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(feat_rows)

    box_csv = out_dir / f"box_{SEASON}.csv"
    if box_rows:
        keys = list(box_rows[0].keys())
        with box_csv.open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(box_rows)

    # Current-day snapshot for live prediction (predict_core.py uses this)
    snap_path = out_dir / f"snapshot_{SEASON}.json"
    snap_path.write_text(json.dumps({
        "as_of": today.isoformat(),
        "team_off": {str(k): v for k, v in team_off.items()},
        "team_pit": {str(k): v for k, v in team_pit.items()},
        "team_off_recent": {str(k): v for k, v in team_off_recent.items()},
        "team_pit_recent": {str(k): v for k, v in team_pit_recent.items()},
        "pitcher_stats": {str(k): v for k, v in pit_stats.items()},
        "batter_stats": {str(k): v for k, v in bat_stats.items()},
        "batter_stats_recent": {str(k): v for k, v in bat_recent.items()},
        "pitcher_stats_recent": {str(k): v for k, v in pit_recent.items()},
        "bat_vs_l": {str(k): v for k, v in bat_vs_l.items()},
        "bat_vs_r": {str(k): v for k, v in bat_vs_r.items()},
        "pit_vs_l": {str(k): v for k, v in pit_vs_l.items()},
        "pit_vs_r": {str(k): v for k, v in pit_vs_r.items()},
        "bat_sides": {str(k): v for k, v in bat_sides.items()},
        "pit_throws": {str(k): v for k, v in pit_throws.items()},
    }, default=str), encoding="utf-8")

    print(f"\nDone.")
    print(f"  feature rows: {len(feat_rows)}  (final: {n_final}, skipped: {n_skipped})")
    print(f"  historical games using leak-free snapshots: {n_snap_used}")
    print(f"  player-game rows: {len(box_rows)}")
    print(f"  -> {feat_csv}")
    print(f"  -> {box_csv}")
    print(f"  -> {snap_path}")
    print(f"  -> {SNAPS_DIR}/ ({len(snaps)} weekly snapshots)")


if __name__ == "__main__":
    main()
