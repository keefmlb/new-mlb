"""Thin client over MLB Stats API (statsapi.mlb.com).

No API key required. We cache responses on disk because we re-pull the same
historical games during backtests.
"""
from __future__ import annotations
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import requests

BASE = "https://statsapi.mlb.com/api"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "-")
    return CACHE_DIR / f"{safe}.json"


def _get(path: str, params: dict | None = None, ttl_seconds: int = 86400, force: bool = False) -> dict:
    """GET with disk cache. ttl_seconds = how stale before re-fetching."""
    qs = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    key = f"{path}?{qs}"
    cp = _cache_path(key)
    if not force and cp.exists():
        age = time.time() - cp.stat().st_mtime
        if age < ttl_seconds:
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    url = f"{BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cp.write_text(json.dumps(data), encoding="utf-8")
    return data


def schedule(d: date | str, hydrate: str = "probablePitcher,lineups,team,venue,weather") -> list[dict]:
    """Return the list of games on a given date with probable pitchers and venue info."""
    if isinstance(d, date):
        d = d.isoformat()
    # 15-min TTL so confirmed lineups (posted 1-3hrs pre-game) aren't stale long
    data = _get("/v1/schedule", {"sportId": 1, "date": d, "hydrate": hydrate}, ttl_seconds=900)
    games: list[dict] = []
    for entry in data.get("dates", []):
        games.extend(entry.get("games", []))
    return games


def extract_lineups(game: dict) -> dict[str, list[int]]:
    """Extract confirmed lineup player IDs from a hydrated schedule game dict.

    Returns {"home": [pid, ...], "away": [pid, ...]} sorted by batting order.
    Empty list means lineup not yet posted — callers should fall back to PA leaderboard.
    """
    lineups = game.get("lineups") or {}
    result: dict[str, list[int]] = {"home": [], "away": []}
    for side, key in [("home", "homePlayers"), ("away", "awayPlayers")]:
        players = lineups.get(key) or []
        # batOrder comes as "100","200",...,"900" (1st through 9th)
        def _bat_order(p: dict) -> int:
            v = p.get("batOrder", 9999)
            try:
                return int(v)
            except (TypeError, ValueError):
                return 9999
        sorted_players = sorted(players, key=_bat_order)
        result[side] = [p["id"] for p in sorted_players if p.get("id")]
    return result


def extract_starting_catchers(game: dict) -> dict[str, int | None]:
    """Pull the starting catcher's player_id from a hydrated game dict.

    Returns {"home": pid|None, "away": pid|None}. Uses primaryPosition.code
    or primaryPosition.abbreviation = 'C' if present.
    """
    lineups = game.get("lineups") or {}
    result: dict[str, int | None] = {"home": None, "away": None}
    for side, key in [("home", "homePlayers"), ("away", "awayPlayers")]:
        for p in lineups.get(key) or []:
            pos = p.get("primaryPosition") or {}
            if pos.get("abbreviation") == "C" or pos.get("code") == "2":
                result[side] = p.get("id")
                break
    return result


def schedule_range(start: date, end: date) -> list[dict]:
    """All games between [start, end] inclusive — single API call when possible."""
    s, e = start.isoformat(), end.isoformat()
    data = _get(
        "/v1/schedule",
        {"sportId": 1, "startDate": s, "endDate": e, "hydrate": "probablePitcher,team,venue"},
        ttl_seconds=3600,
    )
    games: list[dict] = []
    for entry in data.get("dates", []):
        games.extend(entry.get("games", []))
    return games


def game_feed(game_pk: int) -> dict:
    """Live feed for a game — final box score for completed games."""
    return _get(f"/v1.1/game/{game_pk}/feed/live", ttl_seconds=86400 * 30)


def boxscore(game_pk: int) -> dict:
    return _get(f"/v1/game/{game_pk}/boxscore", ttl_seconds=86400 * 30)


def player_season_stats(player_id: int, season: int, group: str) -> dict:
    """group ∈ {'hitting','pitching'}. Returns standard + sabermetric stats."""
    out: dict[str, Any] = {}
    for kind in ("season", "sabermetrics"):
        d = _get(
            f"/v1/people/{player_id}/stats",
            {"stats": kind, "group": group, "season": season},
            ttl_seconds=3600,
        )
        for s in d.get("stats", []):
            for split in s.get("splits", []):
                out.update(split.get("stat", {}))
    return out


def player_season_stats_bulk(season: int, group: str, qualified: bool = False) -> dict[int, dict]:
    """One bulk call returning {player_id: stat_dict} merging standard + sabermetric stats."""
    pool = "qualified" if qualified else "all"
    out: dict[int, dict] = {}
    for kind in ("season", "sabermetrics"):
        d = _get(
            "/v1/stats",
            {"stats": kind, "group": group, "season": season, "sportId": 1, "playerPool": pool, "limit": 2000},
            ttl_seconds=3600,
        )
        for s in d.get("stats", []):
            for split in s.get("splits", []):
                pid = split.get("player", {}).get("id")
                if pid is None:
                    continue
                row = out.setdefault(pid, {"player_id": pid, "name": split["player"]["fullName"]})
                row.update(split.get("stat", {}))
                team = split.get("team", {})
                if team:
                    row["team_id"] = team.get("id")
    return out


def player_splits_bulk(season: int, group: str, vs_hand: str) -> dict[int, dict]:
    """Pull platoon (vs L/R) splits in one call.

    `vs_hand` ∈ {'l', 'r'}.
      - For hitters: vs LHP / vs RHP
      - For pitchers: vs LHB / vs RHB
    Returns {player_id: stat_dict_for_this_split}. Players without a meaningful
    split sample (no PA/BF in this matchup) just won't appear.
    """
    sit = "vl" if vs_hand.lower() == "l" else "vr"
    out: dict[int, dict] = {}
    d = _get(
        "/v1/stats",
        {"stats": "statSplits", "sitCodes": sit, "group": group,
         "season": season, "sportId": 1, "playerPool": "all", "limit": 2000},
        ttl_seconds=3600,
    )
    for s in d.get("stats", []):
        for split in s.get("splits", []):
            pid = split.get("player", {}).get("id")
            if pid is None:
                continue
            row = out.setdefault(pid, {"player_id": pid,
                                       "name": split.get("player", {}).get("fullName", ""),
                                       "vs": vs_hand.upper()})
            row.update(split.get("stat", {}))
            team = split.get("team", {})
            if team:
                row["team_id"] = team.get("id")
    return out


def pitcher_throws_bulk(season: int) -> dict[int, str]:
    """Get every pitcher's throwing hand ('L' / 'R' / 'S') from rosters.

    Pulls each team's roster and reads `pitchHand.code`. Returns {pid: code}.
    """
    out: dict[int, str] = {}
    teams_d = _get("/v1/teams", {"sportId": 1, "season": season}, ttl_seconds=86400)
    for t in teams_d.get("teams", []):
        tid = t.get("id")
        if not tid:
            continue
        try:
            r = _get(f"/v1/teams/{tid}/roster",
                     {"rosterType": "fullRoster", "season": season},
                     ttl_seconds=86400)
        except Exception:
            continue
        for entry in r.get("roster", []):
            person = entry.get("person", {})
            pid = person.get("id")
            if not pid:
                continue
            # We need to peek into person.pitchHand which lives on the
            # player record, not the roster entry. Pull it lazily.
    # Bulk pitcher hand from /v1/sports/1/players is more efficient.
    try:
        d = _get("/v1/sports/1/players", {"season": season}, ttl_seconds=86400)
        for p in d.get("people", []):
            ph = p.get("pitchHand", {}).get("code")
            if ph:
                out[p["id"]] = ph
    except Exception:
        pass
    return out


def batter_bats_bulk(season: int) -> dict[int, str]:
    """Get every player's batting side ('L' / 'R' / 'S' for switch)."""
    out: dict[int, str] = {}
    try:
        d = _get("/v1/sports/1/players", {"season": season}, ttl_seconds=86400)
        for p in d.get("people", []):
            bs = p.get("batSide", {}).get("code")
            if bs:
                out[p["id"]] = bs
    except Exception:
        pass
    return out


def team_season_stats_bulk(season: int, group: str) -> dict[int, dict]:
    """{team_id: stat_dict} for hitting or pitching."""
    out: dict[int, dict] = {}
    d = _get(
        "/v1/teams/stats",
        {"stats": "season", "group": group, "season": season, "sportId": 1},
        ttl_seconds=3600,
    )
    for s in d.get("stats", []):
        for split in s.get("splits", []):
            tid = split.get("team", {}).get("id")
            if tid is None:
                continue
            row = out.setdefault(tid, {"team_id": tid, "name": split["team"]["name"]})
            row.update(split.get("stat", {}))
    return out


def teams(season: int) -> list[dict]:
    d = _get("/v1/teams", {"sportId": 1, "season": season}, ttl_seconds=86400)
    return d.get("teams", [])


def team_stats_by_range(season: int, group: str, start: date | str, end: date | str) -> dict[int, dict]:
    """Team stats restricted to a date window — used for 'recent form' / hot streak features."""
    s = start.isoformat() if isinstance(start, date) else start
    e = end.isoformat() if isinstance(end, date) else end
    out: dict[int, dict] = {}
    d = _get(
        "/v1/teams/stats",
        {"stats": "byDateRange", "group": group, "season": season, "sportId": 1,
         "startDate": s, "endDate": e},
        ttl_seconds=3600,
    )
    for stat_block in d.get("stats", []):
        for split in stat_block.get("splits", []):
            tid = split.get("team", {}).get("id")
            if tid is None:
                continue
            row = out.setdefault(tid, {"team_id": tid, "name": split["team"]["name"]})
            row.update(split.get("stat", {}))
    return out


def player_stats_by_range(season: int, group: str, start: date | str, end: date | str,
                          qualified: bool = False) -> dict[int, dict]:
    """Per-player stats restricted to a date window (for last-N-day rolling form)."""
    s = start.isoformat() if isinstance(start, date) else start
    e = end.isoformat() if isinstance(end, date) else end
    pool = "qualified" if qualified else "all"
    out: dict[int, dict] = {}
    d = _get(
        "/v1/stats",
        {"stats": "byDateRange", "group": group, "season": season, "sportId": 1,
         "playerPool": pool, "startDate": s, "endDate": e, "limit": 2000},
        ttl_seconds=3600,
    )
    for stat_block in d.get("stats", []):
        for split in stat_block.get("splits", []):
            pid = split.get("player", {}).get("id")
            if pid is None:
                continue
            row = out.setdefault(pid, {"player_id": pid, "name": split["player"]["fullName"]})
            row.update(split.get("stat", {}))
            tm = split.get("team", {})
            if tm:
                row["team_id"] = tm.get("id")
    return out


def parse_game_time(game: dict) -> datetime:
    """Returns the scheduled first-pitch time as a UTC datetime."""
    return datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00"))
