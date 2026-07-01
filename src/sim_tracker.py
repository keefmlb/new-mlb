"""Record and grade Simulation Leaderboard picks.

The Simulation Leaderboard (app.py / game_sim.build_sim_leaderboard_by_market)
ranks each market's offered bets by how often they HIT across 10,000 sims per
game. This module persists those picks the day they're built and grades them
against boxscores once the games are final — the sim's counterpart to the
value-bet log (bet_tracker), so we can ask "do the sim's most-confident picks
actually win?" per market over time.

Storage: data/bets/sim_picks.json — one entry per
(date, game_pk, player_id, market, side, line). Re-logging the same slate is
idempotent (latest sim_hit wins; outcome preserved once graded).

Grading reuses bet_tracker._resolve_outcome so sim picks settle identically to
logged bets (W/L/P, whole-number lines push).

Usage:
  from src import sim_tracker
  sim_tracker.log_sim_picks(date, by_market)   # after building the leaderboard
  sim_tracker.evaluate_sim_outcomes()           # in the daily pipeline
  rec = sim_tracker.get_sim_record(days=30)      # for the app
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import bet_tracker

ROOT = Path(__file__).resolve().parent.parent
SIM_LOG_PATH = ROOT / "data" / "bets" / "sim_picks.json"
GAMES_CSV = ROOT / "data" / "games" / "games_2026.csv"
BOX_CSV   = ROOT / "data" / "games" / "box_2026.csv"


def _load() -> list[dict]:
    if not SIM_LOG_PATH.exists():
        return []
    try:
        return json.loads(SIM_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(entries: list[dict]) -> None:
    SIM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIM_LOG_PATH.write_text(json.dumps(entries, indent=2, default=str),
                            encoding="utf-8")


def _key(e: dict) -> tuple:
    return (
        e.get("date"),
        e.get("game_pk"),
        e.get("player_id"),
        e.get("market"),
        e.get("side") or e.get("description"),
        round(float(e.get("line") or 0), 1),
    )


def log_sim_picks(date: str, by_market: dict[str, list[dict]]) -> int:
    """Persist the per-market leaderboard rows for `date`. Returns the number
    of NEW picks added. Existing picks (same key) have their sim_hit refreshed
    but keep any outcome already graded — re-running the leaderboard the same
    day never double-logs or wipes settled results."""
    if not by_market:
        return 0
    entries = _load()
    index = {_key(e): e for e in entries}
    added = 0
    logged_at = datetime.now(timezone.utc).isoformat()
    for market, rows in by_market.items():
        for rank, r in enumerate(rows, start=1):
            e = {
                "date": date,
                "logged_at": logged_at,
                "market": market,
                "rank": rank,
                "matchup": r.get("matchup", ""),
                "description": r.get("description", ""),
                "line": r.get("line"),
                "side": r.get("side", ""),
                "odds": r.get("odds", 0),
                "game_pk": r.get("game_pk"),
                "player_id": r.get("player_id"),
                "sim_hit": r.get("sim_hit"),
                "sim_hits_n": r.get("sim_hits_n"),
                "sim_n": r.get("n"),
                "novig_prob": r.get("novig_prob"),
                "outcome": None,
                "actual": None,
            }
            k = _key(e)
            if k in index:
                prev = index[k]
                prev["sim_hit"] = e["sim_hit"]
                prev["sim_hits_n"] = e["sim_hits_n"]
                prev["rank"] = e["rank"]
            else:
                entries.append(e)
                index[k] = e
                added += 1
    _save(entries)
    return added


def evaluate_sim_outcomes() -> int:
    """Grade any logged sim picks whose games are now final. Reuses
    bet_tracker._resolve_outcome. Returns the count updated."""
    entries = _load()
    if not entries:
        return 0
    try:
        import pandas as pd
        games = pd.read_csv(GAMES_CSV) if GAMES_CSV.exists() else None
        box   = pd.read_csv(BOX_CSV)   if BOX_CSV.exists()   else None
    except Exception:
        return 0

    today = datetime.now(timezone.utc).date()
    updated = 0
    for e in entries:
        if e.get("outcome") is not None:
            continue
        if e["date"] >= today.isoformat():
            continue
        result = bet_tracker._resolve_outcome(e, games, box)
        if result is not None:
            e["outcome"] = result["outcome"]
            e["actual"]  = result["actual"]
            updated += 1
    if updated:
        _save(entries)
    return updated


def get_sim_record(days: int = 30) -> dict:
    """Summary of graded sim picks over the last `days`, broken out by market.
    Mirrors bet_tracker.get_track_record shape (total/wins/losses/.../by_market)
    plus a `calibration` block: for each market, mean simulated hit rate vs the
    realized win rate — the sim is well-calibrated when those track."""
    entries = _load()
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    recent = [e for e in entries if e.get("date", "") >= cutoff]

    def _blank() -> dict:
        return {"total": 0, "wins": 0, "losses": 0, "pushes": 0, "pending": 0,
                "sim_hit_sum": 0.0, "sim_hit_n": 0}

    by_market: dict[str, dict] = {}
    for e in recent:
        bm = by_market.setdefault(e.get("market", "unknown"), _blank())
        bm["total"] += 1
        o = e.get("outcome")
        if o == "W":
            bm["wins"] += 1
        elif o == "L":
            bm["losses"] += 1
        elif o == "P":
            bm["pushes"] += 1
        else:
            bm["pending"] += 1
        sh = e.get("sim_hit")
        if sh is not None:
            bm["sim_hit_sum"] += float(sh)
            bm["sim_hit_n"] += 1

    calibration = {}
    for m, bm in by_market.items():
        decided = bm["wins"] + bm["losses"]
        calibration[m] = {
            "mean_sim_hit": (bm["sim_hit_sum"] / bm["sim_hit_n"]
                             if bm["sim_hit_n"] else None),
            "win_rate": (bm["wins"] / decided if decided else None),
            "decided": decided,
        }

    wins   = sum(b["wins"] for b in by_market.values())
    losses = sum(b["losses"] for b in by_market.values())
    pushes = sum(b["pushes"] for b in by_market.values())
    pending = sum(b["pending"] for b in by_market.values())
    decided = wins + losses
    return {
        "total":    len(recent),
        "wins":     wins,
        "losses":   losses,
        "pushes":   pushes,
        "pending":  pending,
        "win_rate": wins / decided if decided else None,
        "by_market": by_market,
        "calibration": calibration,
        "entries":  sorted(recent, key=lambda x: (x["date"], x.get("market", "")),
                           reverse=True),
    }


if __name__ == "__main__":
    n = evaluate_sim_outcomes()
    print(f"Evaluated sim outcomes: {n} picks updated.")
