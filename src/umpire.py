"""Umpire K-rate lookup.

Home plate umpires with wider/tighter zones shift game K totals by ~5%.
We compute a per-umpire K-rate multiplier (relative to league average)
using Empirical Bayes shrinkage and cache it at data/cache/umpire_rates.json.

Workflow:
  scripts/build_dataset.py extracts HP umpire from each cached boxscore and
  writes umpire_rates.json. predict_core.py reads the rates and applies
  get_k_mult() per game.

League average: ~8.7 total K per team per game (both teams ~17/game combined).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
RATES_PATH = ROOT / "data" / "cache" / "umpire_rates.json"

# BOTH-teams strikeouts per game (league avg) — build_dataset accumulates
# home_k + away_k per umpire, so the league constant must be on the same
# scale. (Bug found Jun 10 2026: this was 8.7 — the PER-TEAM rate — which
# centered the "multiplier" at ~1.84 instead of 1.0, made the EB shrink
# target of 1.0 mean "umpire who halves strikeouts", and made the live
# fallback of 1.0 a 3.5-sigma outlier vs the trained feature distribution.)
LG_K_PER_GAME = 17.4
PRIOR_GAMES   = 30    # EB prior: at 30 games weight = 0.5


def get_hp_umpire_from_boxscore(boxscore: dict) -> str | None:
    """Extract HP umpire name from a boxscore API response."""
    for o in (boxscore.get("officials") or []):
        if o.get("officialType") == "Home Plate":
            return o.get("official", {}).get("fullName")
    return None


def get_hp_umpire_from_game_feed(feed: dict) -> str | None:
    """Extract HP umpire from a game feed (/v1.1/game/{pk}/feed/live) response."""
    officials = (
        feed.get("liveData", {})
            .get("boxscore", {})
            .get("officials") or []
    )
    for o in officials:
        if o.get("officialType") == "Home Plate":
            return o.get("official", {}).get("fullName")
    return None


def load_rates() -> dict[str, dict]:
    """Load umpire K-rate records from cache."""
    if not RATES_PATH.exists():
        return {}
    try:
        return json.loads(RATES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_rates(rates: dict[str, dict]) -> None:
    RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Recompute k_per_game before saving
    for r in rates.values():
        g = r.get("games", 0)
        r["k_per_game"] = r["total_k"] / g if g else LG_K_PER_GAME
    RATES_PATH.write_text(json.dumps(rates, indent=2), encoding="utf-8")


def get_k_mult(umpire_name: str | None, rates: dict[str, dict] | None = None) -> float:
    """Return EB-shrunk K-rate multiplier relative to league average.

    1.0 = league average; 1.05 = 5% more Ks than average.
    Falls back to 1.0 for unknown or None umpires.
    """
    if rates is None:
        rates = load_rates()
    if not umpire_name:
        return 1.0
    r = rates.get(umpire_name)
    if not r:
        return 1.0
    games = r.get("games", 0)
    if games == 0:
        return 1.0
    k_per_game = r.get("k_per_game") or (r.get("total_k", 0) / games if games else LG_K_PER_GAME)
    raw_mult = k_per_game / LG_K_PER_GAME
    w = games / (games + PRIOR_GAMES)
    return round(w * raw_mult + (1 - w) * 1.0, 4)
