"""Baseball Savant (Statcast) data fetcher with disk cache.

Pulls season batting and pitching leaderboards from Baseball Savant's free
CSV endpoint. No API key required. Cached in data/cache/:
  - Current season: 6-hour TTL (Savant updates daily)
  - Prior seasons:  permanent (historical data never changes)

The Savant custom leaderboard returns only the columns you request plus
player_id, year, and the player name field. It does NOT include team_id,
so callers must supply a player→team mapping (from the MLB Stats API)
when aggregating to team level. See get_team_batting().

Typical usage:
    from src import statcast
    # In build_dataset.py / predict_core.py, after fetching bat_stats:
    player_team_map = {pid: int(s["team_id"])
                       for pid, s in bat_stats.items() if s.get("team_id")}
    sc_bat = statcast.get_team_batting(2026, player_team_map)
    sc_pit = statcast.get_pitcher_stats(2026)
"""
from __future__ import annotations
import io
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

_ROOT  = Path(__file__).resolve().parent.parent
_CACHE = _ROOT / "data" / "cache"

_SAVANT = "https://baseballsavant.mlb.com/leaderboard/custom"

# League-average priors used for EB shrinkage
LG_XWOBA      = 0.315   # by definition (xwOBA is scaled to match wOBA)
LG_BARREL_PCT = 7.8     # percent (Savant reports 0-100)
LG_HARD_HIT   = 37.0    # percent (exit velo >= 95 mph)
LG_XERA       = 4.20    # expected ERA based on contact quality

# EB prior weight — at this many PA/BF the data weight = 0.5
_PRIOR_PA_TEAM = 500    # ≈ 25 games × 20 PA/game
_PRIOR_BF_PIT  = 150    # ≈ 50 IP

# Cache version suffix — bump when selections change so stale files are ignored
_BAT_VER = "v3"   # bumped when xba added
_PIT_VER = "v4"   # v4: added called_strike_percent (csp) to pitcher dict


# ---------- Internals ----------

def _ttl_hours(year: int) -> int:
    return 6 if year >= date.today().year else 8760


def _is_stale(path: Path, ttl_h: int) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > ttl_h * 3600


def _safe(x, default=None):
    if x is None:
        return default
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _fetch_leaderboard(year: int, player_type: str,
                       selections: str, min_pa: int,
                       cache_key: str) -> pd.DataFrame:
    cache = _CACHE / f"statcast_{cache_key}_{year}.json"
    ttl   = _ttl_hours(year)

    if not _is_stale(cache, ttl):
        return pd.DataFrame(json.loads(cache.read_text(encoding="utf-8")))

    params = urllib.parse.urlencode({
        "year": year, "type": player_type,
        "filter": "", "sort": "3", "sortDir": "asc",
        "min": min_pa, "selections": selections,
        "chart": "false", "csv": "true",
    })
    url = f"{_SAVANT}?{params}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; mlb-predictor/1.0)",
            "Accept": "text/csv,*/*",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")))
    except Exception as exc:
        if cache.exists():
            print(f"[statcast] fetch failed ({exc}); using cached data")
            return pd.DataFrame(json.loads(cache.read_text(encoding="utf-8")))
        raise RuntimeError(f"Statcast fetch failed and no cache: {exc}") from exc

    _CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(df.to_dict(orient="records")), encoding="utf-8"
    )
    return df


# ---------- Player-level accessors ----------

def get_batter_stats(year: int) -> dict[int, dict]:
    """Per-player Statcast batting stats. Keys are MLB player_id ints.

    Returns: xwoba, xba, barrel_pct, hard_hit, exit_velo, pa.
    xba is expected batting average — strips BABIP variance from the AVG
    signal, stabilises faster than raw AVG, and serves as a cleaner
    prior anchor than the flat league mean (h_per_ab=0.245).
    """
    df = _fetch_leaderboard(
        year, "batter",
        "pa,xwoba,xba,barrel_batted_rate,hard_hit_percent,avg_exit_velocity",
        min_pa=10,
        cache_key=f"batter_{_BAT_VER}",
    )
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        pid = _safe(row.get("player_id"))
        if pid is None:
            continue
        out[int(pid)] = {
            "xwoba":      _safe(row.get("xwoba")),
            "xba":        _safe(row.get("xba")),
            "barrel_pct": _safe(row.get("barrel_batted_rate")),
            "hard_hit":   _safe(row.get("hard_hit_percent")),
            "exit_velo":  _safe(row.get("avg_exit_velocity")),
            "pa":         _safe(row.get("pa"), 0.0),
        }
    return out


def get_pitcher_stats(year: int) -> dict[int, dict]:
    """Per-player Statcast pitching stats (contact quality allowed).
    Keys are MLB player_id ints.

    Returns: xera, xwoba (against), barrel_pct (allowed), whiff_percent
    (% of swings that miss — stabilises ~30 BF, much faster than K%),
    k_percent, pa (≈ batters faced).
    Values are raw; apply EB shrinkage via shrunk_pitcher_sc().
    """
    df = _fetch_leaderboard(
        year, "pitcher",
        "pa,xera,xwoba,barrel_batted_rate,hard_hit_percent,whiff_percent,k_percent,called_strike_percent",
        min_pa=10,
        cache_key=f"pitcher_{_PIT_VER}",
    )
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        pid = _safe(row.get("player_id"))
        if pid is None:
            continue
        out[int(pid)] = {
            "xera":       _safe(row.get("xera")),
            "xwoba":      _safe(row.get("xwoba")),
            "barrel_pct": _safe(row.get("barrel_batted_rate")),
            "hard_hit":   _safe(row.get("hard_hit_percent")),
            "whiff_pct":  _safe(row.get("whiff_percent")),
            "k_pct":      _safe(row.get("k_percent")),
            "csp":        _safe(row.get("called_strike_percent")),  # called-strike% — stabilises ~10 BF
            "bf":         _safe(row.get("pa"), 0.0),   # Savant returns "pa" for pitchers too
        }
    return out


# ---------- Team-level aggregation ----------

def get_team_batting(year: int,
                     player_team_map: dict[int, int]) -> dict[int, dict]:
    """PA-weighted team Statcast batting aggregates with EB shrinkage.

    player_team_map maps MLB player_id → team_id. Build it from the MLB Stats
    API bat_stats dict (which carries team_id per player):
        player_team_map = {pid: int(s["team_id"])
                           for pid, s in bat_stats.items() if s.get("team_id")}

    Returns dict[team_id → {xwoba, barrel_pct, hard_hit, pa}] — all shrunk.
    """
    bat = get_batter_stats(year)

    acc: dict[int, dict] = {}
    for pid, stats in bat.items():
        tid = player_team_map.get(pid)
        if tid is None:
            continue
        pa = stats.get("pa") or 0.0
        if pa <= 0:
            continue
        if tid not in acc:
            acc[tid] = {"xwoba_pa": 0.0, "barrel_pa": 0.0, "pa": 0.0}
        t = acc[tid]
        t["pa"] += pa
        if stats["xwoba"] is not None:
            t["xwoba_pa"]  += stats["xwoba"]      * pa
        if stats["barrel_pct"] is not None:
            t["barrel_pa"] += stats["barrel_pct"] * pa

    out: dict[int, dict] = {}
    for tid, t in acc.items():
        pa = t["pa"] or 1.0
        w  = pa / (pa + _PRIOR_PA_TEAM)
        raw_x = t["xwoba_pa"]  / pa if t["xwoba_pa"]  else LG_XWOBA
        raw_b = t["barrel_pa"] / pa if t["barrel_pa"] else LG_BARREL_PCT
        out[tid] = {
            "xwoba":      w * raw_x + (1 - w) * LG_XWOBA,
            "barrel_pct": w * raw_b + (1 - w) * LG_BARREL_PCT,
            "pa":         pa,
        }
    return out


# ---------- Fielding / defense ----------

def get_team_fielding(year: int,
                      player_team_map: dict[int, int]) -> dict[int, dict]:
    """Per-team aggregated defensive value from Statcast fielder leaderboard.

    Pulls Outs Above Average (OAA) and Fielding Runs Value — Savant's
    park/defense-neutral defensive metrics. Sums per-fielder values to team.

    OAA: extra outs vs an average fielder (positive = better). League neutral 0.
    FRV: defensive runs above average. About OAA * 0.78 (one out ~= 0.78 runs).

    Pass the same player_team_map used for batting (player_id -> team_id).
    Returns dict[team_id -> {oaa, frv, n_fielders}].
    """
    try:
        df = _fetch_leaderboard(
            year, "fielder",
            "fielding_runs_prevented,outs_above_average",
            min_pa=1,                  # min attempts, not PA
            cache_key=f"fielder_v1",
        )
    except Exception as exc:
        print(f"[statcast] fielder fetch failed: {exc}; returning empty")
        return {}

    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        pid = _safe(row.get("player_id"))
        if pid is None:
            continue
        tid = player_team_map.get(int(pid))
        if tid is None:
            continue
        oaa = _safe(row.get("outs_above_average"), 0.0)
        frv = _safe(row.get("fielding_runs_prevented"), 0.0)
        if oaa is None and frv is None:
            continue
        acc = out.setdefault(int(tid), {"oaa": 0.0, "frv": 0.0, "n": 0})
        acc["oaa"] += (oaa or 0.0)
        acc["frv"] += (frv or 0.0)
        acc["n"]   += 1
    return out


# ---------- Catcher framing ----------

def get_catcher_framing(year: int) -> dict[int, dict]:
    """Per-catcher framing runs from Statcast.

    framing_runs: extra called strikes converted to runs (positive = better).
    A great framer is worth +10 runs/season; average 0; bad -10.

    Returns dict[player_id -> {framing_runs}].
    """
    try:
        df = _fetch_leaderboard(
            year, "catcher",
            "runs_extra_strikes",
            min_pa=1,
            cache_key=f"catcher_v1",
        )
    except Exception as exc:
        print(f"[statcast] catcher fetch failed: {exc}; returning empty")
        return {}

    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        pid = _safe(row.get("player_id"))
        if pid is None:
            continue
        out[int(pid)] = {
            "framing_runs": _safe(row.get("runs_extra_strikes"), 0.0),
        }
    return out


# ---------- Pitcher arsenal ----------

def get_pitcher_arsenal(year: int) -> dict[int, dict]:
    """Per-pitcher pitch-mix profile from Baseball Savant.

    Returns dict[player_id -> {
        n_pitches:         total pitches thrown
        num_pitch_types:   count of pitch types thrown >= 5% of the time
        fb_avg_velo:       average fastball velocity (4-seam or sinker, whichever more used)
        velo_gap:          fastball velo minus changeup velo (deception)
        fastball_pct:      % of pitches that are FF or SI
    }]

    Pitch-mix diversity correlates with pitcher quality independently of
    K%/whiff% — a 3-pitch starter is more vulnerable on TTO than a 5-pitch
    starter even at the same overall stats. Velocity gap captures changeup
    effectiveness which raw stats miss.
    """
    sel = ("pitch_count,"
           "n_ff_formatted,ff_avg_speed,"
           "n_si_formatted,si_avg_speed,"
           "n_fc_formatted,fc_avg_speed,"
           "n_sl_formatted,sl_avg_speed,"
           "n_cu_formatted,cu_avg_speed,"
           "n_ch_formatted,ch_avg_speed,"
           "n_fs_formatted,fs_avg_speed,"
           "n_st_formatted,st_avg_speed")
    try:
        df = _fetch_leaderboard(
            year, "pitcher", sel, min_pa=30,
            cache_key=f"arsenal_v1",
        )
    except Exception as exc:
        print(f"[statcast] arsenal fetch failed: {exc}; returning empty")
        return {}

    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        pid = _safe(row.get("player_id"))
        if pid is None:
            continue
        # Pitch usage percentages per pitch type
        usages = {}
        velos = {}
        for code in ("ff", "si", "fc", "sl", "cu", "ch", "fs", "st"):
            pct = _safe(row.get(f"n_{code}_formatted"))
            v   = _safe(row.get(f"{code}_avg_speed"))
            if pct is not None and pct > 0:
                usages[code] = pct
                if v is not None and v > 0:
                    velos[code] = v
        if not usages:
            continue
        # Number of pitches used at least 5%
        num_pt = sum(1 for p in usages.values() if p >= 5.0)
        # Fastball velocity: average of FF+SI weighted by usage
        fb_velo = 0.0
        fb_pct  = 0.0
        for c in ("ff", "si"):
            if c in velos and c in usages:
                fb_velo += velos[c] * usages[c]
                fb_pct  += usages[c]
        fb_avg_velo = (fb_velo / fb_pct) if fb_pct > 0 else 92.0
        # Velo gap: FB velo - changeup velo (deception measure)
        ch_velo = velos.get("ch")
        velo_gap = (fb_avg_velo - ch_velo) if ch_velo else 8.0   # league avg ~8 mph
        out[int(pid)] = {
            "n_pitches":       _safe(row.get("pitch_count"), 0.0),
            "num_pitch_types": float(num_pt),
            "fb_avg_velo":     fb_avg_velo,
            "velo_gap":        velo_gap,
            "fastball_pct":    fb_pct,
        }
    return out


# ---------- Pitcher enrichment ----------

def shrunk_pitcher_sc(sc_stats: dict | None) -> dict:
    """Apply EB shrinkage to raw pitcher Statcast stats.

    Returns dict with keys: xera_sc, barrel_pct_sc.
    Falls back to league averages when sc_stats is None or missing fields.
    """
    if not sc_stats:
        return {"xera_sc": LG_XERA, "barrel_pct_sc": LG_BARREL_PCT}
    bf  = sc_stats.get("bf") or 0.0
    w   = bf / (bf + _PRIOR_BF_PIT) if bf > 0 else 0.0
    raw_xera   = sc_stats.get("xera")
    raw_barrel = sc_stats.get("barrel_pct")
    return {
        "xera_sc":       w * raw_xera   + (1 - w) * LG_XERA       if raw_xera   is not None else LG_XERA,
        "barrel_pct_sc": w * raw_barrel + (1 - w) * LG_BARREL_PCT  if raw_barrel is not None else LG_BARREL_PCT,
    }
