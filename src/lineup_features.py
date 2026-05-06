"""Lineup-weighted offense features for the team-runs model.

The team-aggregate offense features (`off_ops`, `off_woba`, `off_xwoba_sc`)
include every player who has appeared for the team — including bench, IL,
and rotated-out role players. On any given night the actual nine starters
are a subset, so team aggregates are biased toward the team's typical
*pool* rather than tonight's *lineup*.

This module computes PA-weighted aggregates over the 9 confirmed starters,
plus a platoon-adjusted xwOBA against the opposing starter's throwing hand.
Both are used as features in `model.FEATURES`.

For historical training rows the lineup is recovered from boxscore
`battingOrder` fields (digits ending in '00' = starter). For live
prediction the lineup comes from `mlb_api.extract_lineups()`.
"""
from __future__ import annotations
from typing import Iterable, Optional

# Typical PA share by batting order spot, normalized to sum to 1.0.
# From 2024-2025 league averages: leadoff gets ~12%, 9-hole gets ~9%.
LINEUP_PA_WEIGHTS = [0.121, 0.117, 0.114, 0.111, 0.108, 0.105, 0.103, 0.101, 0.100]

# League defaults for empty-bayes shrinkage of per-batter rates.
LG_OPS = 0.720
LG_WOBA = 0.315
LG_K_PCT = 0.225
LG_BB_PCT = 0.085
LG_ISO = 0.150
LG_XWOBA = 0.315


def _safe_float(x, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def extract_starting_lineup(boxscore: dict, side: str) -> list[int]:
    """Return the 9 starting batters for `side` ("home" or "away") in batting order.

    Reads `battingOrder` from each player entry. Starters have orders like
    "100", "200", ..., "900" (some games store as "0100" with leading zero).
    """
    pdata = (boxscore.get("teams", {}).get(side, {}).get("players", {}))
    starters: list[tuple[int, int]] = []   # (order_int, player_id)
    for _, pl in pdata.items():
        bo = pl.get("battingOrder")
        if not bo:
            continue
        bo_str = str(bo)
        if not bo_str.endswith("00"):
            continue
        try:
            order = int(bo_str)
        except ValueError:
            continue
        pid = pl.get("person", {}).get("id")
        if pid:
            starters.append((order, int(pid)))
    starters.sort(key=lambda t: t[0])
    return [pid for _, pid in starters[:9]]


def _batter_index(stats: dict) -> dict:
    """Empirical-Bayes shrunk per-batter rates from MLB API stats dict."""
    pa = _safe_float(stats.get("plateAppearances"), 0.0)
    ab = _safe_float(stats.get("atBats"), 0.0)
    h = _safe_float(stats.get("hits"))
    hr = _safe_float(stats.get("homeRuns"))
    bb = _safe_float(stats.get("baseOnBalls"))
    k = _safe_float(stats.get("strikeOuts"))
    tb = _safe_float(stats.get("totalBases"))
    obp = _safe_float(stats.get("onBasePercentage"), 0.0)
    slg = _safe_float(stats.get("slg"), 0.0)
    ops_raw = obp + slg if obp and slg else _safe_float(stats.get("ops"), 0.0)

    # Shrink to league mean by PA. At PA=150 the batter is ~50% themselves.
    prior = 150.0
    w = pa / (pa + prior) if pa > 0 else 0.0

    # wOBA-lite from box-score components (FanGraphs-ish weights).
    if pa > 0:
        d = _safe_float(stats.get("doubles"))
        t = _safe_float(stats.get("triples"))
        woba_raw = (0.69 * bb + 0.88 * (h - hr - d - t)
                    + 1.24 * d + 1.56 * t + 2.00 * hr) / pa
    else:
        woba_raw = LG_WOBA

    return {
        "ops":   w * ops_raw + (1 - w) * LG_OPS if ops_raw > 0 else LG_OPS,
        "woba":  w * woba_raw + (1 - w) * LG_WOBA,
        "k_pct": w * (k / pa if pa else LG_K_PCT) + (1 - w) * LG_K_PCT,
        "bb_pct": w * (bb / pa if pa else LG_BB_PCT) + (1 - w) * LG_BB_PCT,
        "iso":   w * ((tb - h) / ab if ab else LG_ISO) + (1 - w) * LG_ISO,
        "pa":    pa,
        "shrink_w": w,
    }


def lineup_offense(lineup_ids: Iterable[int], batter_stats: dict[int, dict]) -> dict:
    """PA-weighted offense aggregate over the 9 starters.

    `batter_stats` maps player_id -> raw stats dict (from mlb_api bulks). PAs
    used as the weight come from each batter's season PA — this captures
    "playing time" rather than relying on positional priors. Falls back to
    LINEUP_PA_WEIGHTS when no PA info is available.
    """
    indices = []
    for i, pid in enumerate(lineup_ids):
        s = batter_stats.get(int(pid))
        if not s:
            # Unknown batter — fall back to league averages
            indices.append((LINEUP_PA_WEIGHTS[i] if i < 9 else 0.1,
                            {"ops": LG_OPS, "woba": LG_WOBA, "k_pct": LG_K_PCT,
                             "bb_pct": LG_BB_PCT, "iso": LG_ISO}))
            continue
        idx = _batter_index(s)
        # Use season PA as weight when available; otherwise use positional default.
        w = idx["pa"] if idx["pa"] > 20 else (LINEUP_PA_WEIGHTS[i] * 100 if i < 9 else 50)
        indices.append((w, idx))

    if not indices:
        return {"ops": LG_OPS, "woba": LG_WOBA, "k_pct": LG_K_PCT,
                "bb_pct": LG_BB_PCT, "iso": LG_ISO, "n": 0}

    total = sum(w for w, _ in indices)
    out = {}
    for k in ("ops", "woba", "k_pct", "bb_pct", "iso"):
        out[k] = sum(w * idx[k] for w, idx in indices) / total
    out["n"] = len(indices)
    return out


def _wOBA_from_split(split_stats: dict, fallback: float) -> Optional[float]:
    """Compute wOBA-lite from a platoon-split stats dict. Returns None if no data."""
    pa = _safe_float(split_stats.get("plateAppearances"), 0.0)
    if pa < 5:
        return None
    bb = _safe_float(split_stats.get("baseOnBalls"))
    h = _safe_float(split_stats.get("hits"))
    hr = _safe_float(split_stats.get("homeRuns"))
    d = _safe_float(split_stats.get("doubles"))
    t = _safe_float(split_stats.get("triples"))
    woba = (0.69 * bb + 0.88 * (h - hr - d - t)
            + 1.24 * d + 1.56 * t + 2.00 * hr) / pa
    return woba


def lineup_xwoba_vs_hand(
    lineup_ids: Iterable[int],
    opp_throw: str,                    # "L" or "R"
    bat_vs_l: dict[int, dict],
    bat_vs_r: dict[int, dict],
    bat_sides: dict[int, str],
    batter_stats: dict[int, dict],
) -> float:
    """Average wOBA across the lineup vs the opposing starter's throwing hand.

    For each batter:
      - Switch hitters (bat_side == 'S') hit opposite to opp_throw.
      - Pull the batter's split stats vs the relevant pitcher hand.
      - EB-shrink to the batter's overall wOBA at PA=80 (split samples are noisier).
      - If split stats missing, fall back to overall wOBA.
    Then PA-weight (or simple-average if PA info is missing) across the 9 starters.
    """
    opp_throw = (opp_throw or "R").upper()
    splits_against = bat_vs_l if opp_throw == "L" else bat_vs_r

    contribs: list[tuple[float, float]] = []   # (weight, wOBA-vs-hand)
    for i, pid in enumerate(lineup_ids):
        pid = int(pid)
        season_stats = batter_stats.get(pid, {})
        season_idx = _batter_index(season_stats) if season_stats else None
        season_woba = season_idx["woba"] if season_idx else LG_WOBA
        weight = season_idx["pa"] if season_idx and season_idx["pa"] > 20 else (
            LINEUP_PA_WEIGHTS[i] * 100 if i < 9 else 50)

        # Switch hitter: bats opposite to pitcher's throw, so use the other split
        side = (bat_sides.get(pid, "") or "").upper()
        if side == "S":
            split_pool = bat_vs_r if opp_throw == "L" else bat_vs_l
        else:
            split_pool = splits_against

        split = split_pool.get(pid, {})
        split_woba = _wOBA_from_split(split, season_woba)
        if split_woba is None:
            contribs.append((weight, season_woba))
            continue

        # Shrink the split stat to the batter's overall wOBA. At PA_split=80
        # the split is ~50% itself.
        split_pa = _safe_float(split.get("plateAppearances"), 0.0)
        prior = 80.0
        w = split_pa / (split_pa + prior)
        adj = w * split_woba + (1 - w) * season_woba
        contribs.append((weight, adj))

    if not contribs:
        return LG_WOBA
    total = sum(w for w, _ in contribs)
    return sum(w * v for w, v in contribs) / total


def lineup_recent_form(
    lineup_ids: Iterable[int],
    batter_stats: dict[int, dict],
    batter_recent: dict[int, dict],
) -> dict:
    """PA-weighted recent-form aggregate over the 9 starters.

    Mirrors `lineup_offense` but uses each batter's last-14-day stats. When a
    starter has no recent stats (rookie call-up, returning from IL), falls
    back to that player's season line. When they have neither, falls back to
    league averages.

    The team-runs model historically used team-level recent form (`off_rpg_recent`)
    which averaged the entire active roster. A lineup of 5 hot hitters
    projects identically to a lineup of 5 cold hitters under that approach.
    This function fixes that by aggregating the actual nine starters.
    """
    indices = []
    for i, pid in enumerate(lineup_ids):
        rs = batter_recent.get(int(pid))
        ss = batter_stats.get(int(pid))
        # Prefer recent stats if the batter has a meaningful 14d sample;
        # otherwise fall back to season; otherwise league.
        if rs and _safe_float(rs.get("plateAppearances")) >= 10:
            stat = rs
        elif ss:
            stat = ss
        else:
            indices.append((LINEUP_PA_WEIGHTS[i] if i < 9 else 0.1,
                            {"ops": LG_OPS, "woba": LG_WOBA, "k_pct": LG_K_PCT,
                             "bb_pct": LG_BB_PCT, "iso": LG_ISO}))
            continue
        idx = _batter_index(stat)
        # PA weight: combination of season PA (playing time) and recent PA
        # (recency relevance). Use season PA as the primary weight so a hot
        # 30-PA cameo doesn't dominate over a regular's full-season sample.
        season_pa = _safe_float((ss or {}).get("plateAppearances"), 100.0) or 100.0
        indices.append((season_pa, idx))

    total_w = sum(w for w, _ in indices) or 1.0
    out = {"ops": 0.0, "woba": 0.0, "k_pct": 0.0, "bb_pct": 0.0, "iso": 0.0}
    for w, idx in indices:
        for k in out:
            out[k] += (w / total_w) * idx[k]
    return out


def lineup_lhb_pct(
    lineup_ids: Iterable[int],
    bat_sides: dict,
    pit_throws: str = "R",
) -> float:
    """Fraction of the lineup batting LEFT against this pitcher's throwing hand.
    Switch hitters bat opposite the pitcher hand (S vs RHP -> L; S vs LHP -> R).
    Returns 0.40 (league average LHB exposure) when the lineup is empty.
    """
    ids = list(lineup_ids or [])
    if not ids:
        return 0.40
    lhb = 0
    pit_throws = (pit_throws or "R").upper()
    for bid in ids:
        side = ((bat_sides.get(int(bid)) or bat_sides.get(str(bid)) or "R")).upper()
        if side == "L":
            lhb += 1
        elif side == "S":
            # Switch hitter — bats lefty vs RHP, righty vs LHP
            lhb += 1 if pit_throws == "R" else 0
    return lhb / len(ids)


def pitcher_split_vs_lineup(
    pitcher_id: int | None,
    lineup_ids: Iterable[int],
    bat_sides: dict,
    pit_throws: str,
    pit_vs_l: dict[int, dict],
    pit_vs_r: dict[int, dict],
) -> dict | None:
    """Lineup-mix-weighted pitcher platoon stats.

    Returns a synthetic stats dict representing the pitcher's K%, BB%, OPS-allowed
    blended by THIS specific lineup's hand composition. Returns None when no
    split data exists for the pitcher.

    Why this matters: a pitcher with a huge reverse-platoon (OPS .800 vs L,
    .600 vs R) facing a lineup of 7 RHB performs very differently than the
    season aggregate would suggest.  Until now we exposed batter-vs-pitcher
    splits but used pitcher-side aggregates.
    """
    if pitcher_id is None:
        return None
    vl = pit_vs_l.get(int(pitcher_id)) or pit_vs_l.get(str(pitcher_id))
    vr = pit_vs_r.get(int(pitcher_id)) or pit_vs_r.get(str(pitcher_id))
    if not vl and not vr:
        return None

    lhb_pct = lineup_lhb_pct(lineup_ids, bat_sides, pit_throws)

    def _rate(s: dict | None, num_key: str) -> float | None:
        if not s:
            return None
        denom = (_safe_float(s.get("battersFaced"))
                 or _safe_float(s.get("plateAppearances")))
        if denom <= 0:
            return None
        return _safe_float(s.get(num_key)) / denom

    def _ops(s: dict | None) -> float | None:
        if not s:
            return None
        v = s.get("ops")
        if v in (None, "", ".---"):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _blend(ml, mr, default):
        if ml is not None and mr is not None:
            return lhb_pct * ml + (1 - lhb_pct) * mr
        if ml is not None:
            return ml
        if mr is not None:
            return mr
        return default

    return {
        "k_pct":   _blend(_rate(vl, "strikeOuts"),   _rate(vr, "strikeOuts"),   0.225),
        "bb_pct":  _blend(_rate(vl, "baseOnBalls"),  _rate(vr, "baseOnBalls"),  0.085),
        "hr_pct":  _blend(_rate(vl, "homeRuns"),     _rate(vr, "homeRuns"),     0.030),
        "ops":     _blend(_ops(vl),                  _ops(vr),                  0.720),
        "lhb_pct": lhb_pct,
    }


def parse_lineup_ids(s: str | None) -> list[int]:
    """Parse a CSV-stored lineup_ids field back to a list of ints."""
    if not s or (isinstance(s, float)):
        return []
    s = str(s).strip()
    if not s or s in ("nan", "None"):
        return []
    out: list[int] = []
    for tok in s.split("|"):
        tok = tok.strip()
        if tok:
            try:
                out.append(int(tok))
            except ValueError:
                pass
    return out


def serialize_lineup_ids(ids: Iterable[int]) -> str:
    return "|".join(str(int(x)) for x in ids)
