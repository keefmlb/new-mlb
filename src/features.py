"""Feature extraction for MLB game prediction.

Each completed game becomes two training rows (one per team batting).
For prediction we build the same features for upcoming games.

We deliberately avoid leakage: when computing features for a game on date D
we only use stats accumulated *before* D. The MLB Stats API returns
season-to-date stats, so for backtesting we use rolling-window splits
(stats at the time of the game would require day-by-day re-pulls; instead
we use late-season stats and split into train/test on time).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import mlb_api, parks, weather, statcast as sc, lineup_features as lf


# ---------- Pitcher quality ----------
LEAGUE_FIP = 4.10            # 2024-2025 league average FIP
LEAGUE_K9 = 8.7
LEAGUE_BB9 = 3.2
LEAGUE_WHIP = 1.28
LEAGUE_WOBA = 0.315
LEAGUE_RPG = 4.50            # league avg runs scored per game per team


def _safe_float(x, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _ip_to_outs(ip_str) -> float:
    """MLB API returns IP as a string like '6.1' meaning 6 and 1/3 innings."""
    if ip_str is None or ip_str == "":
        return 0.0
    s = str(ip_str)
    if "." in s:
        whole, frac = s.split(".")
        return int(whole) * 3 + int(frac)
    return int(s) * 3


def pitcher_quality_index(stats: dict, sc_stats: dict | None = None) -> dict:
    """Normalize a pitcher's season stats into a small set of ratios.

    Uses Bayesian shrinkage toward the league mean — early-season noise gets
    pulled toward average until the pitcher has a meaningful sample.
    The shrinkage prior weight is in *batters faced* (BF). At BF=200 a starter's
    rate stats are about 50% themselves and 50% league.

    Pulls advanced metrics directly from the API saber endpoint when present:
    xFIP (more predictive than FIP), FIP- (park-adjusted), strikePercentage.

    sc_stats: optional Statcast pitcher stats (whiff_pct, k_pct). When whiff%
    is available, the K/9 shrinkage anchor is replaced with a whiff-derived
    K/9 prior — whiff% stabilises ~30 BF (vs ~70 for K%), so a high-whiff
    pitcher with low BF gets pulled toward "elite K/9" instead of league mean.
    May 2026 calibration: top-quartile pitcher K under-projected by 0.85 K/start
    on a flat LEAGUE_K9 prior.
    """
    bf = _safe_float(stats.get("battersFaced"), 0.0)
    outs = _ip_to_outs(stats.get("inningsPitched", 0))
    ip = outs / 3.0
    k = _safe_float(stats.get("strikeOuts"))
    bb = _safe_float(stats.get("baseOnBalls"))
    hr = _safe_float(stats.get("homeRuns"))
    h = _safe_float(stats.get("hits"))
    er = _safe_float(stats.get("earnedRuns"))

    # Lowered from 100 to 50 after calibration showed top-bin pitcher props
    # under-projected by ~20% (top-tier K stuff was over-regressed). At ~80 BF
    # for an end-of-April starter, w jumps from ~0.44 to ~0.62 on themselves.
    prior = 50.0
    w = bf / (bf + prior) if bf > 0 else 0.0
    # K/9 stabilises faster than other rate stats (~70 BF) AND showed the worst
    # compression in the May 2026 7-day backtest: top-quartile pitcher K bin
    # under-projected by 0.75 K (5.87 proj vs 6.62 actual). Use a tighter K-only
    # prior so power arms hold their K/9 more aggressively.
    # K/9 prior loosened from 25 to 15. May 2026 holdout still showed
    # top-decile pitcher K under-projecting by 1.48 K/start even with a
    # whiff-anchored prior — the season-rate shrinkage was still pulling
    # elites down. At BF=80, w_k jumps from 0.76 (prior 25) to 0.84 (prior 15).
    prior_k = 15.0
    w_k = bf / (bf + prior_k) if bf > 0 else 0.0

    # K/9 prior hierarchy (most → least direct signal):
    #   1. k_pct from Statcast (>= 50 BF): K% × 38.7 converts directly to K/9
    #      at the league-average 4.3 BF/IP rate. Most accurate for pitchers
    #      with 3+ starts; league at 22.5% → 8.7 K/9, elite 32% → 12.4 K/9.
    #   2. whiff_pct from Statcast (>= 15 BF): stabilises faster (~15 BF) so
    #      useful for 1-2 start samples where k_pct is still noisy. Coefficient
    #      0.38: league 25% whiff → 9.5 K/9, elite 35% → 13.3 K/9.
    #      BF floor lowered 30 → 15 so early-season starters hit the prior sooner.
    #   3. LEAGUE_K9 fallback when neither has enough sample.
    # May 2026 deep backtest: top-decile pitcher K under-projected by 0.833 K
    # with whiff-only prior. k_pct at 50 BF gives a tighter signal for elites.
    sc_bf    = (sc_stats or {}).get("bf") or 0.0
    sc_kpct  = (sc_stats or {}).get("k_pct")
    sc_whiff = (sc_stats or {}).get("whiff_pct")
    sc_csp   = (sc_stats or {}).get("csp")       # called-strike%; stabilises ~10 BF
    if sc_kpct is not None and sc_bf >= 50:
        # k_pct is reported as a percent (e.g. 27.3 for 27.3%)
        whiff_prior_k9 = max(4.0, min(16.0, (float(sc_kpct) / 100.0) * 38.7))
    elif sc_whiff is not None and sc_bf >= 15:
        # CSP and whiff together explain ~87% of K% variance (Sarris 2020).
        # When both are available, combine: whiff anchors the swing-miss
        # component, CSP anchors the command component. League avg:
        # whiff~25%, csp~17% → K9~8.7.  Weights (0.30 / 0.14) derived
        # from β-coefficients of K% on swStr% + csp% in 2023-25 seasons.
        if sc_csp is not None and sc_bf >= 15:
            whiff_prior_k9 = max(4.0, min(16.0,
                float(sc_whiff) * 0.30 + float(sc_csp) * 0.14))
        else:
            whiff_prior_k9 = max(4.0, min(16.0, float(sc_whiff) * 0.38))
    else:
        whiff_prior_k9 = LEAGUE_K9

    raw_k9 = (k * 9.0 / ip) if ip > 0 else LEAGUE_K9
    raw_bb9 = (bb * 9.0 / ip) if ip > 0 else LEAGUE_BB9
    raw_whip = ((bb + h) / ip) if ip > 0 else LEAGUE_WHIP
    raw_era = (er * 9.0 / ip) if ip > 0 else LEAGUE_RPG
    raw_hr9 = (hr * 9.0 / ip) if ip > 0 else 1.20

    # Prefer API-provided FIP / xFIP. xFIP is FIP with HRs normalized to league
    # rate — more stable for in-season projection. Fall back to local calc.
    api_fip = _safe_float(stats.get("fip"), None) if stats.get("fip") not in (None, "", "-.--") else None
    api_xfip = _safe_float(stats.get("xfip"), None) if stats.get("xfip") not in (None, "", "-.--") else None
    if ip > 0 and api_fip is None:
        api_fip = (13.0 * hr + 3.0 * bb - 2.0 * k) / ip + 3.10
    raw_fip = api_fip if api_fip is not None else LEAGUE_FIP
    raw_xfip = api_xfip if api_xfip is not None else raw_fip

    # FIP- is park-and-league adjusted; values <100 are above-average. The API
    # returns string ".---" when undefined.
    api_fip_minus = _safe_float(stats.get("fipMinus"), 100.0) or 100.0
    strike_pct = _safe_float(stats.get("strikePercentage"), 0.62) or 0.62

    # GB/FB tendency. groundOuts and airOuts together cover all balls in play
    # that produced an out (GB+FB+LD ~= GO+AO ignoring LD which API folds in).
    # gb_pct ranges ~0.30 (extreme FB pitcher) to ~0.62 (extreme GB pitcher);
    # league avg is ~0.43.
    # Why this matters: HR/9 lags badly for FB pitchers in HR parks (they give
    # up clusters of HR that the season HR rate hasn't caught up to), and GB
    # pitchers benefit disproportionately in HR-suppressing environments.
    gb_outs = _safe_float(stats.get("groundOuts"))
    fb_outs = _safe_float(stats.get("airOuts"))
    raw_gb_pct = (gb_outs / (gb_outs + fb_outs)) if (gb_outs + fb_outs) > 0 else 0.43
    gb_to_fb  = _safe_float(stats.get("groundOutsToAirouts"), 1.10)
    # Light shrinkage — GB% stabilises ~70 BIP (~150 BF), so we use the same
    # `w` as the rate stats (BF / (BF + 50)).
    gb_pct = w * raw_gb_pct + (1 - w) * 0.43

    return {
        "k9":   w_k * raw_k9 + (1 - w_k) * whiff_prior_k9,
        "bb9":  w * raw_bb9 + (1 - w) * LEAGUE_BB9,
        "whip": w * raw_whip + (1 - w) * LEAGUE_WHIP,
        "era":  w * raw_era + (1 - w) * LEAGUE_RPG,
        "fip":  w * raw_fip + (1 - w) * LEAGUE_FIP,
        "xfip": w * raw_xfip + (1 - w) * LEAGUE_FIP,
        "hr9":  w * raw_hr9 + (1 - w) * 1.20,
        "fip_minus": w * api_fip_minus + (1 - w) * 100.0,
        "strike_pct": strike_pct,
        "gb_pct":   gb_pct,
        "gb_to_fb": gb_to_fb,
        "bf":   bf,
        "ip":   ip,
        "shrink_w": w,
    }


# ---------- Team offense ----------
def team_offense_index(stats: dict) -> dict:
    """Normalize team offense stats: runs/G, K%, BB%, OPS, ISO + advanced (BABIP)."""
    g = _safe_float(stats.get("gamesPlayed"), 1.0) or 1.0
    pa = _safe_float(stats.get("plateAppearances"), 1.0) or 1.0
    ab = _safe_float(stats.get("atBats"), 1.0) or 1.0
    runs = _safe_float(stats.get("runs"))
    k = _safe_float(stats.get("strikeOuts"))
    bb = _safe_float(stats.get("baseOnBalls"))
    h = _safe_float(stats.get("hits"))
    hr = _safe_float(stats.get("homeRuns"))
    tb = _safe_float(stats.get("totalBases"))
    sf = _safe_float(stats.get("sacFlies"))
    obp = _safe_float(stats.get("onBasePercentage"), 0.320)
    slg = _safe_float(stats.get("slg"), 0.400)
    ops = obp + slg if obp and slg else _safe_float(stats.get("ops"), 0.720)
    avg = _safe_float(stats.get("avg"), 0.245)

    # Shrink toward league mean by games played
    prior_g = 25.0
    w = g / (g + prior_g)
    rpg_raw = runs / g if g else LEAGUE_RPG
    k_pct_raw = k / pa if pa else 0.225
    bb_pct_raw = bb / pa if pa else 0.085
    iso_raw = (tb - h) / ab if ab else 0.150

    # BABIP — regression-to-mean signal. League BABIP ~0.295. Hot teams will
    # regress toward this; cold teams will regress up.
    api_babip = stats.get("babip")
    if api_babip in (None, "", ".---"):
        babip_den = ab - k - hr + sf
        raw_babip = ((h - hr) / babip_den) if babip_den > 0 else 0.295
    else:
        raw_babip = _safe_float(api_babip, 0.295)
    babip = w * raw_babip + (1 - w) * 0.295

    # wOBA-lite — linear-weights run value per PA from box-score components.
    # Coefficients from FanGraphs 2024 weights, slightly rounded. Useful for
    # regression where OPS is "double-counted" with batting average.
    if pa > 0:
        woba_lite = (0.69 * bb + 0.88 * (h - hr - _safe_float(stats.get("doubles"))
                                          - _safe_float(stats.get("triples")))
                     + 1.24 * _safe_float(stats.get("doubles"))
                     + 1.56 * _safe_float(stats.get("triples"))
                     + 2.00 * hr) / pa
    else:
        woba_lite = 0.315
    woba_lite = w * woba_lite + (1 - w) * 0.315

    return {
        "rpg":   w * rpg_raw + (1 - w) * LEAGUE_RPG,
        "k_pct": w * k_pct_raw + (1 - w) * 0.225,
        "bb_pct": w * bb_pct_raw + (1 - w) * 0.085,
        "iso":   w * iso_raw + (1 - w) * 0.150,
        "ops":   w * ops + (1 - w) * 0.720,
        "avg":   w * avg + (1 - w) * 0.245,
        "babip": babip,
        "woba":  woba_lite,
        "hr_rate": (hr / pa) if pa else 0.030,
        "g":     g,
    }


def team_pitching_index(stats: dict) -> dict:
    """Bullpen / staff-wide pitching profile (used for relief innings)."""
    return pitcher_quality_index(stats)


# ---------- Weather effects ----------
def weather_adjustment(park: parks.Park, w: dict) -> dict:
    """Compute multiplicative scaling factors on runs / HR from weather.

    Baseline is league-average warm dry game (75°F, 5 mph wind).
    Domes & closed retractable roofs neutralize wind/precip but not temperature
    (we still respect altitude — Coors is open).
    """
    if park.roof == "dome":
        return {"runs_mult": 1.00, "hr_mult": 1.00, "rain_risk": 0.0,
                "wind_to_cf_mph": 0.0, "temp_f": 72.0}

    temp_f = w.get("temp_f", 70.0)
    humidity = w.get("humidity", 50.0)
    wind_mph = w.get("wind_mph", 5.0)
    wind_dir = w.get("wind_dir_deg", 0.0)
    precip = w.get("precip_in", 0.0)

    # Baseball physics:
    #   ~+1% on HR distance per ~10°F above 70.  HR rate ~ 4-6%/10°F.
    #   Out-blowing wind: ~1% per mph beyond 5 mph baseline.
    #   In-blowing wind: ~0.7% per mph (stronger suppression of fly balls than carry adds).
    #   High humidity slightly reduces HR (denser air? actually moist air is *less* dense — but
    #   the ball gets heavier when humid; net effect is small).
    temp_factor = 1.0 + 0.005 * (temp_f - 70.0)            # runs
    temp_factor_hr = 1.0 + 0.010 * (temp_f - 70.0)         # HR

    cf_wind = weather.wind_component_to_cf(wind_mph, wind_dir, park.orientation_deg)
    if park.roof == "retractable":
        # Assume closed in rain or extreme cold/heat. Otherwise open.
        closed = precip > 0.05 or temp_f < 55.0 or temp_f > 95.0
        if closed:
            cf_wind = 0.0
            temp_f_eff = 72.0
            temp_factor = 1.0 + 0.005 * (temp_f_eff - 70.0)
            temp_factor_hr = 1.0 + 0.010 * (temp_f_eff - 70.0)

    wind_factor_runs = 1.0 + 0.006 * cf_wind     # ~0.6%/mph effective CF wind
    wind_factor_hr = 1.0 + 0.018 * cf_wind       # ~1.8%/mph

    humidity_factor = 1.0 - 0.0005 * (humidity - 50.0)

    return {
        "runs_mult": max(0.7, min(1.4, temp_factor * wind_factor_runs * humidity_factor)),
        "hr_mult":   max(0.5, min(1.7, temp_factor_hr * wind_factor_hr * humidity_factor)),
        "rain_risk": min(1.0, precip / 0.25),     # 0.25" in the hour ≈ near-certain delay
        "wind_to_cf_mph": cf_wind,
        "temp_f": temp_f,
    }


# ---------- Game-level feature row ----------
@dataclass
class GameFeatures:
    game_pk: int
    date: str
    venue: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_score: Optional[int]
    away_score: Optional[int]
    is_final: bool

    # Park
    park_pf_runs: float
    park_pf_hr: float
    park_elev_ft: int
    park_roof: str

    # Weather-derived
    runs_mult: float
    hr_mult: float
    wind_to_cf_mph: float
    temp_f: float

    # Team offense (batting team perspective)
    home_off_rpg: float; home_off_k_pct: float; home_off_bb_pct: float
    home_off_iso: float; home_off_ops: float
    away_off_rpg: float; away_off_k_pct: float; away_off_bb_pct: float
    away_off_iso: float; away_off_ops: float

    # Probable starters
    home_sp_id: Optional[int]; home_sp_name: str
    home_sp_fip: float; home_sp_k9: float; home_sp_bb9: float; home_sp_era: float; home_sp_hr9: float
    away_sp_id: Optional[int]; away_sp_name: str
    away_sp_fip: float; away_sp_k9: float; away_sp_bb9: float; away_sp_era: float; away_sp_hr9: float

    # Bullpen (team-pitching aggregate minus starters; we use full staff as proxy)
    home_bp_era: float; home_bp_fip: float
    away_bp_era: float; away_bp_fip: float

    # Recent form (last 14 days). Always present; equals season values if no recent data.
    home_off_rpg_recent: float; home_off_ops_recent: float
    away_off_rpg_recent: float; away_off_ops_recent: float
    home_pit_era_recent: float; away_pit_era_recent: float

    # Advanced (saber)
    home_sp_xfip: float; away_sp_xfip: float
    home_sp_fip_minus: float; away_sp_fip_minus: float
    home_off_woba: float; away_off_woba: float
    home_off_babip: float; away_off_babip: float

    # Engineered interactions
    home_off_x_park: float; away_off_x_park: float       # offense × park run-factor
    home_sp_x_off: float;   away_sp_x_off: float         # opp pitcher quality × team offense
    home_off_minus_oppsp: float; away_off_minus_oppsp: float   # offense (rpg) minus opp pitcher era

    # Statcast (Baseball Savant) — park/defense-neutral contact quality
    # Falls back to league average when data is unavailable.
    home_off_xwoba_sc: float; away_off_xwoba_sc: float   # team xwOBA (batting, EB-shrunk)
    home_off_barrel_rate: float; away_off_barrel_rate: float  # team barrel % (0-100)
    home_sp_xera_sc: float; away_sp_xera_sc: float        # starter xERA (EB-shrunk)

    # Umpire — HP umpire K-rate multiplier (1.0 = league avg; EB-shrunk)
    ump_k_mult: float = 1.0

    # Lineup-weighted offense (PA-weighted aggregate over the 9 starters).
    # Defaults match the team-aggregate values when no lineup is supplied so
    # rows from older datasets stay sane.
    home_lineup_ops: float = 0.720; away_lineup_ops: float = 0.720
    home_lineup_woba: float = 0.315; away_lineup_woba: float = 0.315
    home_lineup_k_pct: float = 0.225; away_lineup_k_pct: float = 0.225
    home_lineup_bb_pct: float = 0.085; away_lineup_bb_pct: float = 0.085
    # Platoon-adjusted: lineup wOBA vs OPPOSING starter's throwing hand.
    home_lineup_xwoba_vs_hand: float = 0.315
    away_lineup_xwoba_vs_hand: float = 0.315
    # Lineup-weighted recent form (last-14d, computed over the 9 starters).
    # Distinguishes "5 of 9 starters in 14d slumps" from "team's bench is hot".
    home_lineup_woba_recent: float = 0.315; away_lineup_woba_recent: float = 0.315
    home_lineup_ops_recent:  float = 0.720; away_lineup_ops_recent:  float = 0.720
    # Pipe-delimited starter player_ids ("123|456|...") for downstream use.
    home_lineup_ids: str = ""; away_lineup_ids: str = ""


def build_game_features(
    game: dict,
    team_off: dict[int, dict],
    team_pit: dict[int, dict],
    pitcher_stats: dict[int, dict],
    team_off_recent: dict[int, dict] | None = None,
    team_pit_recent: dict[int, dict] | None = None,
    sc_team_bat: dict[int, dict] | None = None,
    sc_pit: dict[int, dict] | None = None,
    home_lineup_ids: list[int] | None = None,
    away_lineup_ids: list[int] | None = None,
    batter_stats: dict[int, dict] | None = None,
    bat_vs_l: dict[int, dict] | None = None,
    bat_vs_r: dict[int, dict] | None = None,
    bat_sides: dict[int, str] | None = None,
    pit_throws: dict[int, str] | None = None,
    batter_recent: dict[int, dict] | None = None,
) -> Optional[GameFeatures]:
    """Build one feature row for a scheduled game given pre-game stats lookups.

    `team_off`, `team_pit`, `pitcher_stats` map id -> raw stats (from mlb_api bulks).
    Returns None if essential fields are missing.
    """
    teams = game.get("teams", {})
    home = teams.get("home", {}); away = teams.get("away", {})
    home_team = home.get("team", {}); away_team = away.get("team", {})
    home_tid = home_team.get("id"); away_tid = away_team.get("id")
    if not home_tid or not away_tid:
        return None

    venue_name = game.get("venue", {}).get("name", "")
    park = parks.get_park(venue_name)

    when = mlb_api.parse_game_time(game)
    w = weather.get_weather(park.lat, park.lon, when)
    wadj = weather_adjustment(park, w)

    home_off = team_offense_index(team_off.get(home_tid, {}))
    away_off = team_offense_index(team_off.get(away_tid, {}))
    home_pit = team_pitching_index(team_pit.get(home_tid, {}))
    away_pit = team_pitching_index(team_pit.get(away_tid, {}))

    # Recent form: blend last-14-day stats with season; defaults to season if no recent data.
    tor = team_off_recent or {}
    tpr = team_pit_recent or {}
    home_off_recent = team_offense_index(tor.get(home_tid, team_off.get(home_tid, {})))
    away_off_recent = team_offense_index(tor.get(away_tid, team_off.get(away_tid, {})))
    home_pit_recent = team_pitching_index(tpr.get(home_tid, team_pit.get(home_tid, {})))
    away_pit_recent = team_pitching_index(tpr.get(away_tid, team_pit.get(away_tid, {})))

    home_sp = home.get("probablePitcher") or {}
    away_sp = away.get("probablePitcher") or {}
    home_sp_id = home_sp.get("id")
    away_sp_id = away_sp.get("id")

    home_sp_q = pitcher_quality_index(pitcher_stats.get(home_sp_id, {})) if home_sp_id else pitcher_quality_index({})
    away_sp_q = pitcher_quality_index(pitcher_stats.get(away_sp_id, {})) if away_sp_id else pitcher_quality_index({})

    state = (game.get("status") or {}).get("codedGameState", "")
    is_final = state == "F"

    # Statcast: team batting xwOBA / barrel% and starter xERA
    _sc_bat = sc_team_bat or {}
    home_sc_bat = _sc_bat.get(home_tid, {})
    away_sc_bat = _sc_bat.get(away_tid, {})
    home_off_xwoba_sc   = home_sc_bat.get("xwoba",      sc.LG_XWOBA)
    away_off_xwoba_sc   = away_sc_bat.get("xwoba",      sc.LG_XWOBA)
    home_off_barrel_rate = home_sc_bat.get("barrel_pct", sc.LG_BARREL_PCT)
    away_off_barrel_rate = away_sc_bat.get("barrel_pct", sc.LG_BARREL_PCT)

    _sc_p = sc_pit or {}
    home_sp_sc = sc.shrunk_pitcher_sc(_sc_p.get(home_sp_id) if home_sp_id else None)
    away_sp_sc = sc.shrunk_pitcher_sc(_sc_p.get(away_sp_id) if away_sp_id else None)
    home_sp_xera_sc = home_sp_sc["xera_sc"]
    away_sp_xera_sc = away_sp_sc["xera_sc"]

    # Lineup features. When no lineup is supplied, fall back to team aggregates
    # so older callers and historical rows missing lineup data stay coherent.
    _bs = batter_stats or {}
    _bvl = bat_vs_l or {}
    _bvr = bat_vs_r or {}
    _bsides = bat_sides or {}
    _pthrows = pit_throws or {}

    home_lu = list(home_lineup_ids or [])
    away_lu = list(away_lineup_ids or [])

    if home_lu and _bs:
        h_lu_off = lf.lineup_offense(home_lu, _bs)
    else:
        h_lu_off = {"ops": home_off["ops"], "woba": home_off["woba"],
                    "k_pct": home_off["k_pct"], "bb_pct": home_off["bb_pct"]}
    if away_lu and _bs:
        a_lu_off = lf.lineup_offense(away_lu, _bs)
    else:
        a_lu_off = {"ops": away_off["ops"], "woba": away_off["woba"],
                    "k_pct": away_off["k_pct"], "bb_pct": away_off["bb_pct"]}

    # Platoon match-up: home lineup vs AWAY starter's throwing hand, etc.
    away_sp_throws = (_pthrows.get(away_sp_id, "") or "R").upper() if away_sp_id else "R"
    home_sp_throws = (_pthrows.get(home_sp_id, "") or "R").upper() if home_sp_id else "R"
    if home_lu and _bs and (_bvl or _bvr):
        h_lu_vs = lf.lineup_xwoba_vs_hand(home_lu, away_sp_throws, _bvl, _bvr, _bsides, _bs)
    else:
        h_lu_vs = h_lu_off["woba"]
    if away_lu and _bs and (_bvl or _bvr):
        a_lu_vs = lf.lineup_xwoba_vs_hand(away_lu, home_sp_throws, _bvl, _bvr, _bsides, _bs)
    else:
        a_lu_vs = a_lu_off["woba"]

    # Lineup recent form — aggregate per-batter 14d stats over the actual 9
    # starters. Falls back to season aggregate when no recent data.
    _br = batter_recent or {}
    if home_lu and _bs and _br:
        h_lu_recent = lf.lineup_recent_form(home_lu, _bs, _br)
    else:
        h_lu_recent = {"ops": h_lu_off["ops"], "woba": h_lu_off["woba"]}
    if away_lu and _bs and _br:
        a_lu_recent = lf.lineup_recent_form(away_lu, _bs, _br)
    else:
        a_lu_recent = {"ops": a_lu_off["ops"], "woba": a_lu_off["woba"]}

    return GameFeatures(
        game_pk=game.get("gamePk"),
        date=game.get("officialDate") or (game.get("gameDate", "")[:10]),
        venue=venue_name,
        home_team=home_team.get("name", ""),
        away_team=away_team.get("name", ""),
        home_team_id=home_tid,
        away_team_id=away_tid,
        home_score=home.get("score"),
        away_score=away.get("score"),
        is_final=is_final,

        park_pf_runs=park.pf_runs,
        park_pf_hr=park.pf_hr,
        park_elev_ft=park.elevation_ft,
        park_roof=park.roof,

        runs_mult=wadj["runs_mult"],
        hr_mult=wadj["hr_mult"],
        wind_to_cf_mph=wadj["wind_to_cf_mph"],
        temp_f=wadj["temp_f"],

        home_off_rpg=home_off["rpg"], home_off_k_pct=home_off["k_pct"], home_off_bb_pct=home_off["bb_pct"],
        home_off_iso=home_off["iso"], home_off_ops=home_off["ops"],
        away_off_rpg=away_off["rpg"], away_off_k_pct=away_off["k_pct"], away_off_bb_pct=away_off["bb_pct"],
        away_off_iso=away_off["iso"], away_off_ops=away_off["ops"],

        home_sp_id=home_sp_id, home_sp_name=home_sp.get("fullName", ""),
        home_sp_fip=home_sp_q["fip"], home_sp_k9=home_sp_q["k9"], home_sp_bb9=home_sp_q["bb9"],
        home_sp_era=home_sp_q["era"], home_sp_hr9=home_sp_q["hr9"],
        away_sp_id=away_sp_id, away_sp_name=away_sp.get("fullName", ""),
        away_sp_fip=away_sp_q["fip"], away_sp_k9=away_sp_q["k9"], away_sp_bb9=away_sp_q["bb9"],
        away_sp_era=away_sp_q["era"], away_sp_hr9=away_sp_q["hr9"],

        home_bp_era=home_pit["era"], home_bp_fip=home_pit["fip"],
        away_bp_era=away_pit["era"], away_bp_fip=away_pit["fip"],

        home_off_rpg_recent=home_off_recent["rpg"], home_off_ops_recent=home_off_recent["ops"],
        away_off_rpg_recent=away_off_recent["rpg"], away_off_ops_recent=away_off_recent["ops"],
        home_pit_era_recent=home_pit_recent["era"], away_pit_era_recent=away_pit_recent["era"],

        home_sp_xfip=home_sp_q.get("xfip", LEAGUE_FIP),
        away_sp_xfip=away_sp_q.get("xfip", LEAGUE_FIP),
        home_sp_fip_minus=home_sp_q.get("fip_minus", 100.0),
        away_sp_fip_minus=away_sp_q.get("fip_minus", 100.0),
        home_off_woba=home_off["woba"], away_off_woba=away_off["woba"],
        home_off_babip=home_off["babip"], away_off_babip=away_off["babip"],

        # Interactions: a team's offensive context against this specific matchup
        home_off_x_park=home_off["ops"] * park.pf_runs,
        away_off_x_park=away_off["ops"] * park.pf_runs,
        home_sp_x_off=home_sp_q.get("xfip", LEAGUE_FIP) * away_off["ops"],     # away batting vs home SP
        away_sp_x_off=away_sp_q.get("xfip", LEAGUE_FIP) * home_off["ops"],     # home batting vs away SP
        home_off_minus_oppsp=home_off["rpg"] - away_sp_q.get("era", LEAGUE_RPG),
        away_off_minus_oppsp=away_off["rpg"] - home_sp_q.get("era", LEAGUE_RPG),

        # Statcast
        home_off_xwoba_sc=home_off_xwoba_sc,
        away_off_xwoba_sc=away_off_xwoba_sc,
        home_off_barrel_rate=home_off_barrel_rate,
        away_off_barrel_rate=away_off_barrel_rate,
        home_sp_xera_sc=home_sp_xera_sc,
        away_sp_xera_sc=away_sp_xera_sc,

        # Lineup-weighted offense
        home_lineup_ops=h_lu_off["ops"], away_lineup_ops=a_lu_off["ops"],
        home_lineup_woba=h_lu_off["woba"], away_lineup_woba=a_lu_off["woba"],
        home_lineup_k_pct=h_lu_off["k_pct"], away_lineup_k_pct=a_lu_off["k_pct"],
        home_lineup_bb_pct=h_lu_off["bb_pct"], away_lineup_bb_pct=a_lu_off["bb_pct"],
        home_lineup_xwoba_vs_hand=h_lu_vs, away_lineup_xwoba_vs_hand=a_lu_vs,
        home_lineup_woba_recent=h_lu_recent["woba"], away_lineup_woba_recent=a_lu_recent["woba"],
        home_lineup_ops_recent=h_lu_recent["ops"],   away_lineup_ops_recent=a_lu_recent["ops"],
        home_lineup_ids=lf.serialize_lineup_ids(home_lu),
        away_lineup_ids=lf.serialize_lineup_ids(away_lu),
    )
