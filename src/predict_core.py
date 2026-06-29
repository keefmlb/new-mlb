"""Core prediction logic — used by both the CLI and the Streamlit app.

`predict_slate()` returns a structured `SlateResult` containing per-game
predictions, projections, sportsbook lines, and value bets. Pure data — no
printing or plotting.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import mlb_api, parks, weather, features as feats, statcast as sc
from . import model as mdl, projections as proj, odds, value, name_match, bet_tracker, umpire as ump, parlays
from . import lineup_features as lf


ROOT = Path(__file__).resolve().parent.parent

# How much to trust the betting market over the model when pricing GAME LINES.
# The closing line is the sharpest single signal (injuries, scratches, weather,
# sharp money); our team-runs model beats a constant baseline by only ~2.5% on
# totals. 0.5 = equal trust. Applied to ML / total / run-line ONLY; player
# projections keep the raw model output.
#
# The blend is applied in PROBABILITY space inside value.evaluate_game_lines:
# raw model prob -> fitted calibration -> blend toward the book's no-vig prob.
# (The run-prediction lambdas are still blended via blend_to_market, but only
# for the DISPLAYED per-team runs / totals on the game card — pricing no
# longer re-derives probabilities from blended lambdas, which double-shrunk.)
MARKET_BLEND_WEIGHT = 0.50

# Tag stamped onto every logged pick so the bet log can be segmented by the
# filter/calibration policy in force. Bump when betting rules change.
# 2026-06-10: fitted per-market prop calibration replaced the blanket 0.70
# logit shrink (prop_calibration.json) — changes every prop probability and
# hence the leaderboard, so it's a new policy.
POLICY_VERSION = "2026-06-24-tb-analytical"
# Raw model-vs-market total disagreement (runs) beyond which game-line bets are
# suppressed entirely — that magnitude of disagreement is model error, not edge.
MARKET_MAX_TOTAL_DISAGREE = 3.0
# Minimum EV/$ for a sharp-value bet to be surfaced (Polymarket prices carry
# bid-ask spread and a couple points of noise; 2% is the floor that was
# previously inlined at the call site).
SHARP_MIN_EV = 0.02


def _persist_closing_lines(rows: list[dict]) -> None:
    """Append per-game market lines to a PERMANENT, game-linked dataset at
    data/odds/closing_lines.csv. De-dupes by game_pk keeping the LATEST capture
    (closest to first pitch ≈ closest to the closing line). This accumulates the
    closing-line history we need to (a) backtest a market-blend weight and
    (b) feature-ize line value — neither is possible from the 14-day rolling
    odds_history snapshot. Best-effort; never raises into the prediction path.
    """
    if not rows:
        return
    import csv as _csv
    path = ROOT / "data" / "odds" / "closing_lines.csv"
    fields = ["game_pk", "date", "away_team", "home_team", "captured_at",
              "market_total", "ml_home", "ml_away",
              "rl_line", "rl_home", "rl_away",
              "sharp_p_home", "sharp_total", "sharp_p_over"]
    try:
        existing: dict[int, dict] = {}
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as fh:
                for r in _csv.DictReader(fh):
                    try:
                        existing[int(r["game_pk"])] = r
                    except (TypeError, ValueError, KeyError):
                        continue
        for row in rows:
            gpk = int(row["game_pk"])
            prev = existing.get(gpk)
            # Keep whichever capture is later (closer to close).
            if prev is None or str(row["captured_at"]) >= str(prev.get("captured_at", "")):
                existing[gpk] = {k: row.get(k, prev.get(k) if prev else "") for k in fields}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for gpk in sorted(existing):
                w.writerow({k: existing[gpk].get(k, "") for k in fields})
    except Exception:
        pass


def _persist_closing_props(rows: list[dict]) -> None:
    """Append per-player prop lines to a PERMANENT log at
    data/odds/closing_props.csv. De-dupes by (game_pk, player_id, market,
    LINE) keeping the LATEST capture (closest to first pitch ≈ closing line).
    The line is part of the key because the feed carries ALT lines ("TB 5.5
    +1500") alongside the main line — collapsing them (the original bug)
    made an alt line masquerade as the close and corrupted line-move CLV.
    This is the prop counterpart of _persist_closing_lines: it makes CLV
    computable for logged prop bets, which are most of the leaderboard and
    where the bet log shows the worst bleed. Best-effort; never raises into
    the prediction path.
    """
    if not rows:
        return
    import csv as _csv
    path = ROOT / "data" / "odds" / "closing_props.csv"
    fields = ["game_pk", "date", "player", "player_id", "market",
              "line", "over", "under", "captured_at"]

    def _key(r: dict) -> tuple:
        pid = r.get("player_id")
        try:
            pid = int(pid) if pid not in (None, "", "0", 0) else None
        except (TypeError, ValueError):
            pid = None
        who = pid if pid else (r.get("player") or "").strip().lower()
        try:
            line = f"{float(r.get('line') or 0):g}"
        except (TypeError, ValueError):
            line = "0"
        return (int(r["game_pk"]), who, r.get("market") or "", line)

    try:
        existing: dict[tuple, dict] = {}
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as fh:
                for r in _csv.DictReader(fh):
                    try:
                        existing[_key(r)] = r
                    except (TypeError, ValueError, KeyError):
                        continue
        for row in rows:
            k = _key(row)
            prev = existing.get(k)
            if prev is None or str(row["captured_at"]) >= str(prev.get("captured_at", "")):
                existing[k] = {f: row.get(f, prev.get(f) if prev else "") for f in fields}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for k in sorted(existing, key=str):
                w.writerow({f: existing[k].get(f, "") for f in fields})
    except Exception:
        pass


def _team_runs_rate_factor() -> float:
    """Load the recent team-runs rate factor (actual_mean / pred_mean over
    last N days). Captures league-wide run-environment shifts that the static
    feature set can't see. Clamped to [0.85, 1.15] by the fit script.
    Returns 1.0 when the file is missing.
    """
    p = ROOT / "data" / "models" / "team_runs_rate.json"
    if not p.exists():
        return 1.0
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("rate_factor", 1.0))
    except Exception:
        return 1.0


# ---------- Data classes ----------
@dataclass
class GamePrediction:
    game_pk: int
    date: str
    venue: str
    park_roof: str
    park_pf_runs: float
    park_pf_hr: float

    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    first_pitch_utc: str             # ISO timestamp

    home_sp_id: Optional[int]
    home_sp_name: str
    home_sp_fip: float
    home_sp_xfip: float
    away_sp_id: Optional[int]
    away_sp_name: str
    away_sp_fip: float
    away_sp_xfip: float

    # Weather for first pitch
    temp_f: float
    wind_to_cf_mph: float
    runs_mult: float
    hr_mult: float

    # Predictions
    pred_home_runs: float
    pred_away_runs: float
    pred_total: float
    p_home_win: float
    p_over_8_5: float

    # Player projections
    home_batters: list[dict] = field(default_factory=list)
    away_batters: list[dict] = field(default_factory=list)
    home_starter: Optional[dict] = None
    away_starter: Optional[dict] = None

    # Sportsbook lines (None if not available)
    book: Optional[dict] = None
    book_source: str = "none"

    # Value bets (filtered to >= edge_threshold)
    game_value: list[dict] = field(default_factory=list)
    prop_value: list[dict] = field(default_factory=list)
    # Every evaluated bet for this game, regardless of edge — used by the
    # Pure Confidence leaderboard (model-certainty only, ignores book agreement).
    all_bets: list[dict] = field(default_factory=list)
    # True only when BOTH starters are announced. UI flags this as a
    # warning and the leaderboard's Top 5 panel excludes bets from
    # unconfirmed games (since pitcher projections fall back to defaults).
    starters_confirmed: bool = True


@dataclass
class SlateResult:
    target_date: str
    odds_source: str
    n_games: int
    n_books: int
    n_props_loaded: int
    games: list[GamePrediction]
    top_value: list[dict]            # ranked across slate
    concentration_warning: Optional[str] = None
    # Slate-wide unfiltered bet pool for Pure Confidence ranking.
    all_bets: list[dict] = field(default_factory=list)
    # Sharp-strategy audit: {"games_checked": int, "n_bets": int,
    # "best_ev": float|None}. games_checked counts games with a Polymarket
    # reference AND a comparable Fanatics line; best_ev is the best EV/$ seen
    # across every comparable side, even when nothing cleared SHARP_MIN_EV —
    # this is what tells you whether the book was efficient or the detector
    # is broken.
    sharp_summary: Optional[dict] = None
    # Bets where the line sits inside the MODEL's confidence region (raw model
    # side prob ≥ MODEL_CI_THRESHOLD), deduped to the best line per pick.
    model_ci_bets: list[dict] = field(default_factory=list)
    # Daily skill-backed parlays (runs/rbi legs, high conviction, one/game).
    parlays: list[dict] = field(default_factory=list)
    # Daily HR-specific parlays (2- and 3-leg OVER 0.5 HR, one player/game,
    # ranked purely by the model's HR conviction since HR is inherently a
    # low-probability event with no useful min_conf threshold).
    hr_parlays: list[dict] = field(default_factory=list)


# Raw model side-probability at/above which a bet is "inside the model's CI".
MODEL_CI_THRESHOLD = 0.60


# ---------- Helpers ----------
def _stats_lookup(d: dict) -> dict[int, dict]:
    return {int(k): v for k, v in d.items()}


def _team_name_match(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    n = needle.lower(); h = haystack.lower()
    return n in h or h in n


def _find_book(books: list[dict], home: str, away: str,
               first_pitch: datetime | None = None) -> dict | None:
    """Find the book entry for a matchup. When several entries match the same
    team pair (doubleheaders), pick the one whose commence_time is closest to
    the game's first pitch — otherwise game 2 silently gets game 1's lines."""
    matches = [b for b in books
               if _team_name_match(b.get("home_team", ""), home)
               and _team_name_match(b.get("away_team", ""), away)]
    if not matches:
        return None
    if len(matches) == 1 or first_pitch is None:
        return matches[0]

    def _time_gap(b: dict) -> float:
        try:
            t = str(b.get("commence_time") or "")
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return abs((dt - first_pitch).total_seconds())
        except (TypeError, ValueError):
            return float("inf")

    return min(matches, key=_time_gap)


def _vb_to_dict(vb: value.ValueBet) -> dict:
    """Render a ValueBet as a JSON-friendly dict."""
    d = asdict(vb)
    d["edge_pct"] = round(d["edge_pct"], 2)
    d["model_prob"] = round(d["model_prob"], 4)
    d["novig_prob"] = round(d["novig_prob"], 4)
    d["ev_per_dollar"] = round(d["ev_per_dollar"], 4)
    d["kelly"] = round(d["kelly"], 4)
    d["confidence"] = round(d.get("confidence", 0.0), 4)
    d["score"] = round(d.get("score", 0.0), 3)
    d["model_prob_raw"] = round(d.get("model_prob_raw") or d["model_prob"], 4)
    return d


# Per-market minimum edge floors — noisy markets need a bigger edge to be
# worth flagging. Based on May 2026 21-day backtest R^2:
#   - prop_runs       R^2 = 0.049   -> need 7% edge
#   - prop_rbi        R^2 = 0.051   -> need 7% edge
#   - prop_tb         R^2 = 0.063   -> need 6% edge
#   - prop_hits       R^2 = 0.064   -> need 6% edge
#   - prop_pitcher_er R^2 = 0.068   -> need 6% edge
#   - prop_pitcher_bb R^2 = 0.114   -> need 5% edge (borderline)
#   - prop_pitcher_hr R^2 = 0.114   -> need 5% edge (borderline)
# Other markets fall through to the user's slider value.
# Floors are applied to the POST-market-blend edge (the model prob is pulled
# halfway to the no-vig market prob before the edge is computed), so a 5% floor
# here corresponds to roughly a 10% raw model-vs-market disagreement.
_MARKET_MIN_EDGE_PCT: dict[str, float] = {
    "prop_runs":       7.0,
    "prop_rbi":        7.0,
    "prop_tb":         6.0,
    "prop_hits":       6.0,
    "prop_pitcher_er": 6.0,
    "prop_pitcher_bb": 5.0,
    "prop_pitcher_hr": 5.0,
    # pitcher_k is our highest-volume AND worst-performing prop market: in the
    # bet log it bled -17% overall and stayed -32% even after market-blending.
    # It previously had NO floor (fell through to the 3% slider). A 6% post-
    # blend floor limits it to the strongest disagreements only.
    "prop_pitcher_k":  6.0,
}


def _effective_edge_threshold_pct(market: str, slider_pct: float) -> float:
    """Return the larger of the user's slider value and the market floor."""
    return max(slider_pct, _MARKET_MIN_EDGE_PCT.get(market, 0.0))


# ---------- Main entry point ----------
def predict_slate(target_date: date | str | None = None,
                  edge_threshold: float = 0.03,
                  fetch_odds: bool = True,
                  top_n: int = 30) -> SlateResult:
    """Predict the slate for `target_date` (default = today UTC).

    Returns a SlateResult that's safe to pass to UI code or serialize to JSON.
    """
    if target_date is None:
        target = datetime.now(timezone.utc).date()
    elif isinstance(target_date, str):
        target = datetime.fromisoformat(target_date).date()
    else:
        target = target_date

    # Statcast — fetched after snap so we can use bat_stats for the player→team map
    # (deferred below, after snap is loaded)

    # Load model + season-stat snapshot
    snap = json.loads((ROOT / "data" / "games" / "snapshot_2026.json").read_text(encoding="utf-8"))
    team_off = _stats_lookup(snap["team_off"])
    team_pit = _stats_lookup(snap["team_pit"])
    pitcher_stats = _stats_lookup(snap["pitcher_stats"])
    # Reliever-only team aggregates for true bullpen ERA/FIP (vs the team_pit
    # staff-wide aggregate dominated by starter innings).
    bullpen_stats = feats.bullpen_stats_by_team(pitcher_stats)
    # Pitcher last-appearance map for days-rest computation
    try:
        _box_df = pd.read_csv(ROOT / "data" / "games" / "box_2026.csv")
        pitcher_last_appearance = feats.build_pitcher_last_appearance(_box_df)
        bullpen_usage = feats.BullpenUsageLookup(_box_df)
        # Catcher framing proxy (k-rate delta from box-score history,
        # EB-shrunk to 0 at PA=300 BF).
        catcher_framing = feats.build_catcher_framing_proxy(_box_df, pitcher_stats)
    except Exception:
        pitcher_last_appearance = {}
        bullpen_usage = None
        catcher_framing = {}
    batter_stats = _stats_lookup(snap["batter_stats"])
    bat_recent = _stats_lookup(snap.get("batter_stats_recent", {}))
    pit_recent = _stats_lookup(snap.get("pitcher_stats_recent", {}))
    # Reliever-only RECENT (14d) ERA per team — captures short-term bullpen form
    bullpen_era_recent = feats.bullpen_recent_era_by_team(pit_recent, pitcher_stats)
    bat_vs_l = _stats_lookup(snap.get("bat_vs_l", {}))
    bat_vs_r = _stats_lookup(snap.get("bat_vs_r", {}))
    pit_vs_l = _stats_lookup(snap.get("pit_vs_l", {}))   # pitcher splits vs LHB
    pit_vs_r = _stats_lookup(snap.get("pit_vs_r", {}))   # pitcher splits vs RHB
    bat_sides = {int(k): v for k, v in snap.get("bat_sides", {}).items()}
    pit_throws = {int(k): v for k, v in snap.get("pit_throws", {}).items()}

    # Statcast: build player→team map from MLB API bat_stats, then fetch
    try:
        _player_team_map = {pid: int(s["team_id"])
                            for pid, s in batter_stats.items() if s.get("team_id")}
        sc_team_bat = sc.get_team_batting(2026, _player_team_map)
        sc_pit_data = sc.get_pitcher_stats(2026)
        sc_bat_data = sc.get_batter_stats(2026)
        sc_team_def = sc.get_team_fielding(2026, _player_team_map)
        sc_team_run = sc.get_team_baserunning(2026, _player_team_map)
        sc_arsenal  = sc.get_pitcher_arsenal(2026)
    except Exception:
        sc_team_bat, sc_pit_data, sc_bat_data, sc_team_def, sc_team_run, sc_arsenal = {}, {}, {}, {}, {}, {}

    model = mdl.TeamScoreModel.load(ROOT / "data" / "models" / "team_runs.joblib")
    _model_dir = ROOT / "data" / "models"
    _boot = mdl.load_bootstrap_ensemble(_model_dir)
    _temporal = mdl.load_temporal_ensemble(_model_dir)
    _all_models = [model] + _boot + _temporal   # main + bootstrap + temporal
    _ump_rates = ump.load_rates()

    # Pull schedule
    games_raw = mlb_api.schedule(target)
    if not games_raw:
        return SlateResult(target.isoformat(), "none", 0, 0, 0, [], [])

    # Live odds (skip cleanly if unavailable)
    book_data: list[dict] = []
    live_props: list[dict] = []
    odds_source = "none"
    if fetch_odds:
        try:
            book_data, live_props, odds_source = odds.load_lines_with_fallback()
            if book_data:
                odds.snapshot_odds(book_data, live_props)
        except Exception:
            book_data, live_props, odds_source = [], [], "error"
    manual = odds.load_manual()
    merged_props = list(live_props)
    if manual.get("player_props"):
        merged_props.extend(manual["player_props"])

    all_value_bets: list[value.ValueBet] = []
    games_out: list[GamePrediction] = []
    _closing_rows: list[dict] = []   # accumulate market lines for permanent log
    _closing_prop_rows: list[dict] = []   # accumulate prop lines for prop CLV
    # Sharp-strategy audit counters (reported in SlateResult.sharp_summary)
    _sharp_games_checked = 0
    _sharp_best_ev: Optional[float] = None
    _n_sharp_bets = 0

    for g in games_raw:
        st = (g.get("status") or {}).get("codedGameState", "")
        if st in ("D", "C"):
            continue

        # Pull confirmed lineups first so they feed into the team-runs features.
        lineups = mlb_api.extract_lineups(g)
        home_lineup_ids = lineups["home"] or None
        away_lineup_ids = lineups["away"] or None
        # Starting catchers (for framing proxy lookup)
        _catchers = mlb_api.extract_starting_catchers(g)

        f = feats.build_game_features(g, team_off, team_pit, pitcher_stats,
                                      sc_team_bat=sc_team_bat, sc_pit=sc_pit_data,
                                      sc_team_def=sc_team_def,
                                      sc_team_run=sc_team_run,
                                      home_lineup_ids=home_lineup_ids,
                                      away_lineup_ids=away_lineup_ids,
                                      batter_stats=batter_stats,
                                      bat_vs_l=bat_vs_l, bat_vs_r=bat_vs_r,
                                      bat_sides=bat_sides, pit_throws=pit_throws,
                                      batter_recent=bat_recent,
                                      bullpen_stats=bullpen_stats,
                                      pitcher_last_appearance=pitcher_last_appearance,
                                      catcher_framing=catcher_framing,
                                      home_catcher_id=_catchers["home"],
                                      away_catcher_id=_catchers["away"],
                                      pit_recent=pit_recent,
                                      bullpen_era_recent=bullpen_era_recent,
                                      bullpen_usage=bullpen_usage)
        if f is None:
            continue

        # Umpire: try game feed (2-hr cache) for today's games; fall back to 1.0
        try:
            _feed = mlb_api._get(f"/v1.1/game/{f.game_pk}/feed/live", ttl_seconds=7200)
            _hp = ump.get_hp_umpire_from_game_feed(_feed)
            f.ump_k_mult = ump.get_k_mult(_hp, _ump_rates)
        except Exception:
            f.ump_k_mult = 1.0

        long = mdl.long_form(pd.DataFrame([asdict(f)]))
        long["pred_runs"] = mdl.predict_ensemble(_all_models, long)
        # League-level rate-factor correction. Captures recent run-environment
        # shifts (cool offensive weeks, league-wide BABIP regression) that the
        # static feature set can't see. Persisted in team_runs_rate.json by
        # scripts/fit_team_runs_rate.py.
        _trrf = _team_runs_rate_factor()
        if _trrf != 1.0:
            long["pred_runs"] = long["pred_runs"] * _trrf
        home_pred = float(long[long["is_home"] == 1].iloc[0]["pred_runs"])
        away_pred = float(long[long["is_home"] == 0].iloc[0]["pred_runs"])

        park = parks.get_park(f.venue)
        utc_dt = mlb_api.parse_game_time(g)

        # Book lookup (first_pitch disambiguates doubleheader twin games).
        # `bk` is reused for game-line pricing and the sportsbook card below.
        bk = _find_book(book_data, f.home_team, f.away_team,
                        first_pitch=utc_dt) if book_data else None
        if bk is None and manual:
            bk = _find_book(manual.get("games", []), f.home_team, f.away_team,
                            first_pitch=utc_dt)
        # Display lambdas: blended toward the market-implied runs so the game
        # card shows our best point estimate. Pricing below uses the RAW
        # lambdas + probability-space blending instead (see MARKET_BLEND_WEIGHT).
        home_pred_g, away_pred_g, market_total = (home_pred, away_pred, None)
        if bk is not None:
            home_pred_g, away_pred_g, market_total = value.blend_to_market(
                home_pred, away_pred, bk, MARKET_BLEND_WEIGHT)
            # Capture the market line into the permanent game-linked log.
            _ml = bk.get("moneyline") or {}
            _tot = bk.get("total") or {}
            _rl = bk.get("run_line") or {}
            _sh = bk.get("sharp") or {}
            _closing_rows.append({
                "game_pk": int(f.game_pk), "date": f.date,
                "away_team": f.away_team, "home_team": f.home_team,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "market_total": _tot.get("line"),
                "ml_home": _ml.get("home"), "ml_away": _ml.get("away"),
                "rl_line": _rl.get("line"), "rl_home": _rl.get("home"),
                "rl_away": _rl.get("away"),
                # Polymarket sharp no-vig reference (the gold-standard CLV baseline)
                "sharp_p_home": _sh.get("ml_home"), "sharp_total": _sh.get("total_line"),
                "sharp_p_over": _sh.get("p_over"),
            })

        # Displayed win probability: same quantity the moneyline pricing uses —
        # calibrated raw-model prob, blended toward the book's no-vig prob.
        p_home_win = value.calibrate_winprob(
            value.home_win_prob(home_pred, away_pred), "moneyline")
        _ml_disp = (bk or {}).get("moneyline") or {}
        if "home" in _ml_disp and "away" in _ml_disp:
            try:
                _nv_h, _ = value.devig_two_way(
                    value.american_to_prob(int(_ml_disp["home"])),
                    value.american_to_prob(int(_ml_disp["away"])))
                p_home_win = ((1 - MARKET_BLEND_WEIGHT) * p_home_win
                              + MARKET_BLEND_WEIGHT * _nv_h)
            except (TypeError, ValueError):
                pass
        p_over_8_5 = value.total_over_prob(home_pred_g, away_pred_g, 8.5)
        total_pred = home_pred_g + away_pred_g
        total_pred_raw = home_pred + away_pred

        # Starters confirmed: a multi-source check.
        # 1. MLB API claims both starters (rejects "TBD"/"?"/empty names).
        # 2. When a sportsbook is loaded, the book ALSO needs to price the
        #    matchup (pitcher_k markets exist for both probable starters
        #    OR the moneyline is up). Books are conservative — they don't
        #    price what isn't yet finalised, so this catches "MLB has a
        #    probable but the team hasn't formally announced" cases.
        def _is_real_starter(sp_id, sp_name) -> bool:
            if not sp_id:
                return False
            n = (sp_name or "").strip().lower()
            return n not in ("", "?", "tbd", "tba", "unknown", "to be announced")

        _mlb_confirmed = (_is_real_starter(f.home_sp_id, f.home_sp_name)
                          and _is_real_starter(f.away_sp_id, f.away_sp_name))

        # Book cross-check: if any props were loaded, look for pitcher_k
        # markets matching either starter. We don't require BOTH (Bovada
        # often misses the lower-K-rate guy in mismatch games), only that
        # at least one is priced. Empty merged_props means we ran without
        # odds — fall back to MLB-only signal.
        _book_confirmed = True
        if merged_props:
            _hp = (f.home_sp_name or "").lower()
            _ap = (f.away_sp_name or "").lower()
            _priced = {p.get("player", "").lower() for p in merged_props
                       if p.get("market") == "pitcher_k"}
            _book_confirmed = any(
                (sp_name and any(sp_name in pn or pn in sp_name for pn in _priced))
                for sp_name in (_hp, _ap) if sp_name
            )

        starters_confirmed = _mlb_confirmed and _book_confirmed

        gp = GamePrediction(
            game_pk=int(f.game_pk), date=f.date, venue=f.venue,
            park_roof=park.roof, park_pf_runs=park.pf_runs, park_pf_hr=park.pf_hr,
            home_team=f.home_team, away_team=f.away_team,
            home_team_id=f.home_team_id, away_team_id=f.away_team_id,
            first_pitch_utc=utc_dt.isoformat(),

            home_sp_id=f.home_sp_id, home_sp_name=f.home_sp_name,
            home_sp_fip=f.home_sp_fip, home_sp_xfip=f.home_sp_xfip,
            away_sp_id=f.away_sp_id, away_sp_name=f.away_sp_name,
            away_sp_fip=f.away_sp_fip, away_sp_xfip=f.away_sp_xfip,
            starters_confirmed=starters_confirmed,

            temp_f=f.temp_f, wind_to_cf_mph=f.wind_to_cf_mph,
            runs_mult=f.runs_mult, hr_mult=f.hr_mult,

            # Display the market-blended game-line predictions so the card is
            # internally consistent (pred_home + pred_away == pred_total).
            pred_home_runs=home_pred_g, pred_away_runs=away_pred_g,
            pred_total=total_pred, p_home_win=p_home_win, p_over_8_5=p_over_8_5,
        )

        # Build player projections
        for side, tid, otid, sp_id_self, sp_id_opp, team_pred, opp_pred, lineup_ids, opp_lineup_ids in [
            ("away", f.away_team_id, f.home_team_id, f.away_sp_id, f.home_sp_id, away_pred, home_pred, away_lineup_ids, home_lineup_ids),
            ("home", f.home_team_id, f.away_team_id, f.home_sp_id, f.away_sp_id, home_pred, away_pred, home_lineup_ids, away_lineup_ids),
        ]:
            opp_sp_q = (
                feats.pitcher_quality_index(pitcher_stats.get(sp_id_opp, {}),
                                            sc_stats=sc_pit_data.get(sp_id_opp))
                if sp_id_opp else feats.pitcher_quality_index({})
            )
            opp_off_idx = feats.team_offense_index(team_off.get(otid, {}))
            wadj = {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult,
                    "wind_to_cf_mph": f.wind_to_cf_mph, "temp_f": f.temp_f}

            # Lineup-specific K% for pitcher K projection.
            # When the opposing lineup is confirmed, blend 65% lineup K% with
            # 35% team season K% — lineup K% better reflects today's actual
            # batters (e.g. rest days, bench players). Falls back to team K%
            # when lineup isn't posted yet.
            if opp_lineup_ids:
                lk = proj.lineup_k_pct(opp_lineup_ids, batter_stats)
                team_k = opp_off_idx.get("k_pct", 0.225)
                opp_off_idx = dict(opp_off_idx)
                opp_off_idx["k_pct"] = 0.65 * lk + 0.35 * team_k

            batters_out = []
            order = 1
            for bs in proj.get_likely_batters(tid, batter_stats, lineup_ids=lineup_ids):
                pid = int(bs.get("player_id") or 0)
                pl = proj.resolve_platoon(pid, sp_id_opp, bat_sides, pit_throws,
                                          bat_vs_l, bat_vs_r)
                rs = bat_recent.get(pid)
                p = proj.project_batter(bs, order, team_pred, opp_sp_q, park, wadj,
                                        recent_stats=rs,
                                        bat_side=pl["bat_side"],
                                        opp_pit_throws=pl["opp_pit_throws"],
                                        bat_split=pl["bat_split"],
                                        is_switch=pl.get("is_switch", False),
                                        sc_stats=sc_bat_data.get(pid),
                                        game_ctx={
                                            "ump_k_mult": f.ump_k_mult,
                                            "opp_catcher_framing": getattr(
                                                f, ("away" if side == "home" else "home")
                                                + "_catcher_framing", None),
                                        })
                batters_out.append(asdict(p))
                order += 1

            if side == "away":
                gp.away_batters = batters_out
            else:
                gp.home_batters = batters_out

            if sp_id_self:
                ps = pitcher_stats.get(sp_id_self, {"player_id": sp_id_self, "name": "?", "team_id": tid})
                ps_recent = pit_recent.get(int(sp_id_self))
                pp = proj.project_pitcher(ps, tid, opp_off_idx, opp_pred, park, wadj,
                                          recent_stats=ps_recent,
                                          sc_stats=sc_pit_data.get(int(sp_id_self)),
                                          game_ctx={
                                              "ump_k_mult": f.ump_k_mult,
                                              "own_catcher_framing": getattr(f, side + "_catcher_framing", None),
                                              "sp_days_rest": getattr(f, side + "_sp_days_rest", None),
                                              "bp_ip_72h": getattr(f, side + "_bp_ip_72h", None),
                                              "bp_top_rest": getattr(f, side + "_bp_top_rest", None),
                                          })
                if side == "away":
                    gp.away_starter = asdict(pp)
                else:
                    gp.home_starter = asdict(pp)

        # Sportsbook lookup — `bk` was resolved earlier for the market blend.
        if bk is not None:
            gp.book = bk
            gp.book_source = odds_source if (book_data and bk in book_data) else "manual"
            # Price game lines from the RAW run predictions; calibration and
            # the market blend happen in probability space inside
            # evaluate_game_lines (calibrate first, then blend — avoids the
            # old double-shrink of calibrating already-blended probabilities).
            game_value_all = value.evaluate_game_lines(
                f.home_team, f.away_team, home_pred, away_pred, bk,
                edge_threshold=-1.0,
                market_blend=MARKET_BLEND_WEIGHT,
            )
            for vb in game_value_all:
                vb.game_pk = int(f.game_pk)
                vb.starters_confirmed = starters_confirmed
            game_value = [vb for vb in game_value_all if vb.edge_pct >= edge_threshold * 100]

            # Market disagreement guard: if the RAW model total differed from
            # the market total by more than MARKET_MAX_TOTAL_DISAGREE runs, the
            # disagreement is model error rather than edge — suppress all
            # game-line bets for this game.
            if (market_total is not None
                    and abs(total_pred_raw - market_total) > MARKET_MAX_TOTAL_DISAGREE):
                game_value = []

            # NOTE (Jun 9 2026 policy prune): the "elite ace Over" filter and
            # the "total minimum 1.0-run gap" filter were removed. Both were
            # fit to 1-2 anecdotes; the calibrated total probabilities + the
            # market blend + the direction-consistency check below are the
            # principled versions of what they were patching.

            # Total bet direction consistency: the model prediction must agree
            # with the bet direction. An Under bet when model pred > line means
            # we're betting against our own model — NegBin tail probabilities
            # can manufacture apparent edge even when the mean disagrees.
            # Bet log: direction-consistent totals 9W 2L (82%), inconsistent
            # 4W 6L (40%). TX@DET Under 8.0 (pred=9.0) and CWS@SD Under 8.0
            # (pred=9.2) both passed the gap filter (gap=-1.0/-1.2 > -1.0 threshold
            # but directionally wrong) and both lost.
            game_value = [vb for vb in game_value
                          if not (vb.market == "total"
                                  and ((" Over " in vb.description and total_pred < vb.line)
                                       or (" Under " in vb.description and total_pred > vb.line)))]

            # Run-line plus-money restriction: REMOVED Jun 10 2026 after the
            # regrade. The rule was built on the corrupted log (away-RL sign
            # bug): corrected outcomes show plus-money RL 4W-9L (-24%) and
            # minus-money 20W-13L (-1.4%) — the rule kept exactly the WORSE
            # side. Corrected RL overall is ~vig-level noise (n=46), so no
            # directional filter is justified; the calibrate-then-blend
            # pricing and the sharp veto govern run-line bets now.

            gp.all_bets.extend(_vb_to_dict(vb) for vb in game_value_all)

            # Sharp-value bets: where the bettable book's price beats the
            # Polymarket near-vigless true line (model-free advantage betting).
            # Evaluated with no floor first so the slate can report HOW CLOSE
            # the book came to +EV even on days nothing fires (the audit
            # answer to "is the sharp detector broken or is Fanatics just
            # efficient?" — measured Jun 10 2026: zero fires in a week, best
            # side -0.2% EV, mean ~-4.4%).
            sharp_candidates = value.evaluate_sharp_value(
                f.home_team, f.away_team, bk, min_ev=-1.0)
            sharp_bets = [vb for vb in sharp_candidates
                          if vb.ev_per_dollar >= SHARP_MIN_EV]
            for vb in sharp_bets:
                vb.game_pk = int(f.game_pk)
                vb.starters_confirmed = starters_confirmed
            if sharp_candidates:
                _sharp_games_checked += 1
                _best = max(vb.ev_per_dollar for vb in sharp_candidates)
                _sharp_best_ev = (_best if _sharp_best_ev is None
                                  else max(_sharp_best_ev, _best))
                _n_sharp_bets += len(sharp_bets)

            # When a sharp (Polymarket) reference exists, TRUST IT over the
            # model for game lines: the model cannot beat an efficient sharp
            # line (measured: best Fanatics ML is ~-2% EV vs the sharp, mean
            # -4.4%), so its "edges" there are illusory. Bet only sharp-
            # validated +EV. Fall back to the model's game-line bets only when
            # no sharp reference is available (e.g. Bovada-only slate).
            has_sharp = (bk.get("sharp") or {}).get("ml_home") is not None
            leaderboard_game = sharp_bets if has_sharp else game_value
            gp.game_value = [_vb_to_dict(vb) for vb in leaderboard_game]
            gp.all_bets.extend(_vb_to_dict(vb) for vb in sharp_bets)
            all_value_bets.extend(leaderboard_game)

        # Player props for this game
        if merged_props:
            by_name = {}
            for _side_batters in (
                proj.get_likely_batters(f.home_team_id, batter_stats, lineup_ids=home_lineup_ids),
                proj.get_likely_batters(f.away_team_id, batter_stats, lineup_ids=away_lineup_ids),
            ):
                for bs in _side_batters:
                    by_name[bs.get("name", "")] = bs
            for sp_id in (f.home_sp_id, f.away_sp_id):
                if sp_id and pitcher_stats.get(sp_id):
                    by_name[pitcher_stats[sp_id].get("name", "")] = pitcher_stats[sp_id]

            our_props = [pp for pp in merged_props
                         if "game" not in pp
                         or (_team_name_match(pp["game"].split(" @ ")[0], f.away_team)
                             and _team_name_match(pp["game"].split(" @ ")[1], f.home_team))]

            game_prop_value = []
            for pp in our_props:
                name = pp.get("player", "")
                resolved = name_match.find_match(name, by_name.keys())
                pdata = by_name.get(resolved) if resolved else None
                if not pdata:
                    continue
                is_pitcher = pp["market"].startswith("pitcher_") or pp["market"] in ("outs", "ip")
                if is_pitcher:
                    is_home_pitcher = (pdata.get("player_id") == f.home_sp_id)
                    team_id = f.home_team_id if is_home_pitcher else f.away_team_id
                    opp_id = f.away_team_id if is_home_pitcher else f.home_team_id
                    opp_off = feats.team_offense_index(team_off.get(opp_id, {}))
                    # Use the confirmed opposing lineup's K% directly when
                    # available. lineup_k_pct already EB-shrinks each batter to
                    # league at PRIOR_PA=30, so a second shrinkage to team K%
                    # was double-dipping. Team K% additionally biases toward
                    # the team's bench/IL pool, which is the wrong prior when
                    # we know exactly who's hitting.
                    _opp_lineup = away_lineup_ids if is_home_pitcher else home_lineup_ids
                    if _opp_lineup:
                        opp_off = dict(opp_off)
                        opp_off["k_pct"] = proj.lineup_k_pct(_opp_lineup, batter_stats)
                    opp_pred = away_pred if is_home_pitcher else home_pred
                    _pid = int(pdata.get("player_id") or 0)
                    # Lineup-mix-weighted pitcher platoon split. Captures
                    # reverse-platoon arms and lineup-composition matchup edges.
                    _pthrows = (pit_throws.get(_pid) or "R").upper()
                    _split_stats = lf.pitcher_split_vs_lineup(
                        _pid, _opp_lineup or [], bat_sides, _pthrows,
                        pit_vs_l, pit_vs_r,
                    ) if _opp_lineup else None
                    _own = "home" if is_home_pitcher else "away"
                    pproj = proj.project_pitcher(pdata, team_id, opp_off, opp_pred, park,
                                                 {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult},
                                                 recent_stats=pit_recent.get(_pid),
                                                 sc_stats=sc_pit_data.get(_pid),
                                                 split_stats=_split_stats,
                                                 arsenal=sc_arsenal.get(_pid),
                                                 game_ctx={
                                                     "ump_k_mult": f.ump_k_mult,
                                                     "own_catcher_framing": getattr(f, _own + "_catcher_framing", None),
                                                     "sp_days_rest": getattr(f, _own + "_sp_days_rest", None),
                                                     "bp_ip_72h": getattr(f, _own + "_bp_ip_72h", None),
                                                     "bp_top_rest": getattr(f, _own + "_bp_top_rest", None),
                                                 })
                    means = {
                        "pitcher_k": pproj.proj_k, "pitcher_outs": pproj.expected_outs,
                        "pitcher_er": pproj.proj_er, "pitcher_h": pproj.proj_h,
                        "pitcher_bb": pproj.proj_bb, "pitcher_hr": pproj.proj_hr_allowed,
                    }
                    mean = means.get(pp["market"])
                else:
                    is_home_batter = (pdata.get("team_id") == f.home_team_id)
                    sp_id = f.away_sp_id if is_home_batter else f.home_sp_id
                    opp_sp_q = (
                        feats.pitcher_quality_index(pitcher_stats.get(sp_id, {}),
                                                    sc_stats=sc_pit_data.get(sp_id))
                        if sp_id else feats.pitcher_quality_index({})
                    )
                    team_pred_local = home_pred if is_home_batter else away_pred
                    pid = int(pdata.get("player_id") or 0)
                    pl = proj.resolve_platoon(pid, sp_id, bat_sides, pit_throws, bat_vs_l, bat_vs_r)
                    rs = bat_recent.get(pid)
                    bproj = proj.project_batter(pdata, 3, team_pred_local, opp_sp_q, park,
                                                {"runs_mult": f.runs_mult, "hr_mult": f.hr_mult,
                                                 "wind_to_cf_mph": f.wind_to_cf_mph, "temp_f": f.temp_f},
                                                recent_stats=rs,
                                                bat_side=pl["bat_side"],
                                                opp_pit_throws=pl["opp_pit_throws"],
                                                bat_split=pl["bat_split"],
                                                is_switch=pl.get("is_switch", False),
                                                sc_stats=sc_bat_data.get(pid),
                                                game_ctx={
                                                    "ump_k_mult": f.ump_k_mult,
                                                    "opp_catcher_framing": getattr(
                                                        f, ("away" if is_home_batter else "home")
                                                        + "_catcher_framing", None),
                                                })
                    means = {
                        "hr": bproj.proj_hr, "hits": bproj.proj_h, "tb": bproj.proj_tb,
                        "rbi": bproj.proj_rbi, "runs": bproj.proj_runs, "k": bproj.proj_k,
                        "bb": bproj.proj_bb, "sb": bproj.proj_sb,
                    }
                    mean = means.get(pp["market"])
                if mean is None:
                    continue
                # Capture the prop offer into the permanent closing-props log
                # (latest capture per player/market wins ≈ the closing line).
                _closing_prop_rows.append({
                    "game_pk": int(f.game_pk), "date": f.date,
                    "player": name,
                    "player_id": int(pdata.get("player_id") or 0),
                    "market": pp["market"], "line": pp["line"],
                    "over": pp.get("over"), "under": pp.get("under"),
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                })
                vbs_all = value.evaluate_prop(name, pp["market"], mean, pp["line"],
                                              pp.get("over"), pp.get("under"),
                                              edge_threshold=-1.0)
                _pid = int(pdata.get("player_id") or 0)
                # Short-start guard: drop pitcher counting-stat OVERs when we
                # project < 4.5 IP (13.5 outs). Quick-hook starts — rookies on
                # pitch counts, openers, struggling vets — can collapse an
                # OVER pick to zero. The projection's NegBin distribution
                # doesn't capture this discrete-event risk well. May 2 example:
                # Lowder was a top-confidence OVER 4.5 K pick that went 1 K on
                # a quick pull. UNDERs are unaffected (a short start helps them).
                if is_pitcher and vbs_all and pproj.expected_outs < 14.5:
                    vbs_all = [vb for vb in vbs_all if " OVER " not in vb.description]
                # Pitcher K guards. Jun 9 2026 policy prune: only the rules
                # with a structural mechanism survive. Removed: the edge<5%
                # UNDER filter, the line<=2.5 and line<=4.5 low-line UNDER
                # guards, and the model_prob>=0.60 floor — all were tuned on
                # single-digit samples from the (then mis-graded) bet log.
                # The per-market 6% edge floor in _MARKET_MIN_EDGE_PCT and the
                # now-coherent probability calibration are the principled
                # replacements.
                if (is_pitcher and pp["market"] == "pitcher_k" and vbs_all):
                    # Elite-arm / high-line UNDER guard. Mechanism: the prop
                    # projections are compressed toward the mean (documented in
                    # prop_models.py), so we systematically under-project elite
                    # strikeout arms, lean UNDER on their high lines, and lose.
                    _whiff = (sc_pit_data.get(_pid) or {}).get("whiff_pct") or 0.0
                    _kpct  = (sc_pit_data.get(_pid) or {}).get("k_pct")    or 0.0
                    _high_line  = pp["line"] >= 6.0
                    _elite_arm  = _whiff >= 27.0 or _kpct >= 25.0
                    if _high_line or _elite_arm:
                        vbs_all = [vb for vb in vbs_all if " UNDER " not in vb.description]
                    # K OVER blanket block. Mechanism: an OVER needs the starter
                    # to beat a K line set by people who know the pitch-count
                    # plan, matchup, and weather; our compression-corrected
                    # projections overshoot exactly there (-76% ROI, no guard
                    # level rescued it).
                    vbs_all = [vb for vb in vbs_all if " OVER " not in vb.description]
                    # K prop edge cap at 15%. Mechanism: adverse selection — a
                    # huge model-vs-book gap on a K line almost always means the
                    # book knows something we don't (injury, planned short
                    # start, pitch count).
                    vbs_all = [vb for vb in vbs_all if vb.edge_pct <= 15.0]
                # Pitcher OUTS workhorse UNDER guard (mirror to short-start):
                # 21-day backtest top-decile pitcher outs under-projects by
                # 1.18 outs (proj 17.0, actual 18.2). Top arms exceed our
                # outs projection, so UNDER picks at high lines lose the same
                # way pitcher K UNDERs lose for elite arms. Drop pitcher_outs
                # UNDERs when projected outs >= 16.5 (≈ 5.5 IP) OR when the
                # book line is >= 17.5.
                if (is_pitcher and pp["market"] == "pitcher_outs" and vbs_all):
                    _high_outs_line = pp["line"] >= 17.5
                    _workhorse = pproj.expected_outs >= 16.5
                    if _high_outs_line or _workhorse:
                        vbs_all = [vb for vb in vbs_all if " UNDER " not in vb.description]
                # (Jun 9 2026 policy prune: the runs-OVER lineup-spot filter was
                # removed — n=2 anecdotes, and lineup order is already an input
                # to the runs projection itself.)
                # Batter TB UNDER guard: top-decile TB projections under-shoot
                # by 0.67 TB (proj 1.84, actual 2.51) — elite power hitters
                # drive way more bases than projected. Drop TB UNDERs when
                # the batter's Statcast barrel% is elite (>= 12), parallel to
                # the elite-K pitcher UNDER guard.
                if (not is_pitcher and pp["market"] == "tb" and vbs_all):
                    _barrel = (sc_bat_data.get(_pid) or {}).get("barrel_pct") or 0.0
                    _xwoba  = (sc_bat_data.get(_pid) or {}).get("xwoba")      or 0.0
                    if _barrel >= 12.0 or _xwoba >= 0.380:
                        vbs_all = [vb for vb in vbs_all if " UNDER " not in vb.description]
                if vbs_all:
                    for vb in vbs_all:
                        vb.game_pk = int(f.game_pk)
                        vb.player_id = _pid
                        vb.starters_confirmed = starters_confirmed
                    gp.all_bets.extend(_vb_to_dict(vb) for vb in vbs_all)
                    # Apply per-market edge floor: noisy markets (hits/RBI/runs/
                    # TB/pitcher_er, all R^2 < 0.06) require a bigger edge to
                    # be flagged. Reliable markets fall through to the slider.
                    vbs = [vb for vb in vbs_all
                           if vb.edge_pct >= _effective_edge_threshold_pct(
                               vb.market, edge_threshold * 100)]
                    game_prop_value.extend(vbs)
                    all_value_bets.extend(vbs)
            gp.prop_value = [_vb_to_dict(vb) for vb in game_prop_value]

        games_out.append(gp)

    # Persist captured market lines (game-linked, permanent) for future
    # market-blend backtesting and line-value features.
    _persist_closing_lines(_closing_rows)
    _persist_closing_props(_closing_prop_rows)

    # Build slate-wide leaderboard. We rank by `score` (variance-adjusted edge)
    # rather than raw edge — this puts reliable, near-50/50 plays ahead of
    # high-variance lottery tickets with the same nominal edge.
    ranked = sorted(all_value_bets, key=lambda x: -getattr(x, "score", x.edge_pct))[:top_n]
    top_value = [_vb_to_dict(vb) for vb in ranked]

    # Concentration check: if a single game's bets occupy > 40% of the top
    # leaderboard, warn that it's effectively one bet.
    concentration_warning = None
    if ranked:
        # Tag each bet with the game it came from (best-effort match by team
        # names appearing in the description)
        team_names = []
        for gp in games_out:
            team_names.append((gp.home_team, gp.away_team, gp.game_pk))
        from collections import Counter
        bet_games = []
        for vb in ranked:
            desc_l = vb.description.lower()
            for h, a, pk in team_names:
                if h.lower() in desc_l or a.lower() in desc_l:
                    bet_games.append(pk)
                    break
        if bet_games:
            cnt = Counter(bet_games)
            top_pk, top_n_bets = cnt.most_common(1)[0]
            share = top_n_bets / len(ranked)
            if share >= 0.40:
                # Look up team names for the offending game
                gp = next((x for x in games_out if x.game_pk == top_pk), None)
                if gp:
                    concentration_warning = (
                        f"{top_n_bets} of the top {len(ranked)} value bets "
                        f"({share:.0%}) come from {gp.away_team} @ {gp.home_team}. "
                        f"They share one underlying signal — treat as ~1 position, not {top_n_bets}."
                    )

    # Log the top of the DISPLAYED (score-ranked) leaderboard for outcome
    # tracking — the log must record the same picks the user sees, or every
    # constant later tuned against it inherits a selection bias.
    try:
        if top_value and target == datetime.now(timezone.utc).date():
            bet_tracker.log_picks(target, top_value, top_n=10,
                                  policy_version=POLICY_VERSION)
            # Shadow-log EVERY floor-clearing bet on the slate (not just the
            # top-10). Shadow picks are excluded from the headline record but
            # graded and CLV-tracked identically — they exist to shrink the
            # confidence intervals on per-market records and CLV ~5-10x
            # faster than 10 picks/day ever could.
            _shadow_pool = sorted(all_value_bets,
                                  key=lambda x: -getattr(x, "score", x.edge_pct))
            bet_tracker.log_picks(target, [_vb_to_dict(vb) for vb in _shadow_pool],
                                  top_n=len(_shadow_pool),
                                  policy_version=POLICY_VERSION, shadow=True)
            bet_tracker.evaluate_outcomes()
    except Exception:
        pass

    slate_all_bets: list[dict] = []
    for gp in games_out:
        slate_all_bets.extend(gp.all_bets)

    # Model-CI leaderboard + skill-backed parlays, from the full alt-line pool.
    model_ci_bets = parlays.model_confident_bets(slate_all_bets, MODEL_CI_THRESHOLD)
    parlay_list = parlays.build_parlays(slate_all_bets)
    # HR-specific 2- and 3-leg parlays — one player per game, ranked purely
    # by the model's HR conviction (no min_conf floor; HR overs are
    # inherently ~10-15% per game so a threshold isn't useful).
    hr_parlay_list = parlays.build_parlays(
        slate_all_bets, sizes=(2, 3),
        skill_markets=("prop_hr",), min_conf=0.0,
        sides=("OVER",), name_prefix="HR")

    return SlateResult(
        target_date=target.isoformat(),
        odds_source=odds_source,
        n_games=len(games_out),
        n_books=len(book_data),
        n_props_loaded=len(merged_props),
        games=games_out,
        top_value=top_value,
        concentration_warning=concentration_warning,
        all_bets=slate_all_bets,
        sharp_summary={"games_checked": _sharp_games_checked,
                       "n_bets": _n_sharp_bets,
                       "best_ev": _sharp_best_ev},
        model_ci_bets=model_ci_bets,
        parlays=parlay_list,
        hr_parlays=hr_parlay_list,
    )
