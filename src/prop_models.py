"""Per-stat ML projection models for player props.

We train one HistGradientBoostingRegressor per stat. Each model takes:
  - the analytical projection from `projections.project_batter`/`project_pitcher`
    as a feature (so it learns residuals — a stacking approach)
  - the player's season + recent (14d) rate stats
  - the opposing pitcher / team quality
  - park + weather
  - lineup context (expected PA / outs)

Loss = Poisson for all count stats. Calibration is checked against held-out
games and compared to the rule-based projection.

The point of this layer:
  - The analytical projection is well-calibrated near the mean but
    SYSTEMATICALLY compressed toward the mean at the tails (top hitters
    under-projected on HR by ~30% in the 2026 backtest).
  - A residual GBT learns "when the analytical model says X, what's the
    actual mean?" — letting it expand the tails when warranted.

If a model file is missing or fails to load, the codebase falls back to the
analytical projection (no degradation).
"""
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------- Stat configs ----------
@dataclass
class StatSpec:
    name: str               # e.g. "h", "hr", "tb"
    actual_col: str         # column in box CSV
    proj_col: str           # column on the analytical BatterProjection / PitcherProjection
    is_pitcher: bool = False
    loss: str = "poisson"   # "poisson" | "squared_error"


BATTER_STATS: list[StatSpec] = [
    StatSpec("h",     "h",       "proj_h"),
    StatSpec("hr",    "hr",      "proj_hr"),
    StatSpec("tb",    "tb",      "proj_tb"),
    StatSpec("rbi",   "rbi",     "proj_rbi"),
    StatSpec("runs",  "runs_b",  "proj_runs"),
    StatSpec("k",     "k_b",     "proj_k"),
    StatSpec("bb",    "bb_b",    "proj_bb"),
]

PITCHER_STATS: list[StatSpec] = [
    StatSpec("k",     "k_p",     "proj_k",        is_pitcher=True),
    StatSpec("outs",  "outs",    "expected_outs", is_pitcher=True),
    StatSpec("er",    "er",      "proj_er",       is_pitcher=True),
    StatSpec("h",     "h_p",     "proj_h",        is_pitcher=True),
    StatSpec("bb",    "bb_p",    "proj_bb",       is_pitcher=True),
    StatSpec("hr",    "hr_p",    "proj_hr_allowed", is_pitcher=True),
]


# ---------- Feature builders ----------
def batter_feature_row(
    bproj,                          # BatterProjection
    bat_stats: dict,
    rec_stats: dict | None,
    opp_sp_q: dict,
    park,
    weather_adj: dict,
    team_pred_runs: float,
    sc_stats: dict | None = None,   # player-level Statcast (barrel_pct, hard_hit)
) -> dict:
    """One row of features for a batter projection."""
    pa = float(bat_stats.get("plateAppearances") or 0)
    ab = float(bat_stats.get("atBats") or 0)
    h = float(bat_stats.get("hits") or 0)
    hr = float(bat_stats.get("homeRuns") or 0)
    bb = float(bat_stats.get("baseOnBalls") or 0)
    k = float(bat_stats.get("strikeOuts") or 0)
    tb = float(bat_stats.get("totalBases") or 0)
    rbi = float(bat_stats.get("rbi") or 0)
    g = float(bat_stats.get("gamesPlayed") or 1) or 1.0

    rec_pa = float((rec_stats or {}).get("plateAppearances") or 0)
    rec_ab = float((rec_stats or {}).get("atBats") or 0)
    rec_h  = float((rec_stats or {}).get("hits") or 0)
    rec_hr = float((rec_stats or {}).get("homeRuns") or 0)

    return {
        "proj_h":  float(bproj.proj_h),
        "proj_hr": float(bproj.proj_hr),
        "proj_tb": float(bproj.proj_tb),
        "proj_rbi": float(bproj.proj_rbi),
        "proj_runs": float(bproj.proj_runs),
        "proj_k":  float(bproj.proj_k),
        "proj_bb": float(bproj.proj_bb),
        "expected_pa": float(bproj.expected_pa),
        "bat_order": int(bproj.bat_order),
        "team_pred_runs": float(team_pred_runs),

        "season_pa": pa, "season_g": g,
        "season_avg":   (h / ab) if ab else 0.245,
        "season_obp":   (h + bb) / pa if pa else 0.32,
        "season_hr_pa": (hr / pa) if pa else 0.030,
        "season_iso":   ((tb - h) / ab) if ab else 0.150,
        "season_k_pa":  (k / pa) if pa else 0.225,
        "season_bb_pa": (bb / pa) if pa else 0.085,
        "season_rbi_pa": (rbi / pa) if pa else 0.115,

        "recent_pa":     rec_pa,
        "recent_avg":    (rec_h / rec_ab) if rec_ab else 0.245,
        "recent_hr_pa":  (rec_hr / rec_pa) if rec_pa else 0.030,
        "recent_form":   (rec_h / rec_ab - h / ab) if (rec_ab and ab) else 0.0,

        "opp_sp_k9":  float(opp_sp_q.get("k9", 8.7)),
        "opp_sp_bb9": float(opp_sp_q.get("bb9", 3.2)),
        "opp_sp_hr9": float(opp_sp_q.get("hr9", 1.20)),
        "opp_sp_era": float(opp_sp_q.get("era", 4.50)),
        "opp_sp_fip": float(opp_sp_q.get("fip", 4.10)),
        "opp_sp_xfip": float(opp_sp_q.get("xfip", 4.10)),

        "park_pf_runs": float(park.pf_runs),
        "park_pf_hr":   float(park.pf_hr),
        "park_elev":    float(park.elevation_ft),

        "runs_mult":   float(weather_adj.get("runs_mult", 1.0)),
        "hr_mult":     float(weather_adj.get("hr_mult", 1.0)),
        "wind_to_cf":  float(weather_adj.get("wind_to_cf_mph", 0.0)),
        "temp_f":      float(weather_adj.get("temp_f", 70.0)),

        # Statcast — contact quality (league-avg defaults if not available).
        # xwoba and xba strip BABIP/sequencing luck from raw rate stats and
        # stabilise faster, so the GBT can see the "true talent" signal even
        # when raw season AVG/SLG is still noisy in April-May. exit_velo
        # correlates with HR/TB independently of barrel rate.
        "sc_barrel_pct": float((sc_stats or {}).get("barrel_pct") or 8.0),
        "sc_hard_hit":   float((sc_stats or {}).get("hard_hit")   or 37.0),
        "sc_xwoba":      float((sc_stats or {}).get("xwoba")      or 0.315),
        "sc_xba":        float((sc_stats or {}).get("xba")        or 0.245),
        "sc_exit_velo":  float((sc_stats or {}).get("exit_velo")  or 88.5),
        # Plate discipline — addresses the per-team gap where OPS-weighted
        # features under-project contact-driven teams (Nationals, Cubs,
        # Pirates) and over-project power-driven free swingers (Tigers, Phillies).
        "sc_oz_swing":   float((sc_stats or {}).get("oz_swing")   or 32.0),   # league ~32%
        "sc_oz_contact": float((sc_stats or {}).get("oz_contact") or 58.0),   # league ~58%
        "sc_z_swing":    float((sc_stats or {}).get("z_swing")    or 68.0),   # league ~68%
        "sc_sweet_spot": float((sc_stats or {}).get("sweet_spot") or 33.0),   # league ~33%
    }


def pitcher_feature_row(
    pproj,                         # PitcherProjection
    pit_stats: dict,
    rec_stats: dict | None,
    opp_off_idx: dict,
    park,
    weather_adj: dict,
    opp_pred_runs: float,
    sc_stats: dict | None = None,  # player-level Statcast (xera, xwoba, whiff%, k%, csp, ...)
    split_stats: dict | None = None,  # lineup-mix-weighted L/R splits (k_pct, bb_pct, ops, lhb_pct)
    arsenal: dict | None = None,   # pitch-mix profile (num_pitch_types, fb_avg_velo, velo_gap)
) -> dict:
    bf = float(pit_stats.get("battersFaced") or 0)
    ip_str = pit_stats.get("inningsPitched", 0)
    # Reuse the helper from features for consistency
    from .features import _ip_to_outs
    season_outs = _ip_to_outs(ip_str)
    season_ip = season_outs / 3.0 if season_outs else 1.0

    rec_bf  = float((rec_stats or {}).get("battersFaced") or 0)
    rec_ip_str = (rec_stats or {}).get("inningsPitched", 0)
    rec_outs = _ip_to_outs(rec_ip_str)
    rec_ip   = rec_outs / 3.0 if rec_outs else 0.0
    rec_k    = float((rec_stats or {}).get("strikeOuts") or 0)
    rec_er   = float((rec_stats or {}).get("earnedRuns") or 0)

    # Pull saber metrics directly from MLB Stats API when present (parsed by
    # `mlb_api`). FIP/xFIP are far more predictive than ERA in-season, so
    # exposing them as direct features lets the GBT pick them up rather than
    # relying solely on the more contaminated `season_era`.
    season_fip = pit_stats.get("fip")
    season_fip = float(season_fip) if season_fip not in (None, "", "-.--", ".---") else 4.10
    season_xfip = pit_stats.get("xfip")
    season_xfip = float(season_xfip) if season_xfip not in (None, "", "-.--", ".---") else season_fip

    # GB tendency. Stabilises ~70 BIP. Computed from groundOuts/airOuts which
    # the MLB API returns for every pitcher. FB pitchers in HR parks get
    # systematically under-projected on HR; GB pitchers benefit in HR-suppressing
    # environments. Falls back to league-avg 0.43 GB%, 1.10 GO/AO.
    gb_outs = float(pit_stats.get("groundOuts") or 0)
    fb_outs = float(pit_stats.get("airOuts") or 0)
    season_gb_pct = (gb_outs / (gb_outs + fb_outs)) if (gb_outs + fb_outs) > 0 else 0.43
    _gtfo = pit_stats.get("groundOutsToAirouts")
    try:
        season_gb_to_fb = float(_gtfo) if _gtfo not in (None, "", "-.--", ".---") else 1.10
    except (TypeError, ValueError):
        season_gb_to_fb = 1.10

    sc = sc_stats or {}
    sp = split_stats or {}

    return {
        "proj_k":      float(pproj.proj_k),
        "proj_bb":     float(pproj.proj_bb),
        "proj_h":      float(pproj.proj_h),
        "proj_er":     float(pproj.proj_er),
        "proj_hr":     float(pproj.proj_hr_allowed),
        "expected_outs": float(pproj.expected_outs),
        "expected_ip":   float(pproj.expected_ip),
        "opp_pred_runs": float(opp_pred_runs),

        "season_bf": bf,
        "season_ip": season_ip,
        "season_k9":  (float(pit_stats.get("strikeOuts") or 0)  * 9 / season_ip) if season_ip else 8.7,
        "season_bb9": (float(pit_stats.get("baseOnBalls") or 0) * 9 / season_ip) if season_ip else 3.2,
        "season_hr9": (float(pit_stats.get("homeRuns") or 0)    * 9 / season_ip) if season_ip else 1.20,
        "season_era": (float(pit_stats.get("earnedRuns") or 0)  * 9 / season_ip) if season_ip else 4.5,
        "season_fip": season_fip,
        "season_xfip": season_xfip,
        "season_gb_pct":   season_gb_pct,
        "season_gb_to_fb": season_gb_to_fb,

        "recent_bf": rec_bf,
        "recent_k9": (rec_k * 9 / rec_ip) if rec_ip else 8.7,
        "recent_era": (rec_er * 9 / rec_ip) if rec_ip else 4.5,

        "opp_off_rpg":    float(opp_off_idx.get("rpg", 4.5)),
        "opp_off_ops":    float(opp_off_idx.get("ops", 0.720)),
        "opp_off_k_pct":  float(opp_off_idx.get("k_pct", 0.225)),
        "opp_off_bb_pct": float(opp_off_idx.get("bb_pct", 0.085)),
        "opp_off_iso":    float(opp_off_idx.get("iso", 0.150)),
        "opp_off_woba":   float(opp_off_idx.get("woba", 0.315)),

        "park_pf_runs": float(park.pf_runs),
        "park_pf_hr":   float(park.pf_hr),
        "park_elev":    float(park.elevation_ft),

        "runs_mult":   float(weather_adj.get("runs_mult", 1.0)),
        "hr_mult":     float(weather_adj.get("hr_mult", 1.0)),
        "wind_to_cf":  float(weather_adj.get("wind_to_cf_mph", 0.0)),
        "temp_f":      float(weather_adj.get("temp_f", 70.0)),

        # Statcast — contact quality + swing-and-miss profile. Pitcher prop
        # models had ZERO Statcast features previously; this is the single
        # biggest gap in the feature set. xera+xwoba capture true contact
        # quality allowed (vs. ERA/AVG which are BABIP/sequencing-contaminated).
        # whiff_pct + k_pct + csp directly anchor K projections; barrel% +
        # hard_hit drive HR/ER variance.
        "sc_xera":       float(sc.get("xera")       or 4.20),
        "sc_xwoba":      float(sc.get("xwoba")      or 0.315),
        "sc_whiff_pct":  float(sc.get("whiff_pct")  or 25.0),
        "sc_k_pct":      float(sc.get("k_pct")      or 22.5),
        "sc_csp":        float(sc.get("csp")        or 17.0),
        "sc_barrel_pct": float(sc.get("barrel_pct") or 7.8),
        "sc_hard_hit":   float(sc.get("hard_hit")   or 37.0),

        # Lineup-mix-weighted platoon splits. lhb_pct = % of opposing lineup
        # batting LH against this pitcher's hand (switch hitters bat opposite
        # the pitcher). Splits captured for K%, BB%, HR%, OPS-allowed —
        # captures reverse-platoon pitchers and lineup-composition-driven
        # matchup edges that the season aggregates miss.
        "split_lhb_pct":  float(sp.get("lhb_pct") or 0.40),
        "split_k_pct":    float(sp.get("k_pct")   or 0.225),
        "split_bb_pct":   float(sp.get("bb_pct")  or 0.085),
        "split_hr_pct":   float(sp.get("hr_pct")  or 0.030),
        "split_ops":      float(sp.get("ops")     or 0.720),

        # Pitch-mix arsenal: # of pitch types thrown >=5%, fastball velocity,
        # velocity gap (FB - changeup). 3-pitch starters get hit harder TTO+2;
        # high velo + big gap = better deception independent of K rate.
        "arsenal_num_pt":    float((arsenal or {}).get("num_pitch_types") or 4.0),
        "arsenal_fb_velo":   float((arsenal or {}).get("fb_avg_velo")     or 93.0),
        "arsenal_velo_gap":  float((arsenal or {}).get("velo_gap")        or 8.0),
        "arsenal_fb_pct":    float((arsenal or {}).get("fastball_pct")    or 55.0),
        # Spin + induced vertical break — distinguishes "good fastball" from
        # "fast fastball". High-spin risers (IVB > 17) beat the top of the
        # zone; gyro/heavy fastballs (IVB < 14) get groundballs but fewer Ks.
        "arsenal_fb_spin":   float((arsenal or {}).get("fb_avg_spin")     or 2300.0),
        "arsenal_fb_ivb":    float((arsenal or {}).get("fb_ivb")          or 15.0),
        "arsenal_sl_spin":   float((arsenal or {}).get("sl_spin")         or 2400.0),
        "arsenal_sl_drop":   float((arsenal or {}).get("sl_drop")         or 3.0),
        # Pitch effectiveness — plate-discipline metrics against this pitcher.
        # Captures the same signal per-pitch run-values would convey, but
        # season-aggregated (per-pitch CSV endpoint not exposed).
        # - iz_contact:   % of in-zone pitches that get contact (lower = miss-bat stuff)
        # - induce_chase: % of out-of-zone batters chase (higher = deception/tunnel)
        # - oz_contact:   % of out-of-zone swings that make contact (lower = unhittable)
        # - z_swing:      % of in-zone pitches batters swing at (lower = pitcher hides)
        "sc_iz_contact":   float((sc.get("iz_contact")   or 82.0)),
        "sc_induce_chase": float((sc.get("induce_chase") or 31.0)),
        "sc_oz_contact":   float((sc.get("oz_contact")   or 60.0)),
        "sc_z_swing":      float((sc.get("z_swing")      or 67.0)),
    }


# ---------- Train / load / predict ----------
@dataclass
class StatModel:
    """Wraps a trained GBT with its feature column ordering.

    `blend_weight` is the holdout-tuned weight to give the ML model when
    blending with the analytical projection (1.0 = pure ML, 0.0 = pure
    analytical). Computed automatically during training.
    """
    model: HistGradientBoostingRegressor
    feature_cols: list[str]
    train_n: int
    train_mae: float
    holdout_mae: float | None
    blend_weight: float = 0.5
    holdout_mae_blend: float | None = None
    holdout_mae_anal: float | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        return np.clip(self.model.predict(X[self.feature_cols].values), 0.0, None)


def _train_one(df: pd.DataFrame, target_col: str, feature_cols: list[str],
               anal_col: str | None = None,
               loss: str = "poisson", random_state: int = 0,
               holdout_frac: float = 0.15,
               blend_frac: float = 0.15) -> StatModel:
    """Fit a HistGBT for one stat. Uses temporal holdout if a 'date' column
    is present, else random.

    Uses a 3-way split to avoid leaking the reported MAE into blend-weight tuning:
      train  (1 - holdout_frac - blend_frac): fit GBT
      blend  (blend_frac):                    tune blend weight vs analytical
      test   (holdout_frac):                  report honest MAE (never touched during tuning)

    If `anal_col` is not provided, blend and test are merged into a single
    holdout (same behaviour as before, minus the leak).
    """
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(float).clip(lower=0)

    n = len(df)
    if "date" in df.columns:
        df = df.sort_values("date")
    else:
        df = df.iloc[np.random.default_rng(random_state).permutation(n)]

    if anal_col:
        train_end  = int(n * (1 - holdout_frac - blend_frac))
        blend_end  = int(n * (1 - holdout_frac))
        train = df.iloc[:train_end]
        blend = df.iloc[train_end:blend_end]
        test  = df.iloc[blend_end:]
    else:
        train_end = int(n * (1 - holdout_frac))
        train = df.iloc[:train_end]
        blend = df.iloc[0:0]   # empty
        test  = df.iloc[train_end:]

    Xtr = train[feature_cols].values
    ytr = train[target_col].values
    Xte = test[feature_cols].values
    yte = test[target_col].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = HistGradientBoostingRegressor(
            loss=loss, learning_rate=0.05, max_iter=500, max_depth=4,
            min_samples_leaf=60, l2_regularization=1.5,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=random_state,
        )
        m.fit(Xtr, ytr)

    train_mae = float(np.mean(np.abs(m.predict(Xtr) - ytr)))
    test_mae  = float(np.mean(np.abs(m.predict(Xte) - yte))) if len(yte) else None

    # Tune blend weight on the separate blend split, then evaluate on clean test
    blend_w  = 0.5
    blend_mae = test_mae
    anal_mae  = None
    if anal_col and anal_col in blend.columns and len(blend) >= 30:
        ybl       = blend[target_col].values
        ml_bl     = np.clip(m.predict(blend[feature_cols].values), 0, None)
        anal_bl   = blend[anal_col].values
        # Cap ML blend weight at 0.50. The GBT minimises MAE by regressing
        # projections toward the mean, which hurts top-bin calibration —
        # 21-day backtest had EVERY top-decile counting stat under-projecting
        # (pitcher outs by 1.18, pitcher K by 0.66, batter TB by 0.67, etc).
        # Lower cap forces the analytical (with its xBA/whiff/barrel priors)
        # to contribute >= 50%, preserving more spread at the extremes.
        # Cost: 5-10% MAE worsening on noisy stats. Benefit: extreme bins
        # are noticeably better calibrated, which is what we actually bet on.
        best_w, best_m = 0.5, float("inf")
        for w in np.linspace(0.0, 0.50, 11):
            mae_w = float(np.mean(np.abs(w * ml_bl + (1 - w) * anal_bl - ybl)))
            if mae_w < best_m:
                best_w, best_m = float(w), mae_w
        blend_w = best_w

    # Report MAE using the tuned weight on the untouched test split
    if anal_col and anal_col in test.columns and len(yte) >= 10:
        ml_te   = np.clip(m.predict(Xte), 0, None)
        anal_te = test[anal_col].values
        anal_mae  = float(np.mean(np.abs(anal_te - yte)))
        blend_mae = float(np.mean(np.abs(blend_w * ml_te + (1 - blend_w) * anal_te - yte)))

    return StatModel(
        model=m, feature_cols=list(feature_cols),
        train_n=len(train), train_mae=train_mae, holdout_mae=test_mae,
        blend_weight=blend_w, holdout_mae_blend=blend_mae, holdout_mae_anal=anal_mae,
    )


def train_all(
    batter_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, dict[str, StatModel]]:
    """Train one model per stat for batters and pitchers. Saves to disk."""
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, StatModel]] = {"batter": {}, "pitcher": {}}

    # Determine feature columns from the dataframe
    bat_feature_cols = [c for c in batter_df.columns
                        if c not in ("game_pk", "date", "side", "team_id", "opp_team_id",
                                     "player_id", "name", "position",
                                     "actual_h", "actual_hr", "actual_2b", "actual_tb",
                                     "actual_rbi", "actual_runs", "actual_k", "actual_bb",
                                     "actual_pa", "actual_sb",
                                     "h", "hr", "tb", "rbi", "runs_b", "k_b", "bb_b",
                                     "doubles", "triples", "ab", "pa", "sb",
                                     "ip", "outs", "h_p", "er", "k_p", "bb_p", "hr_p", "bf",
                                     "started", "venue", "is_final")]
    pit_feature_cols = [c for c in pitcher_df.columns
                        if c not in ("game_pk", "date", "side", "team_id", "opp_team_id",
                                     "player_id", "name", "position",
                                     "actual_h", "actual_hr", "actual_2b", "actual_tb",
                                     "actual_rbi", "actual_runs", "actual_k", "actual_bb",
                                     "actual_pa", "actual_sb",
                                     "h", "hr", "tb", "rbi", "runs_b", "k_b", "bb_b",
                                     "doubles", "triples", "ab", "pa", "sb",
                                     "ip", "outs", "h_p", "er", "k_p", "bb_p", "hr_p", "bf",
                                     "started", "venue", "is_final")]

    print(f"Batter features: {len(bat_feature_cols)} | Pitcher features: {len(pit_feature_cols)}")

    for spec in BATTER_STATS:
        if spec.actual_col not in batter_df.columns:
            continue
        print(f"  [bat] training {spec.name}... ", end="", flush=True)
        sm = _train_one(batter_df, spec.actual_col, bat_feature_cols,
                        anal_col=spec.proj_col, loss=spec.loss)
        models["batter"][spec.name] = sm
        joblib.dump(sm, out_dir / f"prop_bat_{spec.name}.joblib")
        anal = f" anal={sm.holdout_mae_anal:.3f}" if sm.holdout_mae_anal is not None else ""
        print(f"n={sm.train_n} ml_test={sm.holdout_mae:.3f}{anal} blend(w={sm.blend_weight:.2f})_test={sm.holdout_mae_blend:.3f}")

    for spec in PITCHER_STATS:
        if spec.actual_col not in pitcher_df.columns:
            continue
        print(f"  [pit] training {spec.name}... ", end="", flush=True)
        sm = _train_one(pitcher_df, spec.actual_col, pit_feature_cols,
                        anal_col=spec.proj_col, loss=spec.loss)
        models["pitcher"][spec.name] = sm
        joblib.dump(sm, out_dir / f"prop_pit_{spec.name}.joblib")
        anal = f" anal={sm.holdout_mae_anal:.3f}" if sm.holdout_mae_anal is not None else ""
        print(f"n={sm.train_n} ml_test={sm.holdout_mae:.3f}{anal} blend(w={sm.blend_weight:.2f})_test={sm.holdout_mae_blend:.3f}")

    return models


def load_all(model_dir: Path) -> dict[str, dict[str, StatModel]]:
    """Load whatever models are available; missing ones return empty dicts."""
    import joblib
    out = {"batter": {}, "pitcher": {}}
    for spec in BATTER_STATS:
        p = model_dir / f"prop_bat_{spec.name}.joblib"
        if p.exists():
            try:
                out["batter"][spec.name] = joblib.load(p)
            except Exception:
                pass
    for spec in PITCHER_STATS:
        p = model_dir / f"prop_pit_{spec.name}.joblib"
        if p.exists():
            try:
                out["pitcher"][spec.name] = joblib.load(p)
            except Exception:
                pass
    return out
