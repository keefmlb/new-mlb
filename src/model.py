"""Team scoring model.

Predicts the runs each team scores in a given game. We treat each game as
two observations (home batting and away batting) so the same model handles both.

Architecture:
  - Linear features for offense, opposing pitcher quality, bullpen, park, weather
  - sklearn PoissonRegressor (GLM with log link) so predictions are non-negative
    and the variance scales with the mean (matches runs).
  - Joint final score uses each team's predicted runs.

Why Poisson and not a tree-based regressor?
  Trees overfit at this sample size (a few hundred games per season). A linear
  Poisson GLM with hand-built features is well-suited to ~400 obs and gives
  calibrated mean predictions, which is what we need for prop pricing.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


# Features used to predict runs scored by the *batting* team.
# Trimmed to reduce multicollinearity. We drop redundant rate-stats (kept OPS,
# wOBA, BB%; dropped K%/ISO since they're absorbed by OPS/wOBA), redundant
# pitcher metrics (kept xFIP, FIP-, BB/9; dropped FIP since xFIP already
# captures it), and the linear-combination interaction off_minus_oppsp
# (perfectly collinear with off_rpg + opp_sp_era).
# Pruned Jun 4 2026: a full-season coefficient audit showed 14 features with
# |coef| < 0.003 on z-scored inputs (effectively dead under L2). They were
# either collinear with stronger features or pure noise, and the model beat a
# constant-mean baseline by only ~2.5% on totals regardless. Removed to cut
# overfit surface and feature-count fragility. The dropped columns are still
# COMPUTED in _half()/GameFeatures (available for prop models or a future
# regularized refit), just not consumed by the team-runs GLM:
#   lineup_ops, lineup_woba, lineup_woba_recent  (collinear w/ lineup_xwoba_vs_hand, off_xwoba_sc)
#   off_sf_pg, off_sprint_speed                  (noise / no signal)
#   opp_sp_fip_minus                             (collinear w/ opp_sp_xfip)
#   opp_def_oaa, opp_bp_ip_72h, opp_bp_top_rest  (dead bullpen/defense workload)
#   opp_catcher_x_ump, opp_sp_ppi, opp_bp_era_recent_gap, opp_sp_fip_recent_gap
#   ump_k_mult                                   (team-level dead; still used for K props)
FEATURES = [
    # Offense (batting team) — season team aggregate (broad context)
    "off_rpg", "off_babip",
    # Lineup-weighted offense (PA-weighted over the 9 confirmed starters)
    "lineup_bb_pct",
    # Sequential offense — small-ball metrics that OPS/wOBA-weighted features miss
    "off_sb_net_pg", "off_gidp_pg",
    # Platoon: lineup wOBA vs the OPPOSING starter's throwing hand (EB-shrunk).
    # The single strongest team-runs feature.
    "lineup_xwoba_vs_hand",
    # Statcast offense — park/defense-neutral contact quality (team-level)
    "off_xwoba_sc",
    # Opposing starter (advanced)
    "opp_sp_xfip", "opp_sp_bb9", "opp_sp_hr9",
    # Opposing starter Statcast — contact quality allowed
    "opp_sp_xera_sc",
    # Opposing bullpen / staff
    "opp_bp_era", "opp_pit_era_recent",
    # Park — runs and HR factors + elevation
    "park_pf_runs", "park_pf_hr", "park_elev_ft",
    # Roof indicator — retractables suppress runs beyond the park factor
    "park_is_retractable",
    # Pitcher rest (of the opposing starter we're hitting against)
    "opp_sp_days_rest",
    # Day game flag
    "is_day_game",
    # Catcher framing: opp catcher's K-rate-boost proxy from box-score history
    "opp_catcher_framing",
    # Weather
    "runs_mult", "hr_mult", "wind_to_cf_mph", "temp_f",
    # Engineered interactions (multiplicative — not linear combos)
    "off_x_park", "opp_sp_x_off",
    # Context
    "is_home",
]


def _pick(g: dict, key: str, default=0.0):
    return g[key] if key in g else default


def _half(g: dict, batting: str) -> dict:
    """Reshape one game row into one (offense=batting) row."""
    home_batting = batting == "home"
    h = "home" if home_batting else "away"
    o = "away" if home_batting else "home"

    def own(field):    # batting team's own field
        return _pick(g, f"{h}_{field}")

    def opp(field):    # opposing team's field
        return _pick(g, f"{o}_{field}")

    row = {
        "game_pk":     g["game_pk"],
        "date":        g["date"],
        "venue":       g["venue"],
        "team":        g["home_team"]   if home_batting else g["away_team"],
        "team_id":     g["home_team_id"] if home_batting else g["away_team_id"],
        "opp":         g["away_team"]   if home_batting else g["home_team"],
        "opp_id":      g["away_team_id"] if home_batting else g["home_team_id"],
        "is_home":     1 if home_batting else 0,
        # Offense (own batting team)
        "off_rpg":       own("off_rpg"),
        "off_k_pct":     own("off_k_pct"),
        "off_bb_pct":    own("off_bb_pct"),
        "off_iso":       own("off_iso"),
        "off_ops":       own("off_ops"),
        "off_woba":      own("off_woba"),
        "off_babip":     own("off_babip"),
        "off_rpg_recent": own("off_rpg_recent"),
        "off_ops_recent": own("off_ops_recent"),
        # Sequential offense — small-ball metrics that OPS/wOBA miss
        "off_sb_pg":     own("off_sb_pg")     if f"{h}_off_sb_pg" in g else 0.60,
        "off_sf_pg":     own("off_sf_pg")     if f"{h}_off_sf_pg" in g else 0.30,
        "off_gidp_pg":   own("off_gidp_pg")   if f"{h}_off_gidp_pg" in g else 0.75,
        "off_sb_net_pg": own("off_sb_net_pg") if f"{h}_off_sb_net_pg" in g else 0.30,
        # Team sprint speed (own batting team) — athleticism/baserunning proxy
        "off_sprint_speed": own("sprint_speed") if f"{h}_sprint_speed" in g else 27.0,
        # Opposing starter
        "opp_sp_fip":      opp("sp_fip"),
        "opp_sp_xfip":     opp("sp_xfip"),
        "opp_sp_fip_minus": opp("sp_fip_minus"),
        "opp_sp_k9":       opp("sp_k9"),
        "opp_sp_bb9":      opp("sp_bb9"),
        "opp_sp_hr9":      opp("sp_hr9"),
        # Opposing bullpen / staff
        "opp_bp_era":      opp("bp_era"),
        "opp_bp_fip":      opp("bp_fip"),
        "opp_pit_era_recent": opp("pit_era_recent"),
        # Park
        "park_pf_runs":    g["park_pf_runs"],
        "park_pf_hr":      g["park_pf_hr"],
        "park_pf_h":       _pick(g, "park_pf_h", 1.0),
        "park_elev_ft":    g.get("park_elev_ft", 500),
        # Roof indicators — derive from park_roof string (present in BOTH
        # 2025 and 2026 CSVs) rather than reading a column that only exists
        # in 2026. Prevents the era-bias bug where 2025 rows would default
        # to 0 and the GLM would learn "dome=1 means 2026" instead of the
        # real roof effect.
        "park_is_dome":        1 if str(g.get("park_roof", "")).lower() == "dome"        else 0,
        "park_is_retractable": 1 if str(g.get("park_roof", "")).lower() == "retractable" else 0,
        # Pitcher days rest — applies to the OPPOSING starter (relevant for
        # how much offense scores). 5 = typical, <4 = quick rest, >7 = rust.
        "opp_sp_days_rest": _pick(g, f"{o}_sp_days_rest", 5.0),
        # Day/night flag
        "is_day_game":      _pick(g, "is_day_game", 0),
        # Opposing team's defense (better defense -> we score less)
        "opp_def_oaa":      _pick(g, f"{o}_def_oaa", 0.0),
        # Opposing bullpen workload: more IP last 72hr -> more fatigue -> we score more.
        # top_rest = days since highest-leverage relievers last pitched (lower = tired).
        "opp_bp_ip_72h":    _pick(g, f"{o}_bp_ip_72h", 0.0),
        "opp_bp_top_rest":  _pick(g, f"{o}_bp_top_rest", 7.0),
        # Opposing catcher framing — positive = catcher steals strikes, more Ks for
        # opp pitching, our offense suffers. Plus interaction with umpire K-mult
        # so we capture "framing pays more under wide-zone umpires" effect.
        "opp_catcher_framing":     _pick(g, f"{o}_catcher_framing", 0.0),
        "opp_catcher_x_ump":       _pick(g, f"{o}_catcher_framing", 0.0) * (
                                       _pick(g, "ump_k_mult", 1.0) - 1.0),
        # Opposing SP recent FIP gap (recent - season). Negative = SP has been
        # running hotter than season (better recent form). Lets the GLM
        # capture "ace stumbling" or "journeyman dialed in" effects that the
        # static season xFIP misses.
        "opp_sp_fip_recent_gap":   (_pick(g, f"{o}_sp_fip_recent", 4.10)
                                    - _pick(g, f"{o}_sp_fip", 4.10)),
        # Opposing SP pitches-per-inning. Low PPI = command, goes deeper,
        # less bullpen exposure -> we score less. League ~15.5.
        "opp_sp_ppi":              _pick(g, f"{o}_sp_ppi", 15.5),
        # Opposing bullpen 14d ERA gap (recent - season). Captures recent
        # bullpen form shift: positive = bullpen running hot, negative =
        # they've been getting torched (good for our offense).
        "opp_bp_era_recent_gap":   (_pick(g, f"{o}_bp_era_recent", 4.30)
                                    - _pick(g, f"{o}_bp_era", 4.30)),
        # Weather
        "runs_mult":       g["runs_mult"],
        "hr_mult":         g["hr_mult"],
        "wind_to_cf_mph":  g.get("wind_to_cf_mph", 0.0),
        "temp_f":          g.get("temp_f", 70.0),
        # Statcast — use explicit defaults so old CSVs without these columns
        # fall back to league average rather than 0.
        "off_xwoba_sc":    _pick(g, f"{h}_off_xwoba_sc",    0.315),
        "off_barrel_rate": _pick(g, f"{h}_off_barrel_rate", 7.8),
        "opp_sp_xera_sc":  _pick(g, f"{o}_sp_xera_sc",      4.20),
        "ump_k_mult":      _pick(g, "ump_k_mult",            1.0),
        # Lineup-weighted offense (own batting team)
        "lineup_ops":      _pick(g, f"{h}_lineup_ops",       0.720),
        "lineup_woba":     _pick(g, f"{h}_lineup_woba",      0.315),
        "lineup_bb_pct":   _pick(g, f"{h}_lineup_bb_pct",    0.085),
        "lineup_k_pct":    _pick(g, f"{h}_lineup_k_pct",     0.225),
        "lineup_xwoba_vs_hand": _pick(g, f"{h}_lineup_xwoba_vs_hand", 0.315),
        # Lineup-weighted recent form — distinguishes 5 hot starters from
        # team's hot bench. Falls back to lineup season aggregate when missing.
        "lineup_woba_recent":  _pick(g, f"{h}_lineup_woba_recent",  0.315),
        "lineup_ops_recent":   _pick(g, f"{h}_lineup_ops_recent",   0.720),
        # Engineered interactions
        # off_x_park and off_minus_oppsp are per-side computed in features.py
        # under the batting team's prefix, so use own().
        # opp_sp_x_off was stored under the *pitching* team's prefix (e.g.
        # `home_sp_x_off` = home SP × away offense → applies when away is batting),
        # so the right lookup is opp().
        "off_x_park":      own("off_x_park"),
        "opp_sp_x_off":    opp("sp_x_off"),
        "off_minus_oppsp": own("off_minus_oppsp"),
        # Targets
        "y_runs":      g["home_score"] if home_batting else g["away_score"],
        "is_final":    g["is_final"],
    }
    return row


def long_form(games_df: pd.DataFrame) -> pd.DataFrame:
    """Convert one row per game -> two rows per game (home_batting, away_batting)."""
    rows: list[dict] = []
    for _, g in games_df.iterrows():
        rows.append(_half(g, "home"))
        rows.append(_half(g, "away"))
    return pd.DataFrame(rows)


@dataclass
class TeamScoreModel:
    """Ensemble of a Poisson GLM and a HistGradientBoostingRegressor with
    optional isotonic recalibration.

    The GLM provides calibration and stability; the GBT captures non-linear
    interactions (e.g. wind-by-park combos). We blend their predictions with
    learned weights from the holdout fold.

    Isotonic recalibration learns a monotone mapping `lambda_raw -> lambda_cal`
    on the holdout, which fixes the compression-toward-the-mean bias common
    in regularized GLMs.
    """
    glm: PoissonRegressor
    gbt: Optional[HistGradientBoostingRegressor]
    feature_means: dict[str, float]
    feature_stds:  dict[str, float]
    blend_weight_glm: float            # weight on GLM in the blend (0..1); GBT gets 1-w
    isotonic: Optional[IsotonicRegression]
    train_games: int
    train_mae: float

    def n_features(self) -> int:
        """Number of features this model was trained on."""
        return self.glm.n_features_in_

    def predict_runs(self, X: pd.DataFrame, mode: str = "ensemble",
                     recalibrate: bool = True) -> np.ndarray:
        # Fill any features present in FEATURES but absent from X with defaults.
        # Handles models trained before a new feature was added.
        X = X.copy()
        for c in FEATURES:
            if c not in X.columns:
                X[c] = 1.0 if c == "ump_k_mult" else 0.0
        Xn = self._normalize(X)
        Xv = Xn[FEATURES].values
        if mode == "glm" or self.gbt is None:
            preds = self.glm.predict(Xv)
        elif mode == "gbt":
            preds = np.clip(self.gbt.predict(Xv), 0.0, None)
        else:
            w = self.blend_weight_glm
            preds = w * self.glm.predict(Xv) + (1 - w) * np.clip(self.gbt.predict(Xv), 0.0, None)
        if recalibrate and self.isotonic is not None:
            preds = self.isotonic.predict(preds)
        return preds

    def _normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in FEATURES:
            if c == "is_home":
                continue
            mu = self.feature_means.get(c, 0.0); sd = self.feature_stds.get(c, 1.0) or 1.0
            out[c] = (out[c] - mu) / sd
        return out

    def save(self, path: str | Path):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "TeamScoreModel":
        import joblib
        return joblib.load(path)


def _normalize_features(df: pd.DataFrame, means: dict, stds: dict) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        if c == "is_home":
            continue
        out[c] = (out[c] - means[c]) / (stds[c] or 1.0)
    return out


def _tune_glm_alpha(Xtr, ytr, alphas: list[float] | None = None) -> float:
    """Tune Poisson ridge alpha using a held-out tail of the training set.

    We use the LAST 10% of train rows as a tuning set to mimic the temporal
    structure (cheap surrogate for nested CV at this sample size).
    """
    alphas = alphas or [0.1, 0.3, 1.0, 3.0, 10.0]
    n = len(ytr)
    cut = int(n * 0.9)
    if cut < 50 or n - cut < 20:
        return 1.0
    Xa, Xb = Xtr[:cut], Xtr[cut:]
    ya, yb = ytr[:cut], ytr[cut:]
    best, best_mae = 1.0, float("inf")
    for a in alphas:
        m = PoissonRegressor(alpha=a, max_iter=5000)
        m.fit(Xa, ya)
        mae = float(np.mean(np.abs(m.predict(Xb) - yb)))
        if mae < best_mae:
            best, best_mae = a, mae
    return best


def _select_blend_weight_cv(train: pd.DataFrame, means: dict, stds: dict,
                            glm_alpha: float, gbt_kwargs: dict,
                            n_folds: int = 6, fold_days: int = 14) -> float:
    """Pick the GLM weight in the GLM/GBT blend via temporal walk-forward CV on
    the TRAIN set, optimising **moneyline accuracy** (does the higher-predicted
    team win?) with run-MAE as a tiebreaker.

    Why ML accuracy, not MAE: run-MAE is essentially flat between the GLM and
    the GBT (single-game scoring is noise-dominated, ~2.5 either way), so an
    MAE objective can't tell them apart and defaults GLM-heavy. But the GBT
    picks winners several points better than the GLM out-of-sample — which is
    what actually matters for the run predictions feeding prop projections and
    win probabilities. Selecting on ML accuracy captures that.
    """
    tr = train.copy()
    tr["date"] = pd.to_datetime(tr["date"])
    tr = tr.sort_values("date")
    dmax = tr["date"].max()
    weights = [round(w, 2) for w in np.linspace(0.0, 1.0, 11)]
    acc_by_w: dict[float, list] = {w: [] for w in weights}
    mae_by_w: dict[float, list] = {w: [] for w in weights}
    for k in range(n_folds, 0, -1):
        lo = dmax - pd.Timedelta(days=fold_days * k)
        hi = dmax - pd.Timedelta(days=fold_days * (k - 1))
        sub_tr = tr[tr["date"] < lo]
        sub_te = tr[(tr["date"] >= lo) & (tr["date"] < hi)]
        if len(sub_tr) < 500 or len(sub_te) < 40:
            continue
        Xa = _normalize_features(sub_tr, means, stds)[FEATURES].values
        ya = sub_tr["y_runs"].values
        Xb = _normalize_features(sub_te, means, stds)[FEATURES].values
        yb = sub_te["y_runs"].values
        g = PoissonRegressor(alpha=glm_alpha, max_iter=5000).fit(Xa, ya)
        pg = g.predict(Xb)
        bt = HistGradientBoostingRegressor(**gbt_kwargs).fit(Xa, ya)
        pb = np.clip(bt.predict(Xb), 0.0, None)
        gpk = sub_te["game_pk"].values
        is_home = sub_te["is_home"].values
        for w in weights:
            blend = w * pg + (1 - w) * pb
            mae_by_w[w].append(float(np.mean(np.abs(blend - yb))))
            # Pair home/away by game to score moneyline accuracy.
            by_g: dict = {}
            for gp_, ih_, pr_, ya_ in zip(gpk, is_home, blend, yb):
                d = by_g.setdefault(gp_, {})
                d["ph" if ih_ == 1 else "pa"] = pr_
                d["yh" if ih_ == 1 else "ya"] = ya_
            correct = tot = 0
            for d in by_g.values():
                if {"ph", "pa", "yh", "ya"} <= set(d):
                    correct += int((d["ph"] > d["pa"]) == (d["yh"] > d["ya"])); tot += 1
            if tot:
                acc_by_w[w].append(correct / tot)
    scored = [(w, float(np.mean(acc_by_w[w])), float(np.mean(mae_by_w[w])))
              for w in weights if acc_by_w[w]]
    if not scored:
        return 0.5
    # Maximise ML accuracy; break ties by lower MAE.
    best = max(scored, key=lambda t: (round(t[1], 4), -t[2]))
    return best[0]


def fit(games_df: pd.DataFrame, holdout_days: int = 7,
        glm_alpha: float | None = None,
        use_gbt: bool = True,
        use_isotonic: bool = False,
        gbt_kwargs: dict | None = None) -> tuple[TeamScoreModel, pd.DataFrame]:
    """Train the model on completed games, hold out the most recent `holdout_days` for eval.

    Returns (model, eval_df_with_preds). Includes Poisson-GLM, GBT, and blended
    ensemble predictions on the holdout. The blend weight is *learned* on the
    holdout (simple grid search over [0,1]) — for full honesty we use a
    walk-forward CV in cv.run_walk_forward().
    """
    final = games_df[games_df["is_final"] == True].copy()
    long = long_form(final)
    long = long.dropna(subset=["y_runs"]).copy()
    long["y_runs"] = long["y_runs"].astype(float)

    # Temporal split
    long = long.sort_values("date")
    cutoff = long["date"].unique()
    if len(cutoff) > holdout_days:
        train_dates = set(cutoff[:-holdout_days])
        train = long[long["date"].isin(train_dates)]
        test  = long[~long["date"].isin(train_dates)]
    else:
        train, test = long.iloc[:-30], long.iloc[-30:]

    means = {c: float(train[c].mean()) for c in FEATURES if c != "is_home"}
    stds  = {c: float(train[c].std() or 1.0) for c in FEATURES if c != "is_home"}

    Xtr = _normalize_features(train, means, stds)[FEATURES].values
    ytr = train["y_runs"].values
    Xte = _normalize_features(test,  means, stds)[FEATURES].values
    yte = test["y_runs"].values

    if glm_alpha is None:
        glm_alpha = _tune_glm_alpha(Xtr, ytr)
    glm = PoissonRegressor(alpha=glm_alpha, max_iter=5000)
    glm.fit(Xtr, ytr)
    glm_te = glm.predict(Xte)

    gbt = None
    gbt_te = None
    if use_gbt:
        # Tree-based regressor with Poisson loss — naturally non-negative outputs
        # and matches the variance structure of run scoring. Defaults are
        # tuned for ~600-row datasets; tighten max_iter to prevent overfit.
        # Strong regularization for this small-sample regime: shallow trees,
        # large leaves, early stopping on a validation tail.
        # Params tuned via 5-fold walk-forward CV on 2025+2026 (Jun 2026):
        # the deeper config (depth 5, lr 0.03, 600 iters, lighter L2) lifts
        # ML accuracy ~+5pp over the old shallow config AND over the GLM —
        # the GLM alone was picking winners below 50% on recent holdouts.
        kwargs = dict(
            loss="poisson",
            learning_rate=0.03,
            max_iter=600,
            max_depth=5,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=0,
        )
        if gbt_kwargs:
            kwargs.update(gbt_kwargs)
        gbt = HistGradientBoostingRegressor(**kwargs)
        gbt.fit(Xtr, ytr)
        gbt_te = np.clip(gbt.predict(Xte), 0.0, None)

    # Select the blend weight via temporal walk-forward CV on the TRAIN set —
    # robust to the single-holdout noise that previously landed GLM-heavy even
    # though the GBT is the more accurate out-of-sample learner.
    if gbt_te is not None:
        best_w = _select_blend_weight_cv(train, means, stds, glm_alpha, kwargs)
        best_mae = float(np.mean(np.abs(best_w * glm_te + (1 - best_w) * gbt_te - yte)))
    else:
        best_w = 1.0
        best_mae = float(np.mean(np.abs(glm_te - yte)))

    eval_df = test.copy()
    eval_df["pred_glm"] = glm_te
    if gbt_te is not None:
        eval_df["pred_gbt"] = gbt_te
        eval_df["pred_runs"] = best_w * glm_te + (1 - best_w) * gbt_te
    else:
        eval_df["pred_runs"] = glm_te
    eval_df["resid"] = eval_df["pred_runs"] - yte

    glm_mae = float(np.mean(np.abs(glm_te - yte)))
    gbt_mae = float(np.mean(np.abs(gbt_te - yte))) if gbt_te is not None else None

    train_mae = float(np.mean(np.abs(glm.predict(Xtr) - ytr)))
    eval_df.attrs["train_mae"] = train_mae
    eval_df.attrs["test_mae"] = best_mae
    eval_df.attrs["test_mae_glm"] = glm_mae
    eval_df.attrs["test_mae_gbt"] = gbt_mae
    eval_df.attrs["blend_w_glm"] = best_w
    eval_df.attrs["glm_alpha"] = glm_alpha

    # ---- Optional isotonic recalibration ----
    # Off by default. Empirically (2026-04 sample) the raw Poisson GLM is
    # already well-calibrated within its 6-bin range and isotonic adds
    # variance without removing material bias. We keep the wiring so it can
    # be re-enabled when the season grows enough to support it.
    iso = None
    if use_isotonic:
        n = len(ytr)
        cut = int(n * 0.85)
        if cut > 50 and (n - cut) >= 20:
            Xa, Xb = Xtr[:cut], Xtr[cut:]
            ya, yb = ytr[:cut], ytr[cut:]
            glm_a = PoissonRegressor(alpha=glm_alpha, max_iter=5000).fit(Xa, ya)
            if use_gbt:
                gbt_a = HistGradientBoostingRegressor(**kwargs).fit(Xa, ya)
                raw_b = best_w * glm_a.predict(Xb) + (1 - best_w) * np.clip(gbt_a.predict(Xb), 0.0, None)
            else:
                raw_b = glm_a.predict(Xb)
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw_b, yb)
            cal_te = iso.predict(eval_df["pred_runs"].values)
            eval_df["pred_runs_calibrated"] = cal_te
            eval_df.attrs["test_mae_calibrated"] = float(np.mean(np.abs(cal_te - yte)))

    model = TeamScoreModel(
        glm=glm, gbt=gbt,
        feature_means=means, feature_stds=stds,
        blend_weight_glm=best_w,
        isotonic=iso,
        train_games=len(train) // 2,
        train_mae=train_mae,
    )
    return model, eval_df


def baseline_predict(long_df: pd.DataFrame) -> np.ndarray:
    """Simple baseline: predict each team's season RPG. Used to benchmark the GLM."""
    return long_df["off_rpg"].values


# ---------- Ensemble utilities ----------

def load_bootstrap_ensemble(model_dir: str | Path) -> list[TeamScoreModel]:
    """Load all bootstrap-replicate models saved by train_combined.py."""
    import joblib
    models: list[TeamScoreModel] = []
    for i in range(20):   # look for up to 20 replicates
        p = Path(model_dir) / f"team_runs_boot_{i}.joblib"
        if not p.exists():
            break
        try:
            models.append(joblib.load(p))
        except Exception:
            pass
    return models


def load_temporal_ensemble(model_dir: str | Path) -> list[TeamScoreModel]:
    """Load 60d and 14d temporal window models saved by train_combined.py."""
    import joblib
    models: list[TeamScoreModel] = []
    for tag in ("60d", "14d"):
        p = Path(model_dir) / f"team_runs_{tag}.joblib"
        if p.exists():
            try:
                models.append(joblib.load(p))
            except Exception:
                pass
    return models


def predict_ensemble(models: list[TeamScoreModel], X: pd.DataFrame,
                     mode: str = "ensemble") -> np.ndarray:
    """Weighted-average prediction across multiple models.

    Weights are proportional to 1/train_mae so better-calibrated models
    contribute more. Falls back gracefully if list is empty or has one model.
    Stale models (wrong feature count) are skipped with a warning.
    """
    if not models:
        raise ValueError("predict_ensemble: empty model list")
    expected = len(FEATURES)
    valid = [m for m in models if m.glm.n_features_in_ == expected]
    if not valid:
        # All models are stale relative to current FEATURES. The previous
        # behaviour was to try them anyway, which produced a cryptic sklearn
        # `_check_n_features` ValueError. Surface a clear message instead so
        # the operator knows to re-run train_combined.
        counts = sorted({m.glm.n_features_in_ for m in models})
        raise RuntimeError(
            f"All ensemble models were trained with a different feature count "
            f"({counts}) than the current FEATURES list ({expected}). "
            f"Re-run `python -m scripts.train_combined` to retrain, or pull "
            f"the latest joblib files."
        )
    if len(valid) == 1:
        return valid[0].predict_runs(X, mode=mode)
    preds = np.array([m.predict_runs(X, mode=mode) for m in valid])
    weights = np.array([1.0 / max(m.train_mae, 0.01) for m in valid])
    weights /= weights.sum()
    return (preds * weights[:, None]).sum(axis=0)
