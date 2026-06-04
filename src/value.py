"""Find +EV bets by comparing model predictions to sportsbook lines.

Value math:
  - American odds -> implied probability:
        p_implied = 100 / (odds + 100)        if odds > 0
                  = -odds / (-odds + 100)     if odds < 0
  - Sportsbooks bake in juice (vig). For two-way markets we de-vig by
    normalizing the two sides' implied probabilities to sum to 1.
  - Edge = model_p - novig_p
  - EV per $1 stake at decimal odds D:  EV = model_p * (D - 1) - (1 - model_p)
  - Kelly fraction: f* = (b*p - q) / b  where b = D-1, p=model_p, q=1-p

Game prediction -> probability conversions:
  - Moneyline: simulate from two independent Poissons (or Skellam). We use a
    closed-form sum over a small grid since means are 4-6 runs.
  - Total: P(total > line) from the same joint distribution.
  - Run line (-1.5/+1.5): same joint distribution, marginal over score diff.

Player props: project_X gives a mean. We assume Poisson for counting stats
(HR, K, hits, RBI), which is the right family at ~0-3 expected events. For
larger-sample stats (TB, K's for pitchers), we use a Negative Binomial with
a fitted dispersion (default phi=1.5) to allow for over-dispersion.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from math import comb, exp, lgamma


# ---------- Probability + odds utilities ----------
def american_to_prob(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / (-odds)


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    s = p_a + p_b
    if s <= 0:
        return p_a, p_b
    return p_a / s, p_b / s


# Shrinkage applied to raw model probabilities before computing edge. The
# logged bet outcomes from the first two slates showed picks rated >=.65 won
# only 33% — strong overconfidence at the high end. Shrinking toward 0.5
# closes the gap between displayed model probability and realised win rate.
# Bumped 0.6 -> 0.5 after May 2026 holdout: 60-70% bucket claimed 64.5% but
# won 45%; 70-80% bucket claimed 75% but won 50%. The wider shrink hurts
# mid-edge picks slightly but cuts the worst overconfidence in half.
CALIBRATION_SHRINK = 0.5


def calibrate_prob(p: float) -> float:
    """Shrink a raw model probability toward 0.5 — only when p > 0.5.

    p_cal = 0.5 + CALIBRATION_SHRINK * (p - 0.5)  if p > 0.5
          = p                                      otherwise

    The shrinkage was calibrated against overconfidence at the HIGH end
    (picks rated >=.65 winning only 33%). Applying it symmetrically to
    low-prob tails inflates them — a raw 6% becomes a "calibrated" 24%,
    manufacturing fake edges on +1200 longshots. Asymmetric shrink fixes
    the overconfidence without touching the tails.
    """
    if p <= 0.5:
        return p
    return 0.5 + CALIBRATION_SHRINK * (p - 0.5)


# ---- Data-driven game-line win-probability calibration ----
# Fitted by scripts/fit_winprob_calibration.py on actual game outcomes:
#   logit(p_cal) = a + b * logit(p_raw)   (b < 1 shrinks toward 0.5)
# This is the COHERENT two-sided calibration for ML and run-line (both sides
# sum to 1), replacing the asymmetric calibrate_prob for those two markets.
# The June 2026 backtest showed the raw joint-Poisson win probs are
# over-dispersed: predicted 0.30 -> actual 0.42, predicted 0.60 -> actual 0.54.
# Over-confidence inflates fake edges on the bets the model is most wrong
# about. Totals and props keep calibrate_prob (totals are already well
# calibrated; props need the asymmetric no-longshot-inflation behaviour).
_WINPROB_CAL: dict | None = None
_WINPROB_CAL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "winprob_calibration.json"
)


def _load_winprob_cal() -> dict:
    global _WINPROB_CAL
    if _WINPROB_CAL is not None:
        return _WINPROB_CAL
    try:
        _WINPROB_CAL = json.loads(_WINPROB_CAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        _WINPROB_CAL = {}
    return _WINPROB_CAL


def reload_winprob_cal() -> None:
    global _WINPROB_CAL
    _WINPROB_CAL = None


def calibrate_winprob(p: float, market: str) -> float:
    """Apply the fitted logistic recalibration for a game-line market
    ('moneyline' or 'run_line'). Falls back to the asymmetric calibrate_prob
    when no fitted parameters exist (safe default before first fit)."""
    cal = _load_winprob_cal().get(market)
    if not cal or "a" not in cal or "b" not in cal:
        return calibrate_prob(p)
    a = float(cal["a"]); b = float(cal["b"])
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = math.log(p / (1 - p))
    z_cal = a + b * z
    return 1.0 / (1.0 + math.exp(-z_cal))


def kelly_fraction(p: float, decimal_odds: float, cap: float = 0.05) -> float:
    """Kelly bet size as fraction of bankroll. Capped to cap (e.g. 5%)."""
    b = decimal_odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, min(cap, f))


def expected_value(model_p: float, american_odds: int) -> float:
    d = american_to_decimal(american_odds)
    return model_p * (d - 1.0) - (1.0 - model_p)


# ---------- Score distributions ----------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(k * math.log(lam) - lam - lgamma(k + 1))


def joint_score_grid(lam_home: float, lam_away: float, max_k: int = 20) -> np.ndarray:
    """Return P[home_runs=i, away_runs=j] under independent Poisson."""
    h = np.array([poisson_pmf(i, lam_home) for i in range(max_k + 1)])
    a = np.array([poisson_pmf(j, lam_away) for j in range(max_k + 1)])
    return np.outer(h, a)


def home_win_prob(lam_home: float, lam_away: float) -> float:
    """P(home > away). MLB games can't tie, but our discrete grid can produce
    ties (extra-innings get folded into the actual score)."""
    grid = joint_score_grid(lam_home, lam_away)
    n = grid.shape[0]
    p_win = 0.0; p_tie = 0.0
    for i in range(n):
        for j in range(n):
            if i > j:
                p_win += grid[i, j]
            elif i == j:
                p_tie += grid[i, j]
    return p_win + p_tie * 0.5    # split ties (equivalent to flipping a coin in extras)


def total_over_prob(lam_home: float, lam_away: float, line: float) -> float:
    """P(home + away > line) — handles half-point lines naturally; for whole
    numbers, ties are pushes (we return strict >)."""
    grid = joint_score_grid(lam_home, lam_away)
    n = grid.shape[0]
    p_over = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > line:
                p_over += grid[i, j]
    return p_over


def run_line_cover_prob(lam_home: float, lam_away: float, home_spread: float) -> float:
    """P(home wins by more than home_spread runs).
    home_spread = -1.5 means home covers if they win by 2+.
    home_spread = +1.5 means home covers if they lose by 1 or win.
    """
    grid = joint_score_grid(lam_home, lam_away)
    n = grid.shape[0]
    p = 0.0
    for i in range(n):
        for j in range(n):
            margin = i - j   # home margin
            if margin + home_spread > 0:
                p += grid[i, j]
    return p


# ---------- Player prop probabilities ----------
def prob_over_count(mean: float, line: float, dispersion: float = 1.0) -> float:
    """P(X > line) for a counting stat with mean `mean`.

    Uses Poisson when dispersion=1, Negative Binomial when dispersion>1.
    For half-point lines (e.g. 0.5, 1.5) — works exactly.
    For whole-number lines, computes P(X >= line+1) (push convention).
    """
    if mean <= 0:
        return 0.0 if line >= 0 else 1.0
    target = math.floor(line) + 1     # need at least this many to cover an "over X.5" or "over X"

    if dispersion <= 1.0:
        # Poisson
        cdf = 0.0
        for k in range(target):
            cdf += poisson_pmf(k, mean)
        return max(0.0, min(1.0, 1.0 - cdf))
    else:
        # Negative Binomial parameterized by mean + dispersion (variance = mean * dispersion)
        # NB(r, p) with mean = r*(1-p)/p, var = r*(1-p)/p^2 = mean / p
        # so p = mean / variance = 1/dispersion ; r = mean * p / (1-p)
        p = 1.0 / dispersion
        r = mean * p / (1.0 - p)
        cdf = 0.0
        for k in range(target):
            # NB pmf: choose(k+r-1, k) * (1-p)^k * p^r ; r may be non-integer => use lgamma
            log_pmf = (lgamma(k + r) - lgamma(r) - lgamma(k + 1)
                       + k * math.log(1 - p) + r * math.log(p))
            cdf += math.exp(log_pmf)
        return max(0.0, min(1.0, 1.0 - cdf))


# ---------- Value records ----------
@dataclass
class ValueBet:
    market: str            # "moneyline_home", "total_over", "prop:Aaron Judge HR"
    description: str
    line: float
    odds: int
    decimal_odds: float
    model_prob: float
    novig_prob: float
    edge_pct: float
    ev_per_dollar: float
    kelly: float
    # Variance-adjusted leaderboard score: edge_pct × stat-reliability ×
    # outcome-information factor. Higher = more reliable on equal edge.
    # See `score_bet` for the formula.
    confidence: float = 1.0
    score: float = 0.0
    # For bet tracking — populated by predict_core after creation
    game_pk: Optional[int] = None
    player_id: Optional[int] = None
    # True when both starting pitchers for the game are confirmed. Bets with
    # this False (one or both pitchers TBD) are excluded from the Top 5 panel
    # and visually flagged in the leaderboard. The model's pitcher
    # projections fall back to a generic placeholder when the starter is
    # unknown, which means edges are unreliable for those games.
    starters_confirmed: bool = True


# Default reliability weights (sqrt of R² from backtest, eyeballed).
# Loaded at runtime from data/models/stat_reliability.json when that file
# exists. Run `value.write_stat_reliability()` once to create the seed file,
# then update it after each backtest run.
_STAT_RELIABILITY_DEFAULTS: dict[str, float] = {
    # Game lines (team-runs model R^2 ≈ 0.18 on totals)
    "moneyline":   0.55,
    "total":       0.55,
    "run_line":    0.50,
    # Pitcher props — model R^2 0.17–0.31, two-sided pricing
    "prop_pitcher_outs":  0.65,
    "prop_pitcher_k":     0.60,
    "prop_pitcher_h":     0.50,
    "prop_pitcher_er":    0.45,
    "prop_pitcher_bb":    0.40,
    "prop_pitcher_hr":    0.40,
    # Batter props — counting stats are very noisy game-to-game
    "prop_k":     0.45,
    "prop_hr":    0.40,
    "prop_tb":    0.40,
    "prop_hits":  0.38,
    "prop_bb":    0.35,
    "prop_rbi":   0.32,
    "prop_runs":  0.30,
    "prop_sb":    0.25,
}

_STAT_RELIABILITY_CACHE: dict | None = None
_STAT_RELIABILITY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "stat_reliability.json"
)


def _get_stat_reliability() -> dict[str, float]:
    global _STAT_RELIABILITY_CACHE
    if _STAT_RELIABILITY_CACHE is not None:
        return _STAT_RELIABILITY_CACHE
    try:
        loaded = json.loads(_STAT_RELIABILITY_PATH.read_text(encoding="utf-8"))
        merged = dict(_STAT_RELIABILITY_DEFAULTS)
        merged.update(loaded)
        _STAT_RELIABILITY_CACHE = merged
    except Exception:
        _STAT_RELIABILITY_CACHE = dict(_STAT_RELIABILITY_DEFAULTS)
    return _STAT_RELIABILITY_CACHE


def write_stat_reliability(weights: dict[str, float] | None = None, path: Path | None = None) -> None:
    """Write reliability weights to disk (seed or update from backtest output).

    Call with no args to write the current defaults as the seed file.
    Pass `weights` to merge updates from a fresh backtest run.
    """
    target = path or _STAT_RELIABILITY_PATH
    existing: dict = {}
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        pass
    merged = dict(_STAT_RELIABILITY_DEFAULTS)
    merged.update(existing)
    if weights:
        merged.update(weights)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    global _STAT_RELIABILITY_CACHE
    _STAT_RELIABILITY_CACHE = None  # force reload next call


# Edge shrinkage for ranking: the bet log shows that bigger projected edges
# are LESS likely to win — 5-10% edges hit 60% of the time, 15%+ edges only
# 39%. The model's extreme projections (which produce big edges) are where
# its biases compound most. Compress edges above 5% with diminishing returns
# so the leaderboard isn't dominated by fake-bonus longshots.
#
# Curve: edges <= 5% pass through; edges above 5% are compressed via
#        eff = 5 + sqrt(max(0, raw - 5) * 5)
# Examples:
#   raw  3% -> eff  3.0%   (unchanged)
#   raw  5% -> eff  5.0%   (unchanged)
#   raw  8% -> eff  8.9%
#   raw 12% -> eff 10.9%
#   raw 18% -> eff 13.1%
#   raw 25% -> eff 15.0%
def _shrunk_edge_for_ranking(edge_pct: float) -> float:
    if edge_pct <= 5.0:
        return float(edge_pct)
    import math
    return 5.0 + math.sqrt(max(0.0, edge_pct - 5.0) * 5.0)


def annotate(vb: ValueBet) -> ValueBet:
    """Populate confidence + score on a ValueBet. Returns the same object."""
    import math
    rel = _get_stat_reliability().get(vb.market, 0.40)
    p = max(0.01, min(0.99, vb.model_prob))
    # confidence: display metric for outcome uncertainty — peaks at p=0.5
    vb.confidence = float(rel * 4.0 * p * (1.0 - p))
    # score uses a shrunk edge so big "fake" edges don't crowd out genuine
    # mid-edge picks. Display edge_pct stays raw; only ranking changes.
    eff_edge = _shrunk_edge_for_ranking(vb.edge_pct)
    vb.score = float(eff_edge * rel / math.sqrt(p * (1.0 - p)))
    return vb


def score_bet(vb: ValueBet) -> float:
    """Sharpe-like ranking score for a value bet.

    score = (shrunk_edge / sqrt(p*(1-p))) × stat_reliability

    Dividing by the Bernoulli standard deviation normalises edge by outcome
    uncertainty, analogous to a Sharpe ratio. The edge_pct is passed through
    `_shrunk_edge_for_ranking` so the model's extreme-projection bias is
    discounted (15% edges win less than 5-10% edges in our bet log).
    """
    import math
    rel = _get_stat_reliability().get(vb.market, 0.40)
    p = max(0.01, min(0.99, vb.model_prob))
    return _shrunk_edge_for_ranking(vb.edge_pct) * rel / math.sqrt(p * (1.0 - p))


def evaluate_game_lines(
    home_team: str, away_team: str,
    lam_home: float, lam_away: float,
    book: dict,                # {"moneyline": {...}, "total": {...}, "run_line": {...}}
    edge_threshold: float = 0.03,
) -> list[ValueBet]:
    out: list[ValueBet] = []
    # Moneyline
    ml = book.get("moneyline") or {}
    if "home" in ml and "away" in ml:
        p_home = calibrate_winprob(home_win_prob(lam_home, lam_away), "moneyline")
        p_away = 1.0 - p_home
        imp_home = american_to_prob(ml["home"])
        imp_away = american_to_prob(ml["away"])
        nv_h, nv_a = devig_two_way(imp_home, imp_away)
        for side, mp, novig, odds in [
            (f"{home_team} ML", p_home, nv_h, ml["home"]),
            (f"{away_team} ML", p_away, nv_a, ml["away"]),
        ]:
            edge = mp - novig
            if edge >= edge_threshold:
                d = american_to_decimal(odds)
                out.append(annotate(ValueBet(
                    market="moneyline", description=side, line=0.0,
                    odds=odds, decimal_odds=d, model_prob=mp, novig_prob=novig,
                    edge_pct=edge * 100,
                    ev_per_dollar=expected_value(mp, odds),
                    kelly=kelly_fraction(mp, d),
                )))

    # Total
    tot = book.get("total") or {}
    if "line" in tot:
        line = float(tot["line"])
        p_over = calibrate_prob(total_over_prob(lam_home, lam_away, line))
        p_under = 1.0 - p_over
        imp_o = american_to_prob(tot.get("over", -110))
        imp_u = american_to_prob(tot.get("under", -110))
        nv_o, nv_u = devig_two_way(imp_o, imp_u)
        for side, mp, novig, odds in [
            (f"{away_team} @ {home_team} Over {line}",  p_over,  nv_o, tot.get("over", -110)),
            (f"{away_team} @ {home_team} Under {line}", p_under, nv_u, tot.get("under", -110)),
        ]:
            edge = mp - novig
            if edge >= edge_threshold:
                d = american_to_decimal(odds)
                out.append(annotate(ValueBet(
                    market="total", description=side, line=line,
                    odds=odds, decimal_odds=d, model_prob=mp, novig_prob=novig,
                    edge_pct=edge * 100,
                    ev_per_dollar=expected_value(mp, odds),
                    kelly=kelly_fraction(mp, d),
                )))

    # Run line. The stored `line` is SIGNED for the home team:
    #   line = -1.5 when home is the favorite (gives 1.5 runs)
    #   line = +1.5 when home is the underdog (gets 1.5 runs)
    rl = book.get("run_line") or {}
    if "line" in rl:
        home_line = float(rl["line"])
        away_line = -home_line
        p_home_cover = calibrate_winprob(run_line_cover_prob(lam_home, lam_away, home_line), "run_line")
        p_away_cover = 1.0 - p_home_cover
        imp_h = american_to_prob(rl.get("home", +160))
        imp_a = american_to_prob(rl.get("away", -185))
        nv_h, nv_a = devig_two_way(imp_h, imp_a)
        for side, mp, novig, odds, spread in [
            (f"{home_team} {home_line:+.1f}", p_home_cover, nv_h, rl.get("home", +160), home_line),
            (f"{away_team} {away_line:+.1f}", p_away_cover, nv_a, rl.get("away", -185), away_line),
        ]:
            edge = mp - novig
            if edge >= edge_threshold:
                d = american_to_decimal(odds)
                out.append(annotate(ValueBet(
                    market="run_line", description=side, line=spread,
                    odds=odds, decimal_odds=d, model_prob=mp, novig_prob=novig,
                    edge_pct=edge * 100,
                    ev_per_dollar=expected_value(mp, odds),
                    kelly=kelly_fraction(mp, d),
                )))
    return out


# Hardcoded fallback dispersions when no empirical fit is loaded. These are
# only used if data/models/dispersion.json is missing. Empirical curves from
# the 2026 season tend to be tighter (more Poisson-like) than these defaults
# for raw counting stats, and looser for derived stats like TB and RBI.
PROP_DISPERSION = {
    "hr": 1.0, "hits": 1.0, "tb": 2.0, "rbi": 1.5, "runs": 1.0,
    "k": 1.0, "bb": 1.0, "sb": 1.0,
    # Pitcher
    "pitcher_k": 1.2, "pitcher_outs": 1.2, "pitcher_er": 1.6, "pitcher_h": 1.2,
    "pitcher_bb": 1.0, "pitcher_hr": 1.2,
}


# Lazy-loaded empirical dispersion fits.
_DISP_FITS: dict | None = None


def _get_dispersion_fits() -> dict:
    global _DISP_FITS
    if _DISP_FITS is not None:
        return _DISP_FITS
    try:
        from pathlib import Path
        from . import dispersion
        path = Path(__file__).resolve().parent.parent / "data" / "models" / "dispersion.json"
        _DISP_FITS = dispersion.load_fits(path)
    except Exception:
        _DISP_FITS = {}
    return _DISP_FITS


def get_dispersion(market: str, mean_proj: float) -> float:
    """μ-conditional empirical dispersion if fitted; otherwise hardcoded fallback."""
    fits = _get_dispersion_fits()
    if market in fits:
        return fits[market].at(mean_proj)
    return PROP_DISPERSION.get(market, 1.3)


def evaluate_prop(name: str, market: str, mean_proj: float, line: float,
                  over_odds: int | None, under_odds: int | None,
                  edge_threshold: float = 0.03,
                  one_sided_juice: float = 0.08,
                  one_sided_max_odds: int = 400) -> list[ValueBet]:
    """Compare a player prop to its model projection.

    Two-sided (over_odds AND under_odds): we de-vig and report both sides.
    One-sided (only over_odds, common for Yes/No props on Bovada): we estimate
    the no-vig prob by stripping a typical book overround. Default
    one_sided_juice = 0.08 (raised from 0.06 May 6 2026). Empirical Bovada
    overround on binary batter props (HR / Hits / TB Yes-only) is 8-12%, not
    6%. Using 6% systematically inflated edge by ~1-2 pct points and crowded
    the leaderboard with phantom +EV one-sided plays. 0.08 is the conservative
    midpoint of the observed band.

    one_sided_max_odds: skip one-sided props with American odds above this
    threshold (default +400 ≈ 20% implied). Bovada's overround on +500 / +800
    / +1200 longshots is much wider than 6% — typically 15-25% — so the no-vig
    estimate is unreliable. Capping prevents fake "edges" on lottery-ticket
    lines like 'Player runs OVER 1.5' from crowding the leaderboard.
    """
    disp = get_dispersion(market, mean_proj)
    p_over = calibrate_prob(prob_over_count(mean_proj, line, disp))
    p_under = 1.0 - p_over

    out: list[ValueBet] = []

    if over_odds is not None and under_odds is not None:
        imp_o = american_to_prob(over_odds); imp_u = american_to_prob(under_odds)
        nv_o, nv_u = devig_two_way(imp_o, imp_u)
        for side_name, mp, novig, odds in [
            (f"{name} {market} OVER {line}",  p_over,  nv_o, over_odds),
            (f"{name} {market} UNDER {line}", p_under, nv_u, under_odds),
        ]:
            edge = mp - novig
            if edge >= edge_threshold:
                d = american_to_decimal(odds)
                out.append(annotate(ValueBet(
                    market=f"prop_{market}", description=side_name, line=line,
                    odds=odds, decimal_odds=d, model_prob=mp, novig_prob=novig,
                    edge_pct=edge * 100,
                    ev_per_dollar=expected_value(mp, odds),
                    kelly=kelly_fraction(mp, d),
                )))
    elif over_odds is not None:
        # Cap longshots: at +400 or worse, Bovada's juice estimate breaks down
        # and we'd manufacture fake edges. Drop the bet entirely.
        if over_odds > one_sided_max_odds:
            return out
        # One-sided: estimate no-vig prob = implied - juice/2. The no-vig
        # estimate is rough (Bovada batter props carry 6-12% overround on
        # binary markets). We label the description so users know.
        imp = american_to_prob(over_odds)
        novig = max(0.01, min(0.99, imp - one_sided_juice / 2.0))
        edge = p_over - novig
        if edge >= edge_threshold:
            d = american_to_decimal(over_odds)
            out.append(annotate(ValueBet(
                market=f"prop_{market}",
                description=f"{name} {market} OVER {line} [1-sided, ~{int(one_sided_juice*100)}% juice est.]",
                line=line, odds=over_odds, decimal_odds=d,
                model_prob=p_over, novig_prob=novig,
                edge_pct=edge * 100,
                ev_per_dollar=expected_value(p_over, over_odds),
                kelly=kelly_fraction(p_over, d),
            )))

    return out
