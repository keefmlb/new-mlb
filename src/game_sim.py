"""Play-by-play Monte Carlo game simulator.

The rest of the system projects each player as an INDEPENDENT marginal — a
batter's RBI and a teammate's runs come from separate distributions that never
share a game state. This engine instead simulates the game as a whole: plate
appearance by plate appearance through both lineups, tracking base-runners,
outs, innings, and the starter→bullpen hook, so runs and RBI EMERGE from the
sequence the way they do in a real game. Run it thousands of times and every
stat becomes a coherent distribution — a clean picture of what the model
actually favors once the pieces interact.

Inputs are the per-player projection dicts the slate already produces
(GamePrediction.home_batters / away_batters / home_starter / away_starter)
plus the team-runs model's projected runs for anchoring. No new model: the
per-PA event rates are derived from the SAME projections, so the simulation
is a coherent re-expression of the current model, not a competing one.

Anchoring: the bottom-up PA rates produce their own team total, which may
differ from the team-runs GLM. We report BOTH — the free bottom-up mean and
the GLM projection — and scale the rates so the headline box score centers on
the GLM (keeping the sim consistent with the lines we price).

Simplifications (documented, not hidden):
  - Base-running advancement is probabilistic but coarse (no defense, steals,
    errors, wild pitches, double plays).
  - ER = all runs charged to the pitcher on the mound (no inherited-runner or
    earned/unearned accounting).
  - Batter rates don't shift when the bullpen enters (the hook only stops
    crediting the starter so his IP/K props stay realistic).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

# Per-PA event order used everywhere in this module.
EVENTS = ("K", "BB", "1B", "2B", "3B", "HR", "OUT")


def batter_pa_probs(b: dict) -> list[float]:
    """Per-PA multinomial [K, BB, 1B, 2B, 3B, HR, OUT] from a batter
    projection dict. Derived from the projected per-game counts ÷ expected PA,
    so it inherits the projection's opposing-pitcher/park/weather adjustments."""
    ePA = max(float(b.get("expected_pa") or 4.0), 1e-6)
    pHR = max(0.0, float(b.get("proj_hr") or 0.0) / ePA)
    p3B = max(0.0, float(b.get("proj_3b") or 0.0) / ePA)
    p2B = max(0.0, float(b.get("proj_2b") or 0.0) / ePA)
    pH = max(0.0, float(b.get("proj_h") or 0.0) / ePA)
    p1B = max(0.0, pH - pHR - p2B - p3B)
    pBB = max(0.0, float(b.get("proj_bb") or 0.0) / ePA)
    pK = max(0.0, float(b.get("proj_k") or 0.0) / ePA)
    on_base = pHR + p3B + p2B + p1B + pBB
    # Cap pathological inputs so OUT stays non-negative.
    if on_base + pK > 0.98:
        scale = 0.98 / (on_base + pK)
        pHR, p3B, p2B, p1B, pBB, pK = (x * scale for x in (pHR, p3B, p2B, p1B, pBB, pK))
    pOUT = max(0.0, 1.0 - (pHR + p3B + p2B + p1B + pBB + pK))
    probs = [pK, pBB, p1B, p2B, p3B, pHR, pOUT]
    s = sum(probs)
    return [x / s for x in probs]


# ---------- baserunning / bullpen realism ----------
# League baselines (2025-26). Used when the caller doesn't supply team rates.
LG_GIDP_PER_OPP = 0.11    # P(double play) given runner on 1st and < 2 outs
LG_SB_ATTEMPT = 0.075     # P(steal attempt) per PA with runner on 1st, 2nd open
LG_SB_SUCCESS = 0.78      # success rate on the attempt
# Relievers are better per-inning than starters. Batter rates bake in the
# OPPOSING STARTER, so once the hook fires the offense must be scaled down.
# Fallback used when bullpen FIP isn't supplied.
LG_BP_OFFENSE_MULT = 0.94
# Blow-up hook: a starter who has been shelled gets pulled regardless of his
# out target (see _sp_target for the mean-preserving correction).
BLOWUP_ER = 5


def _bp_offense_mult(bp_fip, sp_fip) -> float:
    """How much to scale a lineup's on-base rates once the bullpen enters.
    Ratio of starter to bullpen FIP: a better (lower-FIP) pen suppresses more.
    Clamped so a noisy FIP can't distort the game."""
    try:
        bp, sp = float(bp_fip), float(sp_fip)
    except (TypeError, ValueError):
        return LG_BP_OFFENSE_MULT
    if not (bp > 0 and sp > 0):
        return LG_BP_OFFENSE_MULT
    return max(0.80, min(1.15, bp / sp))


# ---------- per-game rate uncertainty (over-dispersion) ----------
# The sim used to treat each batter's projected rates as KNOWN, so it only
# reproduced sampling (binomial) variance across ~4 PA. Reality adds projection
# error: we don't know a hitter's true rate for tonight. Measured on 3,101
# graded sim picks, that made the sim 10-20pp overconfident everywhere between
# 55% and 90% (batter counting props worst: TB -16pp, runs -17pp, hits -22pp),
# while 95%+ stayed accurate and pitcher props — which aggregate over ~25
# batters faced — were fine.
#
# Fix: before each simulated game, multiply every batter's on-base rates by a
# draw from a Gamma(mean 1, var RATE_SIGMA^2). Mixing Poisson-ish counts over a
# Gamma rate is exactly the Negative-Binomial construction the PRICING path
# already uses via the empirically fitted dispersion in dispersion.json, so the
# sim now widens the same way its own NegBin pricing does. Mean is preserved
# (E[g]=1), so the team-runs anchor is unaffected.
# Tuned by scripts/sim_calibration_backtest.py (40 games, ~13k graded predicted
# overs vs real boxscores). Weighted mean |calibration gap| by sigma:
#
#   BEFORE the mechanics fixes (no bullpen split, no GIDP/SB, no walk-off stop,
#   non-convergent anchor):  0.00 4.33pp | 0.35 3.22 | 0.50 2.04 | 0.60 1.88
#   AFTER:                   0.00 2.07pp | 0.10 1.80 | 0.18 1.77 | 0.25 2.33
#                            0.40 2.58   | 0.50 2.84 | 0.60 4.13
#
# Two readings. (1) The structural fixes did the real work: at sigma=0 the error
# halved (4.33 -> 2.07), so the old large sigma was mostly PAPERING OVER missing
# baseball, not modelling genuine rate uncertainty. (2) The residual uncertainty
# is small — the basin is flat from 0.10-0.18, so we take its midpoint rather
# than the exact argmin (40-game window, correlated rows -> overfitting risk).
# Re-run the script after any change to the simulation mechanics.
RATE_SIGMA = 0.15
_RATE_VARIANTS = 32          # discrete draws approximating the Gamma


def _rate_multipliers(sigma: float, k: int = _RATE_VARIANTS,
                      seed: int = 12345) -> list[float]:
    """`k` Gamma draws with mean 1 and variance sigma^2 (shape 1/s^2,
    scale s^2). Precomputed once per game so the per-sim cost is one lookup."""
    if sigma <= 0:
        return [1.0]
    rng = random.Random(seed)
    shape = 1.0 / (sigma * sigma)
    scale = sigma * sigma
    vals = [rng.gammavariate(shape, scale) for _ in range(k)]
    m = sum(vals) / len(vals)
    # Renormalise so the sample mean is exactly 1 (keeps the anchor honest).
    return [v / m for v in vals] if m > 0 else [1.0]


def _scale_offense(probs: list[float], f: float) -> list[float]:
    """Scale the on-base events (BB,1B,2B,3B,HR) by f, absorb the change in
    OUT. f>1 => more offense. Keeps K roughly fixed (a zone/contact trait)."""
    if abs(f - 1.0) < 1e-6:
        return probs
    pK, pBB, p1B, p2B, p3B, pHR, pOUT = probs
    pBB, p1B, p2B, p3B, pHR = (min(x * f, x * 3) for x in (pBB, p1B, p2B, p3B, pHR))
    tot = pK + pBB + p1B + p2B + p3B + pHR
    if tot >= 0.99:
        s = 0.99 / tot
        pK, pBB, p1B, p2B, p3B, pHR = (x * s for x in (pK, pBB, p1B, p2B, p3B, pHR))
        tot = pK + pBB + p1B + p2B + p3B + pHR
    return [pK, pBB, p1B, p2B, p3B, pHR, max(0.0, 1.0 - tot)]


def _cum(probs: list[float]) -> list[float]:
    c, acc = [], 0.0
    for p in probs:
        acc += p
        c.append(acc)
    return c


def _draw(cum: list[float], r: float) -> int:
    for i, c in enumerate(cum):
        if r <= c:
            return i
    return len(cum) - 1


@dataclass
class _BoxLine:
    pa: int = 0
    h: int = 0
    hr: int = 0
    tb: int = 0
    rbi: int = 0
    r: int = 0
    k: int = 0
    bb: int = 0


@dataclass
class _PitLine:
    outs: int = 0
    k: int = 0
    bb: int = 0
    h: int = 0
    hr: int = 0
    er: int = 0


def _advance(bases, bi, ev, outs, rng, gidp_p: float = LG_GIDP_PER_OPP):
    """Apply one event. bases = [first, second, third] holding the runner's
    lineup index or None. Returns (scored_indices, new_bases, rbi, outs_added)."""
    f, s, t = bases
    scored, rbi, oa = [], 0, 0
    if ev == "HR":
        scored = [r for r in (t, s, f) if r is not None] + [bi]
        rbi = len(scored)
        bases = [None, None, None]
    elif ev == "3B":
        scored = [r for r in (t, s, f) if r is not None]
        rbi = len(scored)
        bases = [None, None, bi]
    elif ev == "2B":
        if t is not None:
            scored.append(t)
        if s is not None:
            scored.append(s)
        new_t = None
        if f is not None:
            if rng.random() < 0.45:
                scored.append(f)
            else:
                new_t = f
        rbi = len(scored)
        bases = [None, bi, new_t]
    elif ev == "1B":
        if t is not None:
            scored.append(t)
        new_t = new_s = None
        if s is not None:
            if rng.random() < 0.55:
                scored.append(s)
            else:
                new_t = s
        if f is not None:
            if rng.random() < 0.30 and new_t is None:
                new_t = f
            else:
                new_s = f
        rbi = len(scored)
        bases = [bi, new_s, new_t]
    elif ev == "BB":
        if f is None:
            bases = [bi, s, t]
        elif s is None:
            bases = [bi, f, t]
        elif t is None:
            bases = [bi, f, s]
        else:                       # bases loaded: force in the run from 3rd
            scored.append(t)
            rbi = 1
            bases = [bi, f, s]
    elif ev == "K":
        oa = 1
    else:                           # in-play OUT
        oa = 1
        # Double play: runner on 1st with < 2 outs. Without this the sim never
        # ended an inning on one swing, so innings ran long and offense was
        # systematically inflated.
        if f is not None and outs < 2 and rng.random() < gidp_p:
            oa = 2
            bases = [None, s, t]    # lead runner(s) hold, batter+R1 erased
            return scored, bases, rbi, oa
        if outs + 1 < 3:            # productive out only when it's not the 3rd
            if t is not None and rng.random() < 0.30:
                scored.append(t)
                rbi += 1
                t = None
            if s is not None and t is None and rng.random() < 0.20:
                t, s = s, None
            bases = [f, s, t]
    return scored, bases, rbi, oa


def _try_steal(bases, outs, rng, attempt_p: float, success_p: float):
    """Runner on 1st with 2nd open may attempt a steal before the pitch.
    Returns (new_bases, outs_added). Modelled as a discrete event so speed
    actually converts singles into scoring position (previously impossible)."""
    f, s, t = bases
    if f is None or s is not None or outs >= 2:
        return bases, 0
    if rng.random() >= attempt_p:
        return bases, 0
    if rng.random() < success_p:
        return [None, f, t], 0      # safe at 2nd
    return [None, None, t], 1       # caught stealing


def _record_offense(box: list[_BoxLine], idx: int, ev: str):
    bl = box[idx]
    bl.pa += 1
    if ev == "K":
        bl.k += 1
    elif ev == "BB":
        bl.bb += 1
    elif ev in ("1B", "2B", "3B", "HR"):
        bl.h += 1
        bl.tb += {"1B": 1, "2B": 2, "3B": 3, "HR": 4}[ev]
        if ev == "HR":
            bl.hr += 1


def _half_inning(cums, box, pit_box, start_idx, mound, rng,
                 cums_bp=None, gidp_p: float = LG_GIDP_PER_OPP,
                 sb_attempt: float = LG_SB_ATTEMPT,
                 sb_success: float = LG_SB_SUCCESS,
                 walkoff_need: int | None = None,
                 ghost_runner: bool = False):
    """Play one half-inning. Returns (runs, next_batter_index).

    `cums_bp` are the SAME lineup's rates scaled for the opposing bullpen; they
    take over the moment the hook fires (previously relievers were modelled as
    a statistical clone of the starter). `walkoff_need`, when set, ends the
    half-inning the instant the batting team scores that many runs (a real
    walk-off stops play; simulating the full frame handed home hitters PAs
    that never happen). `ghost_runner` starts extras with a man on 2nd."""
    outs = 0
    # Ghost runner is the man who made the last out = the slot before this one.
    bases = ([None, None, (start_idx - 1) % 9] if ghost_runner
             else [None, None, None])
    bi = start_idx
    runs = 0
    while outs < 3:
        slot = bi % 9
        # Steal attempt before the pitch.
        bases, sb_out = _try_steal(bases, outs, rng, sb_attempt, sb_success)
        if sb_out:
            outs += sb_out
            pit_box[mound[0]].outs += sb_out
            if outs >= 3:
                break
        active = cums_bp if (cums_bp is not None and mound[0] == "BP") else cums
        ev = EVENTS[_draw(active[slot], rng.random())]
        _record_offense(box, slot, ev)
        scored, bases, rbi, oa = _advance(bases, slot, ev, outs, rng, gidp_p)
        outs += oa
        runs += len(scored)
        for sc in scored:
            box[sc].r += 1
        box[slot].rbi += rbi
        # credit the pitcher on the mound
        pl = pit_box[mound[0]]
        pl.outs += oa
        if ev == "K":
            pl.k += 1
        elif ev == "BB":
            pl.bb += 1
        elif ev in ("1B", "2B", "3B", "HR"):
            pl.h += 1
            if ev == "HR":
                pl.hr += 1
        pl.er += len(scored)
        # Hook: out target reached OR the start has blown up. The blow-up exit
        # adds the left tail real starts have (a shelled pitcher is pulled, he
        # doesn't finish his pitch count). It does NOT double-count the value
        # engine's short-start guard: that guard filters BETS using the
        # PROJECTED expected_outs, while this shapes the WITHIN-GAME
        # distribution. _sp_target compensates the mean so projected outs are
        # still hit on average — only the spread changes.
        if mound[0] == "SP" and (pl.outs >= mound[1] or pl.er >= BLOWUP_ER):
            mound[0] = "BP"
        bi += 1
        # Walk-off: play stops the moment the home team takes the lead.
        if walkoff_need is not None and runs >= walkoff_need:
            break
    return runs, bi


def _simulate_one(cums_away, cums_home, sp_away_outs, sp_home_outs, rng,
                  bp_away=None, bp_home=None, ctx: dict | None = None):
    """One full game. Returns (away_box, home_box, pit_away, pit_home,
    away_runs, home_runs)."""
    box_a = [_BoxLine() for _ in range(9)]
    box_h = [_BoxLine() for _ in range(9)]
    pit_a = {"SP": _PitLine(), "BP": _PitLine()}   # away pitchers (face home)
    pit_h = {"SP": _PitLine(), "BP": _PitLine()}   # home pitchers (face away)
    mound_a = ["SP", sp_away_outs]   # away staff target
    mound_h = ["SP", sp_home_outs]
    idx_a = idx_h = 0
    runs_a = runs_h = 0
    inning = 0
    c = ctx or {}
    ga, gh = c.get("gidp_away", LG_GIDP_PER_OPP), c.get("gidp_home", LG_GIDP_PER_OPP)
    sa, sh = c.get("sb_away", LG_SB_ATTEMPT), c.get("sb_home", LG_SB_ATTEMPT)
    while True:
        inning += 1
        ghost = inning >= 10        # extras start a runner on 2nd (2020 rule)
        # top: away bats, faces home pitching
        r, idx_a = _half_inning(cums_away, box_a, pit_h, idx_a, mound_h, rng,
                                cums_bp=bp_away, gidp_p=ga, sb_attempt=sa,
                                ghost_runner=ghost)
        runs_a += r
        # walk-off skip: home leading after the top of the 9th+ doesn't bat
        if inning >= 9 and runs_h > runs_a:
            break
        # bottom: home bats, faces away pitching. From the 9th on, play stops
        # the instant the home team goes ahead (a real walk-off).
        need = (runs_a - runs_h + 1) if inning >= 9 else None
        r, idx_h = _half_inning(cums_home, box_h, pit_a, idx_h, mound_a, rng,
                                cums_bp=bp_home, gidp_p=gh, sb_attempt=sh,
                                walkoff_need=need, ghost_runner=ghost)
        runs_h += r
        if inning >= 9 and runs_h != runs_a:
            break
        if inning >= 15:             # safety cap on extras
            break
    return box_a, box_h, pit_a, pit_h, runs_a, runs_h


def _mean_runs(cums_a, cums_h, spa, sph, n, rng,
               bp_away=None, bp_home=None, ctx: dict | None = None):
    """Bottom-up mean runs. MUST be run with the same mechanics (bullpen, GIDP,
    steals, walk-offs) as the headline sim — it sets the anchor factor, so any
    mechanic missing here would be double-counted as an anchor correction."""
    ta = th = 0
    for _ in range(n):
        _, _, _, _, ra, rh = _simulate_one(cums_a, cums_h, spa, sph, rng,
                                           bp_away=bp_away, bp_home=bp_home,
                                           ctx=ctx)
        ta += ra
        th += rh
    return ta / n, th / n


@dataclass
class SimResult:
    n: int
    away_team: str
    home_team: str
    box_away: list[dict]
    box_home: list[dict]
    pit_away: dict
    pit_home: dict
    # team totals
    glm_away: float
    glm_home: float
    free_away: float          # bottom-up (unanchored) mean
    free_home: float
    anchored_away: float      # achieved anchored mean (≈ glm)
    anchored_home: float
    anchor_f_away: float
    anchor_f_home: float
    p_home_win: float
    mean_total: float
    score_dist: dict          # "h-a" -> count for the modal line
    total_hist: dict          # total runs -> probability
    # Distributions for the simulation leaderboard — exact COUNTS over the n
    # sims so any line's hit rate is sum(count where outcome wins) / n.
    bat_hist: dict = field(default_factory=dict)   # {player_id: {stat: {value: count}}}
    pit_hist: dict = field(default_factory=dict)   # {player_id: {stat: {value: count}}}
    margin_counts: dict = field(default_factory=dict)  # {home_margin: count}
    total_counts: dict = field(default_factory=dict)   # {total runs: count}


# Mean-preserving correction for the blow-up hook. Some starts now end early on
# runs allowed, which would drag simulated mean outs BELOW the projection and
# bias every pitcher_outs / K prop low. Nudging the drawn target up by this
# factor restores the mean, so the hook only adds left-tail SHAPE — it does not
# re-litigate the projection (which the value engine's short-start guard
# already acts on separately). Measured: ~6% of starts hit the blow-up exit.
_BLOWUP_MEAN_ADJ = 1.06


def _sp_target(starter: dict, rng: random.Random) -> int:
    base = float((starter or {}).get("expected_outs") or 15.0) * _BLOWUP_MEAN_ADJ
    return max(6, min(24, int(round(rng.gauss(base, 3.0)))))


# ---------- Simulation leaderboard ----------
# market -> (kind, stat) where kind is "bat" or "pit".
_LB_PROP_MAP = {
    "prop_hits": ("bat", "h"), "prop_hr": ("bat", "hr"), "prop_tb": ("bat", "tb"),
    "prop_rbi": ("bat", "rbi"), "prop_runs": ("bat", "r"), "prop_k": ("bat", "k"),
    "prop_bb": ("bat", "bb"), "prop_hrr": ("bat", "hrr"),
    "prop_pitcher_k": ("pit", "k"), "prop_pitcher_bb": ("pit", "bb"),
    "prop_pitcher_h": ("pit", "h"), "prop_pitcher_hr": ("pit", "hr"),
    "prop_pitcher_er": ("pit", "er"), "prop_pitcher_outs": ("pit", "outs"),
}


def _hist_over(hist_stat: dict, line: float) -> int:
    return sum(c for v, c in hist_stat.items() if float(v) > line)


def _hist_under(hist_stat: dict, line: float) -> int:
    return sum(c for v, c in hist_stat.items() if float(v) < line)


def _side_of(desc: str) -> str:
    if " OVER " in desc:
        return "OVER"
    if " UNDER " in desc:
        return "UNDER"
    return ""


def bet_sim_hitrate(res: SimResult, bet: dict) -> float | None:
    """Fraction of the sim's `n` games in which `bet` would WIN. None if the
    bet can't be mapped to a simulated distribution. Pushes count as non-hits."""
    n = res.n
    if not n:
        return None
    market = bet.get("market", "")
    desc = bet.get("description", "") or ""
    try:
        line = float(bet.get("line") or 0)
    except (TypeError, ValueError):
        return None

    # --- player props ---
    if market in _LB_PROP_MAP:
        kind, stat = _LB_PROP_MAP[market]
        pid = bet.get("player_id")
        if pid is None:
            return None
        store = res.bat_hist if kind == "bat" else res.pit_hist
        try:
            hist = store.get(int(pid))
        except (TypeError, ValueError):
            return None
        if not hist or stat not in hist:
            return None
        side = _side_of(desc)
        if side == "OVER":
            return _hist_over(hist[stat], line) / n
        if side == "UNDER":
            return _hist_under(hist[stat], line) / n
        return None

    # --- game lines, scored off the home-margin / total distributions ---
    home, away = res.home_team, res.away_team
    if market in ("moneyline", "sharp_moneyline"):
        wins = 0
        for m, c in res.margin_counts.items():
            if (home in desc and m > 0) or (away in desc and m < 0):
                wins += c
            elif m == 0:
                wins += c * 0.5     # extra-innings coin flip
        return wins / n
    if market in ("total", "sharp_total"):
        side = "OVER" if " Over " in desc else ("UNDER" if " Under " in desc else "")
        if not side:
            return None
        if side == "OVER":
            return sum(c for t, c in res.total_counts.items() if t > line) / n
        return sum(c for t, c in res.total_counts.items() if t < line) / n
    if market in ("run_line", "sharp_run_line"):
        # description: "<team> <signed spread>" e.g. "NYY -1.5" / "BOS +1.5"
        import re
        m = re.search(r"([+-]\d+(?:\.\d+)?)", desc)
        if not m:
            return None
        spread = float(m.group(1))
        team_is_home = home in desc and (away not in desc or desc.index(home) < desc.index(away))
        wins = 0
        for mg, c in res.margin_counts.items():
            t_margin = mg if team_is_home else -mg
            if t_margin + spread > 0:
                wins += c
        return wins / n
    return None


# Odds worse (more negative) than this are excluded from the leaderboard —
# heavy chalk hits most often by construction and floods the "most times hit"
# board (especially fine-tuned pitcher props) with unbettable juice.
MIN_ODDS = -400


def _build_sim_rows(games: list, sim_for_game, offered_bets: list[dict],
                    min_odds: float | None = MIN_ODDS) -> list[dict]:
    """Score every offered prop / game line by its simulated hit rate, deduped
    to one row per (game, player/market/side, line). Returns ALL rows
    (unsorted, untruncated) — callers slice/group as needed. Bets priced worse
    than `min_odds` (-400 by default) are dropped; pass min_odds=None to keep
    every offer (used when the odds feed is degraded/rate-limited so the board
    isn't thinned further)."""
    sims: dict = {}
    _labels: dict = {}
    rows: list[dict] = []
    seen: set = set()
    for b in offered_bets:
        gpk = b.get("game_pk")
        if gpk is None:
            continue
        if min_odds is not None:
            try:
                _o = float(b.get("odds") or 0)
            except (TypeError, ValueError):
                _o = 0.0
            if _o < 0 and _o < min_odds:  # e.g. -450, -500 — heavier than -400
                continue
        if gpk not in sims:
            gp = next((g for g in games if g.game_pk == gpk), None)
            sims[gpk] = sim_for_game(gp) if gp is not None else None
            _labels[gpk] = getattr(gp, "matchup_label", "") if gp is not None else ""
        res = sims[gpk]
        if res is None or isinstance(res, str):
            continue
        _matchup = _labels.get(gpk) or f"{res.away_team} @ {res.home_team}"
        hr = bet_sim_hitrate(res, b)
        if hr is None:
            continue
        key = (gpk, b.get("player_id"), b.get("market"),
               _side_of(b.get("description", "")) or b.get("description", ""),
               round(float(b.get("line") or 0), 1))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "matchup": _matchup,
            "description": b.get("description", ""),
            "market": b.get("market", ""),
            "odds": b.get("odds", 0),
            "line": b.get("line"),
            "side": _side_of(b.get("description", "")),
            "game_pk": gpk,
            "player_id": b.get("player_id"),
            "sim_hit": round(hr, 4),
            "sim_hits_n": int(round(hr * res.n)),
            "n": res.n,
            "novig_prob": b.get("novig_prob"),
        })
    return rows


def build_sim_leaderboard(games: list, sim_for_game, offered_bets: list[dict],
                          top: int = 20,
                          min_odds: float | None = MIN_ODDS) -> list[dict]:
    """Rank every offered prop / game line across the slate by its simulated
    hit rate. `sim_for_game(gp)->SimResult` runs (or caches) the sim for a
    game; `offered_bets` is the slate-wide pool (each dict has market,
    description, line, odds, game_pk, player_id). Returns the top `top` by
    hit rate, deduped to one row per (game, player/market/side, line).
    `min_odds=None` disables the -400 juice filter."""
    rows = _build_sim_rows(games, sim_for_game, offered_bets, min_odds=min_odds)
    rows.sort(key=lambda r: -r["sim_hit"])
    return rows[:top]


def build_sim_leaderboard_by_market(games: list, sim_for_game,
                                    offered_bets: list[dict],
                                    top: int = 20,
                                    min_odds: float | None = MIN_ODDS,
                                    ) -> dict[str, list[dict]]:
    """Same scoring as `build_sim_leaderboard`, but grouped by market (stat).
    Returns {market: top-N rows by sim hit count}, one entry per market that
    has at least one mapped offer. Each market's rows are independently ranked
    by simulated hit frequency and capped at `top`. `min_odds=None` disables
    the -400 juice filter (used when the odds feed is degraded)."""
    rows = _build_sim_rows(games, sim_for_game, offered_bets, min_odds=min_odds)
    by_mkt: dict[str, list[dict]] = {}
    for r in rows:
        by_mkt.setdefault(r["market"], []).append(r)
    for mkt in by_mkt:
        by_mkt[mkt].sort(key=lambda r: -r["sim_hit"])
        by_mkt[mkt] = by_mkt[mkt][:top]
    return by_mkt


def build_sim_boards(games: list, sim_for_game, offered_bets: list[dict],
                     top: int = 20, min_odds: float | None = MIN_ODDS,
                     hi_threshold: float = 0.95) -> dict:
    """Run the sim once and return every leaderboard view the app needs:
      {"by_market": {market: top-N rows},
       "high_conf": [rows >= hi_threshold],
       "all_rows":  [every scored row, best first]}
    `high_conf` is the cross-stat pool of the most confident offers (default
    sim hit >= 95%), and `all_rows` is the complete day's board — both sorted
    by hit rate descending and NOT truncated per market. Building every view
    from one row-scoring pass avoids simulating the slate more than once."""
    rows = _build_sim_rows(games, sim_for_game, offered_bets, min_odds=min_odds)
    return _group_rows_into_boards(rows, top=top, hi_threshold=hi_threshold)


def _group_rows_into_boards(rows: list[dict], top: int = 20,
                            hi_threshold: float = 0.95) -> dict:
    """Group scored leaderboard rows into the app's three views (per-market
    top-N, cross-stat high-confidence, and the full board)."""
    by_mkt: dict[str, list[dict]] = {}
    for r in rows:
        by_mkt.setdefault(r["market"], []).append(r)
    for mkt in by_mkt:
        by_mkt[mkt].sort(key=lambda r: -r["sim_hit"])
        by_mkt[mkt] = by_mkt[mkt][:top]
    all_rows = sorted(rows, key=lambda r: -r["sim_hit"])
    high_conf = [r for r in all_rows if r["sim_hit"] >= hi_threshold]
    return {"by_market": by_mkt, "high_conf": high_conf, "all_rows": all_rows}


# stat -> (internal market, display label) for model-predicted props.
_PRED_BAT = {
    "h": ("prop_hits", "Hits"), "hr": ("prop_hr", "HR"), "tb": ("prop_tb", "TB"),
    "rbi": ("prop_rbi", "RBI"), "r": ("prop_runs", "Runs"), "k": ("prop_k", "K"),
    "bb": ("prop_bb", "BB"), "hrr": ("prop_hrr", "H+R+RBI"),
}
_PRED_PIT = {
    "k": ("prop_pitcher_k", "Pitcher K"), "bb": ("prop_pitcher_bb", "Pitcher BB"),
    "h": ("prop_pitcher_h", "Pitcher H"), "hr": ("prop_pitcher_hr", "Pitcher HR"),
    "er": ("prop_pitcher_er", "Pitcher ER"), "outs": ("prop_pitcher_outs", "Outs"),
}


def _over_line(mean: float) -> float:
    """The half-integer just BELOW the projected mean (floored at 0.5) — the
    line an OVER is favored to clear. For sub-0.5 means (rare events like HR)
    it floors at 0.5, so the over is a real long-shot prop that ranks low."""
    import math
    half = math.floor(mean - 0.5) + 0.5
    return max(0.5, half)


def _hist_mean(hist_stat: dict) -> float:
    tot = sum(hist_stat.values())
    if not tot:
        return 0.0
    return sum(float(v) * c for v, c in hist_stat.items()) / tot


def build_predicted_prop_rows(games: list, sim_for_game) -> list[dict]:
    """Synthesize a leaderboard row for EVERY player prop the model predicts —
    no book offer required. OVERS ONLY: for each player/stat the line is the
    half-integer just below the simulated mean and sim_hit is P(over). Used
    when the odds feed is rate-limited so the board still shows all predicted
    OVER props ranked by the model's own confidence. Odds are None (irrelevant
    here)."""
    rows: list[dict] = []
    for g in games:
        res = sim_for_game(g)
        if res is None or isinstance(res, str):
            continue
        matchup = getattr(g, "matchup_label", "") or \
            f"{getattr(g, 'away_team', '?')} @ {getattr(g, 'home_team', '?')}"
        gpk = getattr(g, "game_pk", None)
        # player_id -> name from the game's batter/starter dicts
        names: dict = {}
        for b in (getattr(g, "away_batters", []) or []) + \
                 (getattr(g, "home_batters", []) or []):
            if b.get("player_id") is not None:
                names[int(b["player_id"])] = b.get("name", "?")
        for sp in (getattr(g, "away_starter", None), getattr(g, "home_starter", None)):
            if sp and sp.get("player_id") is not None:
                names[int(sp["player_id"])] = sp.get("name", "?")

        def _emit(store, statmap):
            for pid, hist in store.items():
                name = names.get(int(pid), "?")
                for stat, (market, label) in statmap.items():
                    h = hist.get(stat)
                    if not h or res.n <= 0:
                        continue
                    mean = _hist_mean(h)
                    # OVERS ONLY, across the standard line ladder (0.5, 1.5, …)
                    # up to one line past the projection. The low lines are where
                    # the confident overs live (over 0.5 = "records the stat");
                    # the near-mean lines are the coin-flips. One row per line so
                    # the high-confidence overs surface at the top of the board.
                    line = 0.5
                    while line <= mean + 1.0:
                        p = _hist_over(h, line) / res.n
                        if p > 0:
                            rows.append({
                                "matchup": matchup,
                                "description": f"{name} OVER {line:g} {label}",
                                "market": market, "odds": None, "line": line,
                                "side": "OVER", "game_pk": gpk, "player_id": int(pid),
                                "sim_hit": round(p, 4),
                                "sim_hits_n": int(round(p * res.n)),
                                "n": res.n, "novig_prob": None,
                            })
                        line += 1.0
        _emit(res.bat_hist, _PRED_BAT)
        _emit(res.pit_hist, _PRED_PIT)
    return rows


def build_tb_leaderboard(games: list, sim_for_game,
                         offered_bets: list[dict] | None = None) -> list[dict]:
    """One row per batter with the sim's P(TB >= 1) and P(TB >= 2).

    These are the two lines the book actually hangs for total bases (over 0.5
    and over 1.5) and TB is the model's best-performing market. Rows cover
    EVERY projected batter — not just those with an offer — so the board is
    complete; book odds are attached per line when an offer exists.
    """
    # (game_pk, player_id, line) -> American odds for the OVER
    odds_map: dict = {}
    for b in (offered_bets or []):
        if b.get("market") != "prop_tb" or " OVER " not in (b.get("description") or ""):
            continue
        try:
            key = (b.get("game_pk"), int(b.get("player_id")),
                   round(float(b.get("line")), 1))
        except (TypeError, ValueError):
            continue
        if key not in odds_map:
            odds_map[key] = b.get("odds")

    rows: list[dict] = []
    for g in games:
        res = sim_for_game(g)
        if res is None or isinstance(res, str):
            continue
        matchup = getattr(g, "matchup_label", "") or \
            f"{getattr(g, 'away_team', '?')} @ {getattr(g, 'home_team', '?')}"
        gpk = getattr(g, "game_pk", None)
        names: dict = {}
        for side, team in ((getattr(g, "away_batters", []) or [],
                            getattr(g, "away_team", "")),
                           (getattr(g, "home_batters", []) or [],
                            getattr(g, "home_team", ""))):
            for b in side:
                if b.get("player_id") is not None:
                    names[int(b["player_id"])] = (b.get("name", "?"), team,
                                                  int(b.get("bat_order") or 0))
        for pid, hists in res.bat_hist.items():
            h = hists.get("tb")
            if not h or res.n <= 0:
                continue
            nm, team, order = names.get(int(pid), ("?", "", 0))
            p1 = _hist_over(h, 0.5) / res.n
            p2 = _hist_over(h, 1.5) / res.n
            rows.append({
                "matchup": matchup, "game_pk": gpk, "player_id": int(pid),
                "player": nm, "team": team, "order": order,
                "p_1tb": round(p1, 4), "p_2tb": round(p2, 4),
                "hits_1tb": int(round(p1 * res.n)),
                "hits_2tb": int(round(p2 * res.n)),
                "odds_1tb": odds_map.get((gpk, int(pid), 0.5)),
                "odds_2tb": odds_map.get((gpk, int(pid), 1.5)),
                "n": res.n,
            })
    return rows


def build_predicted_prop_boards(games: list, sim_for_game, top: int = 20,
                                hi_threshold: float = 0.95) -> dict:
    """Predicted-prop counterpart of build_sim_boards — same {by_market,
    high_conf, all_rows} shape, built from the model's own projections."""
    rows = build_predicted_prop_rows(games, sim_for_game)
    return _group_rows_into_boards(rows, top=top, hi_threshold=hi_threshold)


# ---------- Correlated same-game parlays (true joint from the sim) ----------
# Marginal histograms can't tell you how often two legs hit in the SAME game.
# For that we re-run the sim tracking each candidate leg's per-sim boolean and
# accumulate pairwise co-occurrence, so the joint probability reflects the real
# in-game correlation (a player's hits and his team scoring move together; a
# strikeout prop and the opponent's total move opposite). Books that price a
# same-game parlay by multiplying the legs ignore this — that gap is the edge.

def _american_to_decimal(o) -> float | None:
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if not o:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def _leg_predicate(gp: dict, leg: dict, bat_index: dict,
                   away_sp_id, home_sp_id):
    """Compile a leg (offered bet dict) into a fast per-sim boolean test
    `pred(ba, bh, pa, ph, ra, rh) -> bool`. Returns None if unmappable."""
    market = leg.get("market", "")
    desc = leg.get("description", "") or ""
    try:
        line = float(leg.get("line") or 0)
    except (TypeError, ValueError):
        return None
    side = _side_of(desc)

    if market in _LB_PROP_MAP:
        kind, stat = _LB_PROP_MAP[market]
        pid = leg.get("player_id")
        if pid is None or side not in ("OVER", "UNDER"):
            return None
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None

        def _val_bat(box, j):
            bl = box[j]
            return (bl.h + bl.r + bl.rbi) if stat == "hrr" else getattr(bl, stat)

        if kind == "bat":
            loc = bat_index.get(pid)
            if loc is None:
                return None
            which, j = loc

            def pred(ba, bh, pa, ph, ra, rh, _which=which, _j=j):
                v = _val_bat(ba if _which == "a" else bh, _j)
                return v > line if side == "OVER" else v < line
            return pred
        else:  # pitcher
            if pid == away_sp_id:
                who = "a"
            elif pid == home_sp_id:
                who = "h"
            else:
                return None

            def pred(ba, bh, pa, ph, ra, rh, _who=who):
                sp = (pa if _who == "a" else ph)["SP"]
                v = getattr(sp, stat)
                return v > line if side == "OVER" else v < line
            return pred

    home, away = gp.get("home_team", ""), gp.get("away_team", "")
    if market in ("moneyline", "sharp_moneyline"):
        if home and home in desc:
            return lambda ba, bh, pa, ph, ra, rh: rh > ra
        if away and away in desc:
            return lambda ba, bh, pa, ph, ra, rh: ra > rh
        return None
    if market in ("total", "sharp_total"):
        if " Over " in desc:
            return lambda ba, bh, pa, ph, ra, rh: (ra + rh) > line
        if " Under " in desc:
            return lambda ba, bh, pa, ph, ra, rh: (ra + rh) < line
        return None
    if market in ("run_line", "sharp_run_line"):
        import re
        m = re.search(r"([+-]\d+(?:\.\d+)?)", desc)
        if not m:
            return None
        spread = float(m.group(1))
        team_is_home = home in desc and (away not in desc
                                         or desc.index(home) < desc.index(away))

        def pred(ba, bh, pa, ph, ra, rh):
            margin = (rh - ra) if team_is_home else (ra - rh)
            return margin + spread > 0
        return pred
    return None


def simulate_joint(gp: dict, legs: list[dict], n: int = 4000, seed: int = 0) -> dict:
    """Re-simulate one game tracking `legs`' per-sim outcomes and return their
    marginal and pairwise-joint hit rates. `legs` should already be the small
    candidate set for the game (dedupe / cap upstream). Returns
    {"legs":[{leg, p}], "pairs":[{i,j,p_i,p_j,p_joint,indep,lift,parlay_dec,ev}]}."""
    bat_a = (gp.get("away_batters") or [])[:9]
    bat_h = (gp.get("home_batters") or [])[:9]
    if len(bat_a) < 9 or len(bat_h) < 9:
        return {"legs": [], "pairs": []}
    bat_index: dict = {}
    for j, b in enumerate(bat_a):
        if b.get("player_id") is not None:
            bat_index[int(b["player_id"])] = ("a", j)
    for j, b in enumerate(bat_h):
        if b.get("player_id") is not None:
            bat_index[int(b["player_id"])] = ("h", j)
    away_sp_id = gp.get("away_sp_id")
    home_sp_id = gp.get("home_sp_id")

    preds, kept = [], []
    for leg in legs:
        p = _leg_predicate(gp, leg, bat_index, away_sp_id, home_sp_id)
        if p is not None:
            preds.append(p)
            kept.append(leg)
    L = len(preds)
    if L < 2:
        return {"legs": [], "pairs": []}

    rng = random.Random(seed)
    probs_a = [batter_pa_probs(b) for b in bat_a]
    probs_h = [batter_pa_probs(b) for b in bat_h]
    glm_a = float(gp.get("pred_away_runs") or 0.0)
    glm_h = float(gp.get("pred_home_runs") or 0.0)
    pilot = max(150, n // 5)
    cums_a0 = [_cum(p) for p in probs_a]
    cums_h0 = [_cum(p) for p in probs_h]
    spa = _sp_target(gp.get("away_starter"), rng)
    sph = _sp_target(gp.get("home_starter"), rng)
    free_a, free_h = _mean_runs(cums_a0, cums_h0, spa, sph, pilot,
                                random.Random(seed + 1))
    fa = fh = 1.0
    if free_a > 0.3 and glm_a > 0:
        fa = min(1.8, max(0.55, glm_a / free_a))
    if free_h > 0.3 and glm_h > 0:
        fh = min(1.8, max(0.55, glm_h / free_h))
    cums_a = [_cum(_scale_offense(p, fa)) for p in probs_a]
    cums_h = [_cum(_scale_offense(p, fh)) for p in probs_h]

    win = [0] * L
    cowin = [[0] * L for _ in range(L)]
    for _ in range(n):
        sa = _sp_target(gp.get("away_starter"), rng)
        sh = _sp_target(gp.get("home_starter"), rng)
        ba, bh, pa, ph, ra, rh = _simulate_one(cums_a, cums_h, sa, sh, rng)
        hit = [pr(ba, bh, pa, ph, ra, rh) for pr in preds]
        for a in range(L):
            if hit[a]:
                win[a] += 1
                for b in range(a + 1, L):
                    if hit[b]:
                        cowin[a][b] += 1

    legs_out = [{"leg": kept[k], "p": win[k] / n} for k in range(L)]
    pairs = []
    for a in range(L):
        pa_ = win[a] / n
        for b in range(a + 1, L):
            pb_ = win[b] / n
            pj = cowin[a][b] / n
            indep = pa_ * pb_
            da = _american_to_decimal(kept[a].get("odds"))
            db = _american_to_decimal(kept[b].get("odds"))
            parlay_dec = da * db if (da and db) else None
            ev = (pj * parlay_dec - 1.0) if parlay_dec else None
            pairs.append({
                "i": a, "j": b, "p_i": pa_, "p_j": pb_, "p_joint": pj,
                "indep": indep, "lift": (pj / indep if indep > 0 else None),
                "parlay_dec": parlay_dec, "ev": ev,
            })
    return {"legs": legs_out, "pairs": pairs}


def find_correlated_parlays(games: list, sim_joint_for_game, all_rows: list[dict],
                            cand_lo: float = 0.55, cand_hi: float = 0.97,
                            max_legs_per_game: int = 20,
                            min_ev: float = 0.05, min_lift: float = 1.03,
                            top: int = 40) -> list[dict]:
    """Find +EV, positively-correlated 2-leg same-game parlays across the slate.

    `all_rows` are the marginal-scored board rows (from build_sim_boards) used
    only to pick each game's candidate legs (marginal hit in [cand_lo, cand_hi],
    capped to the `max_legs_per_game` most confident). `sim_joint_for_game(gp,
    legs)->dict` runs the joint sim. A pair is surfaced when its sim joint beats
    the independent product (lift >= min_lift) AND is +EV at the multiplied
    price (ev >= min_ev). Returned rows are sorted by EV descending.

    NOTE: EV uses the two legs' odds multiplied together. Real same-game-parlay
    payouts are usually LOWER (books apply their own correlation haircut), so
    treat EV as an optimistic screen, not a guarantee.
    """
    by_game: dict = {}
    for r in all_rows:
        by_game.setdefault(r.get("game_pk"), []).append(r)

    out: list[dict] = []
    for gpk, rows in by_game.items():
        cand = [r for r in rows if cand_lo <= r["sim_hit"] <= cand_hi]
        # one row per (player, market, side) — its best (most confident) line
        seen: dict = {}
        for r in sorted(cand, key=lambda r: -r["sim_hit"]):
            k = (r.get("player_id"), r["market"], r.get("side"))
            if k not in seen:
                seen[k] = r
        cand = list(seen.values())[:max_legs_per_game]
        if len(cand) < 2:
            continue
        gp = next((g for g in games if getattr(g, "game_pk", None) == gpk), None)
        if gp is None:
            continue
        res = sim_joint_for_game(gp, cand)
        legs_out = res.get("legs", [])
        for pr in res.get("pairs", []):
            if pr["ev"] is None or pr["lift"] is None:
                continue
            if pr["ev"] < min_ev or pr["lift"] < min_lift:
                continue
            li, lj = legs_out[pr["i"]]["leg"], legs_out[pr["j"]]["leg"]
            out.append({
                "matchup": li.get("matchup", ""),
                "leg1": li.get("description", ""), "leg2": lj.get("description", ""),
                "market1": li.get("market", ""), "market2": lj.get("market", ""),
                "odds1": li.get("odds"), "odds2": lj.get("odds"),
                "p_joint": pr["p_joint"], "indep": pr["indep"], "lift": pr["lift"],
                "parlay_dec": pr["parlay_dec"], "ev": pr["ev"],
            })
    out.sort(key=lambda r: -r["ev"])
    return out[:top]


def _dec_to_american(dec: float) -> int:
    if dec <= 1:
        return 0
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def _ticket(legs: list[dict], boost: float) -> dict:
    dec = 1.0
    for l in legs:
        d = _american_to_decimal(l.get("odds"))
        dec *= d if d else 1.0
    games = {l.get("game_pk") for l in legs}
    return {
        "legs": legs, "decimal": round(dec, 3), "american": _dec_to_american(dec),
        "boost": boost, "same_game": len(games) == 1, "n_legs": len(legs),
        "combined_sim": round(_product(l.get("sim_hit") or 0 for l in legs), 4),
    }


def _product(vals) -> float:
    p = 1.0
    for v in vals:
        p *= float(v)
    return p


def build_daily_parlays(all_rows: list[dict], conf_core: float = 0.88,
                        conf_corr: float = 0.85, target_core: float = 5.0,
                        target_corr: float = 6.0, max_legs: int = 6) -> dict:
    """Assemble the day's sim board into the two-lane parlay strategy.

    Lane 'core'        — cross-game (one leg per game), legs >= conf_core,
                         greedily built to ~target_core decimal. The steady,
                         bookable-anywhere lane; the biggest ticket gets the
                         20% boost.
    Lane 'correlation' — same-game stacks, legs >= conf_corr, one ticket per
                         game built to ~target_corr. Harvests in-game
                         correlation; the biggest ticket gets the 10% boost.
                         Verify the real SGP price before betting these.
    Both lanes drop legs with no usable odds. `all_rows` is the marginal board
    from build_sim_boards (already -400-filtered when the feed is healthy)."""
    usable = [r for r in all_rows
              if _american_to_decimal(r.get("odds")) and (r.get("sim_hit") or 0) > 0]

    # ----- Lane A: cross-game core -----
    core_pool = [r for r in usable if (r.get("sim_hit") or 0) >= conf_core]
    best_by_game: dict = {}
    for r in sorted(core_pool, key=lambda r: -(r.get("sim_hit") or 0)):
        g = r.get("game_pk")
        if g not in best_by_game:
            best_by_game[g] = r
    core = sorted(best_by_game.values(), key=lambda r: -(r.get("sim_hit") or 0))
    core_tickets, cur, cur_dec = [], [], 1.0
    for r in core:
        cur.append(r)
        cur_dec *= _american_to_decimal(r["odds"])
        if cur_dec >= target_core or len(cur) >= max_legs:
            core_tickets.append(cur); cur, cur_dec = [], 1.0
    if len(cur) >= 2:
        core_tickets.append(cur)
    A = [_ticket(t, 0.20 if i == 0 else 0.0) for i, t in enumerate(core_tickets)]

    # ----- Lane B: same-game correlation -----
    corr_pool = [r for r in usable if (r.get("sim_hit") or 0) >= conf_corr]
    by_game: dict = {}
    for r in corr_pool:
        by_game.setdefault(r.get("game_pk"), []).append(r)
    B_raw = []
    for g, rows in by_game.items():
        seen: dict = {}
        for r in sorted(rows, key=lambda r: -(r.get("sim_hit") or 0)):
            k = (r.get("player_id"), r.get("market"), r.get("side"))
            if k not in seen:
                seen[k] = r
        legs, cur, cur_dec = [], [], 1.0
        for r in sorted(seen.values(), key=lambda r: -(r.get("sim_hit") or 0)):
            cur.append(r)
            cur_dec *= _american_to_decimal(r["odds"])
            if cur_dec >= target_corr or len(cur) >= max_legs:
                break
        if len(cur) >= 2:
            B_raw.append(cur)
    B_raw.sort(key=lambda t: -_product(_american_to_decimal(l["odds"]) for l in t))
    B = [_ticket(t, 0.10 if i == 0 else 0.0) for i, t in enumerate(B_raw)]

    return {"core": A, "correlation": B}


def simulate_game(gp: dict, n: int = 2000, seed: int = 0,
                  anchor: bool = True,
                  rate_sigma: float = RATE_SIGMA) -> SimResult:
    """Run `n` Monte Carlo games for one GamePrediction dict.

    Returns aggregated box-score distributions plus both the bottom-up and
    GLM team totals. The headline box score uses anchored rates (mean team
    runs scaled to the GLM projection); the free bottom-up mean is reported
    alongside so divergence is visible."""
    rng = random.Random(seed)
    bat_a = gp.get("away_batters") or []
    bat_h = gp.get("home_batters") or []
    if len(bat_a) < 9 or len(bat_h) < 9:
        raise ValueError("need 9 batters per side to simulate")
    bat_a, bat_h = bat_a[:9], bat_h[:9]
    probs_a = [batter_pa_probs(b) for b in bat_a]
    probs_h = [batter_pa_probs(b) for b in bat_h]
    glm_a = float(gp.get("pred_away_runs") or 0.0)
    glm_h = float(gp.get("pred_home_runs") or 0.0)

    # Bullpen scaling + baserunning context are needed BEFORE the pilot: the
    # pilot sets the anchor factor, so it has to run the same mechanics or the
    # anchor silently "corrects" for them and team totals collapse.
    _bp_a = _bp_offense_mult(gp.get("home_bp_fip"), gp.get("home_sp_fip"))
    _bp_h = _bp_offense_mult(gp.get("away_bp_fip"), gp.get("away_sp_fip"))
    _ctx = {
        "gidp_away": float(gp.get("away_gidp_p") or LG_GIDP_PER_OPP),
        "gidp_home": float(gp.get("home_gidp_p") or LG_GIDP_PER_OPP),
        "sb_away": float(gp.get("away_sb_p") or LG_SB_ATTEMPT),
        "sb_home": float(gp.get("home_sb_p") or LG_SB_ATTEMPT),
    }

    # Pilot (free rates) -> bottom-up mean for the "show both" comparison and
    # the anchor factors.
    pilot = max(150, n // 5)
    cums_a0 = [_cum(p) for p in probs_a]
    cums_h0 = [_cum(p) for p in probs_h]
    bp_a0 = [_cum(_scale_offense(p, _bp_a)) for p in probs_a]
    bp_h0 = [_cum(_scale_offense(p, _bp_h)) for p in probs_h]
    spa = _sp_target(gp.get("away_starter"), rng)
    sph = _sp_target(gp.get("home_starter"), rng)
    free_a, free_h = _mean_runs(cums_a0, cums_h0, spa, sph,
                                pilot, random.Random(seed + 1),
                                bp_away=bp_a0, bp_home=bp_h0, ctx=_ctx)

    # Anchor factors. Runs are SUPER-LINEAR in on-base rate (baserunners
    # compound), so a single f = glm/free systematically undershoots whenever
    # the correction is large. Iterate: apply f, re-measure, correct again.
    fa = fh = 1.0
    if anchor and free_a > 0.3 and glm_a > 0:
        fa = min(1.8, max(0.55, glm_a / free_a))
    if anchor and free_h > 0.3 and glm_h > 0:
        fh = min(1.8, max(0.55, glm_h / free_h))
    if anchor:
        # DAMPED update. Runs scale roughly as f^2 (baserunners compound), so
        # the naive f *= target/achieved overshoots and oscillates — measured
        # 1.00 -> 1.34 -> 0.95 -> 1.37 on a real game. Raising the ratio to
        # ~1/2 inverts the quadratic response and converges in 2-3 passes.
        _pilot_it = min(pilot, 600)
        for _it in range(4):
            ca_t = [_cum(_scale_offense(p, fa)) for p in probs_a]
            ch_t = [_cum(_scale_offense(p, fh)) for p in probs_h]
            bpa_t = [_cum(_scale_offense(p, fa * _bp_a)) for p in probs_a]
            bph_t = [_cum(_scale_offense(p, fh * _bp_h)) for p in probs_h]
            ga, gh_ = _mean_runs(ca_t, ch_t, spa, sph, _pilot_it,
                                 random.Random(seed + 17 + _it),
                                 bp_away=bpa_t, bp_home=bph_t, ctx=_ctx)
            ok_a = ga <= 0.3 or glm_a <= 0 or abs(ga - glm_a) < 0.08
            ok_h = gh_ <= 0.3 or glm_h <= 0 or abs(gh_ - glm_h) < 0.08
            if ok_a and ok_h:
                break
            if ga > 0.3 and glm_a > 0:
                fa = min(1.8, max(0.55, fa * (glm_a / ga) ** 0.5))
            if gh_ > 0.3 and glm_h > 0:
                fh = min(1.8, max(0.55, fh * (glm_h / gh_) ** 0.5))
    cums_a = [_cum(_scale_offense(p, fa)) for p in probs_a]
    cums_h = [_cum(_scale_offense(p, fh)) for p in probs_h]
    # Per-game rate uncertainty: precompute each batter's rate variants once,
    # then pick one per batter per sim (see RATE_SIGMA). Mean multiplier is 1,
    # so the anchored team means are unchanged — only the SPREAD widens.
    _mults = _rate_multipliers(rate_sigma)
    _nv = len(_mults)
    var_a = [[_cum(_scale_offense(p, fa * m)) for m in _mults] for p in probs_a]
    var_h = [[_cum(_scale_offense(p, fh * m)) for m in _mults] for p in probs_h]
    # Bullpen-scaled twins of the same lineups (see _bp_a/_bp_h above).
    varbp_a = [[_cum(_scale_offense(p, fa * m * _bp_a)) for m in _mults]
               for p in probs_a]
    varbp_h = [[_cum(_scale_offense(p, fh * m * _bp_h)) for m in _mults]
               for p in probs_h]

    # Full anchored run
    agg_a = [_BoxLine() for _ in range(9)]
    agg_h = [_BoxLine() for _ in range(9)]
    # accumulate sums and "got a hit / hr" counts
    sum_a = [dict(h=0, hr=0, tb=0, rbi=0, r=0, k=0, bb=0, pa=0, ph=0, phr=0) for _ in range(9)]
    sum_h = [dict(h=0, hr=0, tb=0, rbi=0, r=0, k=0, bb=0, pa=0, ph=0, phr=0) for _ in range(9)]
    psum_a = dict(outs=0, k=0, bb=0, h=0, hr=0, er=0)
    psum_h = dict(outs=0, k=0, bb=0, h=0, hr=0, er=0)
    home_wins = 0
    total_sum = 0
    score_counter: dict = {}
    total_hist: dict = {}
    # Per-player value->count histograms for the leaderboard.
    # "hrr" = Hits+Runs+RBIs, a real book market. Accumulated per SIM (h+r+rbi
    # from the same simulated game), so the correlation between the three is
    # captured exactly — which independent marginals cannot do.
    _BSTATS = ("h", "hr", "tb", "rbi", "r", "k", "bb", "hrr")
    _PSTATS = ("k", "bb", "h", "hr", "er", "outs")
    bh_a = [{st: {} for st in _BSTATS} for _ in range(9)]
    bh_h = [{st: {} for st in _BSTATS} for _ in range(9)]
    ph_a = {st: {} for st in _PSTATS}
    ph_h = {st: {} for st in _PSTATS}
    margin_counts: dict = {}
    total_counts: dict = {}
    ta = th = 0
    for i in range(n):
        spa = _sp_target(gp.get("away_starter"), rng)
        sph = _sp_target(gp.get("home_starter"), rng)
        # Draw tonight's rate realisation per batter (over-dispersion).
        if _nv > 1:
            _ix = [rng.randrange(_nv) for _ in range(9)]
            _iy = [rng.randrange(_nv) for _ in range(9)]
            ca = [var_a[j][_ix[j]] for j in range(9)]
            ch = [var_h[j][_iy[j]] for j in range(9)]
            # same rate realisation, bullpen-scaled — a hitter's talent draw
            # must persist across the pitching change within one game.
            cba = [varbp_a[j][_ix[j]] for j in range(9)]
            cbh = [varbp_h[j][_iy[j]] for j in range(9)]
        else:
            ca, ch = cums_a, cums_h
            cba = [varbp_a[j][0] for j in range(9)]
            cbh = [varbp_h[j][0] for j in range(9)]
        ba, bh, pa, ph, ra, rh = _simulate_one(ca, ch, spa, sph, rng,
                                               bp_away=cba, bp_home=cbh,
                                               ctx=_ctx)
        ta += ra
        th += rh
        for side_box, side_sum, side_hist in ((ba, sum_a, bh_a), (bh, sum_h, bh_h)):
            for j in range(9):
                bl = side_box[j]
                s = side_sum[j]
                s["pa"] += bl.pa; s["h"] += bl.h; s["hr"] += bl.hr; s["tb"] += bl.tb
                s["rbi"] += bl.rbi; s["r"] += bl.r; s["k"] += bl.k; s["bb"] += bl.bb
                if bl.h >= 1:
                    s["ph"] += 1
                if bl.hr >= 1:
                    s["phr"] += 1
                hd = side_hist[j]
                for st, val in (("h", bl.h), ("hr", bl.hr), ("tb", bl.tb),
                                ("rbi", bl.rbi), ("r", bl.r), ("k", bl.k), ("bb", bl.bb),
                                ("hrr", bl.h + bl.r + bl.rbi)):
                    d = hd[st]; d[val] = d.get(val, 0) + 1
        for src, dst, dhist in ((pa["SP"], psum_a, ph_a), (ph["SP"], psum_h, ph_h)):
            dst["outs"] += src.outs; dst["k"] += src.k; dst["bb"] += src.bb
            dst["h"] += src.h; dst["hr"] += src.hr; dst["er"] += src.er
            for st, val in (("k", src.k), ("bb", src.bb), ("h", src.h),
                            ("hr", src.hr), ("er", src.er), ("outs", src.outs)):
                d = dhist[st]; d[val] = d.get(val, 0) + 1
        if rh > ra:
            home_wins += 1
        total_sum += ra + rh
        total_hist[ra + rh] = total_hist.get(ra + rh, 0) + 1
        total_counts[ra + rh] = total_counts.get(ra + rh, 0) + 1
        margin_counts[rh - ra] = margin_counts.get(rh - ra, 0) + 1
        key = f"{rh}-{ra}"
        score_counter[key] = score_counter.get(key, 0) + 1

    def _box(batters, sums):
        out = []
        for b, s in zip(batters, sums):
            out.append({
                "name": b.get("name", "?"),
                "order": int(b.get("bat_order") or 0),
                "pa": round(s["pa"] / n, 2),
                "h": round(s["h"] / n, 2), "hr": round(s["hr"] / n, 3),
                "tb": round(s["tb"] / n, 2), "rbi": round(s["rbi"] / n, 2),
                "r": round(s["r"] / n, 2), "k": round(s["k"] / n, 2),
                "bb": round(s["bb"] / n, 2),
                "p_hit": round(s["ph"] / n, 3), "p_hr": round(s["phr"] / n, 3),
            })
        return out

    def _pit(starter, psum):
        return {
            "name": (starter or {}).get("name", "?"),
            "ip": round(psum["outs"] / n / 3.0, 2),
            "k": round(psum["k"] / n, 2), "bb": round(psum["bb"] / n, 2),
            "h": round(psum["h"] / n, 2), "hr": round(psum["hr"] / n, 2),
            "er": round(psum["er"] / n, 2),
        }

    # Key per-player histograms by player_id for leaderboard matching.
    bat_hist: dict = {}
    for jb, b in enumerate(bat_a):
        pid = b.get("player_id")
        if pid is not None:
            bat_hist[int(pid)] = bh_a[jb]
    for jb, b in enumerate(bat_h):
        pid = b.get("player_id")
        if pid is not None:
            bat_hist[int(pid)] = bh_h[jb]
    pit_hist: dict = {}
    for sid_key, dhist in (("away_sp_id", ph_a), ("home_sp_id", ph_h)):
        sid = gp.get(sid_key)
        if sid is not None:
            try:
                pit_hist[int(sid)] = dhist
            except (TypeError, ValueError):
                pass

    return SimResult(
        n=n, away_team=gp.get("away_team", "AWAY"), home_team=gp.get("home_team", "HOME"),
        box_away=_box(bat_a, sum_a), box_home=_box(bat_h, sum_h),
        pit_away=_pit(gp.get("away_starter"), psum_a),
        pit_home=_pit(gp.get("home_starter"), psum_h),
        glm_away=round(glm_a, 2), glm_home=round(glm_h, 2),
        free_away=round(free_a, 2), free_home=round(free_h, 2),
        anchored_away=round(ta / n, 2), anchored_home=round(th / n, 2),
        anchor_f_away=round(fa, 3), anchor_f_home=round(fh, 3),
        p_home_win=round(home_wins / n, 3),
        mean_total=round(total_sum / n, 2),
        score_dist=dict(sorted(score_counter.items(), key=lambda kv: -kv[1])[:6]),
        total_hist={k: round(v / n, 4) for k, v in sorted(total_hist.items())},
        bat_hist=bat_hist, pit_hist=pit_hist,
        margin_counts=margin_counts, total_counts=total_counts,
    )
