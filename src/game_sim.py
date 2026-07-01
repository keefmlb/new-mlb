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


def _advance(bases, bi, ev, outs, rng):
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
        if outs + 1 < 3:            # productive out only when it's not the 3rd
            if t is not None and rng.random() < 0.30:
                scored.append(t)
                rbi += 1
                t = None
            if s is not None and t is None and rng.random() < 0.20:
                t, s = s, None
            bases = [f, s, t]
    return scored, bases, rbi, oa


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


def _half_inning(cums, box, pit_box, start_idx, mound, rng):
    """Play one half-inning. Returns (runs, next_batter_index)."""
    outs = 0
    bases = [None, None, None]
    bi = start_idx
    runs = 0
    while outs < 3:
        slot = bi % 9
        ev = EVENTS[_draw(cums[slot], rng.random())]
        _record_offense(box, slot, ev)
        scored, bases, rbi, oa = _advance(bases, slot, ev, outs, rng)
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
        # hook: pull the starter once his out target is reached
        if mound[0] == "SP" and pl.outs >= mound[1]:
            mound[0] = "BP"
        bi += 1
    return runs, bi


def _simulate_one(cums_away, cums_home, sp_away_outs, sp_home_outs, rng):
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
    while True:
        inning += 1
        # top: away bats, faces home pitching
        r, idx_a = _half_inning(cums_away, box_a, pit_h, idx_a, mound_h, rng)
        runs_a += r
        # walk-off skip: home leading after the top of the 9th+ doesn't bat
        if inning >= 9 and runs_h > runs_a:
            break
        # bottom: home bats, faces away pitching
        r, idx_h = _half_inning(cums_home, box_h, pit_a, idx_h, mound_a, rng)
        runs_h += r
        if inning >= 9 and runs_h != runs_a:
            break
        if inning >= 15:             # safety cap on extras
            break
    return box_a, box_h, pit_a, pit_h, runs_a, runs_h


def _mean_runs(cums_a, cums_h, spa, sph, n, rng):
    ta = th = 0
    for _ in range(n):
        _, _, _, _, ra, rh = _simulate_one(cums_a, cums_h, spa, sph, rng)
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


def _sp_target(starter: dict, rng: random.Random) -> int:
    base = float((starter or {}).get("expected_outs") or 15.0)
    return max(6, min(24, int(round(rng.gauss(base, 3.0)))))


# ---------- Simulation leaderboard ----------
# market -> (kind, stat) where kind is "bat" or "pit".
_LB_PROP_MAP = {
    "prop_hits": ("bat", "h"), "prop_hr": ("bat", "hr"), "prop_tb": ("bat", "tb"),
    "prop_rbi": ("bat", "rbi"), "prop_runs": ("bat", "r"), "prop_k": ("bat", "k"),
    "prop_bb": ("bat", "bb"),
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


def _build_sim_rows(games: list, sim_for_game, offered_bets: list[dict]) -> list[dict]:
    """Score every offered prop / game line by its simulated hit rate, deduped
    to one row per (game, player/market/side, line). Returns ALL rows
    (unsorted, untruncated) — callers slice/group as needed."""
    sims: dict = {}
    rows: list[dict] = []
    seen: set = set()
    for b in offered_bets:
        gpk = b.get("game_pk")
        if gpk is None:
            continue
        if gpk not in sims:
            gp = next((g for g in games if g.game_pk == gpk), None)
            sims[gpk] = sim_for_game(gp) if gp is not None else None
        res = sims[gpk]
        if res is None or isinstance(res, str):
            continue
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
            "matchup": f"{res.away_team} @ {res.home_team}",
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
                          top: int = 20) -> list[dict]:
    """Rank every offered prop / game line across the slate by its simulated
    hit rate. `sim_for_game(gp)->SimResult` runs (or caches) the sim for a
    game; `offered_bets` is the slate-wide pool (each dict has market,
    description, line, odds, game_pk, player_id). Returns the top `top` by
    hit rate, deduped to one row per (game, player/market/side, line)."""
    rows = _build_sim_rows(games, sim_for_game, offered_bets)
    rows.sort(key=lambda r: -r["sim_hit"])
    return rows[:top]


def build_sim_leaderboard_by_market(games: list, sim_for_game,
                                    offered_bets: list[dict],
                                    top: int = 20) -> dict[str, list[dict]]:
    """Same scoring as `build_sim_leaderboard`, but grouped by market (stat).
    Returns {market: top-N rows by sim hit count}, one entry per market that
    has at least one mapped offer. Each market's rows are independently ranked
    by simulated hit frequency and capped at `top`."""
    rows = _build_sim_rows(games, sim_for_game, offered_bets)
    by_mkt: dict[str, list[dict]] = {}
    for r in rows:
        by_mkt.setdefault(r["market"], []).append(r)
    for mkt in by_mkt:
        by_mkt[mkt].sort(key=lambda r: -r["sim_hit"])
        by_mkt[mkt] = by_mkt[mkt][:top]
    return by_mkt


def simulate_game(gp: dict, n: int = 2000, seed: int = 0,
                  anchor: bool = True) -> SimResult:
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

    # Pilot (free rates) -> bottom-up mean for the "show both" comparison and
    # the anchor factors.
    pilot = max(150, n // 5)
    cums_a0 = [_cum(p) for p in probs_a]
    cums_h0 = [_cum(p) for p in probs_h]
    spa = _sp_target(gp.get("away_starter"), rng)
    sph = _sp_target(gp.get("home_starter"), rng)
    free_a, free_h = _mean_runs(cums_a0, cums_h0, spa, sph,
                                pilot, random.Random(seed + 1))

    fa = fh = 1.0
    if anchor and free_a > 0.3 and glm_a > 0:
        fa = min(1.8, max(0.55, glm_a / free_a))
    if anchor and free_h > 0.3 and glm_h > 0:
        fh = min(1.8, max(0.55, glm_h / free_h))
    cums_a = [_cum(_scale_offense(p, fa)) for p in probs_a]
    cums_h = [_cum(_scale_offense(p, fh)) for p in probs_h]

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
    _BSTATS = ("h", "hr", "tb", "rbi", "r", "k", "bb")
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
        ba, bh, pa, ph, ra, rh = _simulate_one(cums_a, cums_h, spa, sph, rng)
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
                                ("rbi", bl.rbi), ("r", bl.r), ("k", bl.k), ("bb", bl.bb)):
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
