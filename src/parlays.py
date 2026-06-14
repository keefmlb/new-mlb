"""Model-confidence leaderboard + skill-backed parlay construction.

Two product surfaces built on the alt-line ladder (now that every offered
line is captured):

1. `model_confident_bets` — bets where the line sits at/beyond the edge of
   the MODEL's own predictive distribution, i.e. the model's raw (pre-blend)
   side probability clears a threshold. "Inside the model's CI" = the model
   is confident in a direction, independent of the market blend.

2. `build_parlays` — daily 3- and 4-leg parlays whose legs come ONLY from
   markets the forecasting scorecard shows real skill in (runs / rbi). One
   leg per game (independence), high model conviction.

Honesty note baked into the design: parlays multiply the book's hold, and
the model's measured edge on these markets is at best weak/unproven, so the
parlay payouts are presented with their EXPECTED value (model_prob ×
decimal − 1), which will usually be negative. The tab exists to TRACK
whether high-conviction skill-backed parlays beat their EV, not to assert
they're winners.
"""
from __future__ import annotations
from itertools import combinations

# Markets with DEMONSTRATED forecasting skill in the scorecard (currently
# runs/rbi, CI excludes 0 on the encompassing test). Kept as a named subset
# the app can offer as a "skill-backed only" filter — but parlay eligibility
# defaults to ALL_PROP_MARKETS now, so legs can come from any prop market
# (TB, HR, hits, K, etc.). Update as forecast_score.py promotes/demotes.
SKILL_MARKETS = ("prop_runs", "prop_rbi")

# Every main prop market we model. Parlay legs and the model-CI leaderboard
# both surface bets from any of these by default. The Model CI threshold and
# the per-parlay min_conf still gate which lines actually qualify, so this is
# eligibility — not an endorsement of every market.
ALL_PROP_MARKETS = (
    "prop_hits", "prop_hr", "prop_tb", "prop_rbi", "prop_runs",
    "prop_k", "prop_bb",
    "prop_pitcher_k", "prop_pitcher_bb", "prop_pitcher_h",
    "prop_pitcher_er", "prop_pitcher_hr", "prop_pitcher_outs",
)


def _side(desc: str) -> str:
    if " OVER " in desc:
        return "OVER"
    if " UNDER " in desc:
        return "UNDER"
    return ""


def _raw(b: dict) -> float:
    return float(b.get("model_prob_raw") or b.get("model_prob") or 0.0)


def model_confident_bets(bets: list[dict], threshold: float = 0.60,
                         markets: tuple | None = None) -> list[dict]:
    """Bets whose RAW model side-probability ≥ threshold (inside the model's
    CI). Deduped to one line per (game, player, market, side): the qualifying
    line with the best EV (most valuable confident bet). Sorted by raw model
    probability descending. `markets` optionally restricts to a market set."""
    best: dict[tuple, dict] = {}
    for b in bets:
        mkt = b.get("market", "")
        if not mkt.startswith("prop_"):
            continue
        if markets is not None and mkt not in markets:
            continue
        if _raw(b) < threshold:
            continue
        # Unconfirmed-starter bets are kept but flagged (the app prefixes ⚠️),
        # matching the main leaderboard — otherwise the tab is empty every
        # morning until lineups post, which is when the user actually looks.
        key = (b.get("game_pk"), b.get("player_id"), mkt, _side(b.get("description", "")))
        cur = best.get(key)
        if cur is None or b.get("ev_per_dollar", -9) > cur.get("ev_per_dollar", -9):
            best[key] = b
    return sorted(best.values(), key=lambda x: -_raw(x))


def _decimal(b: dict) -> float:
    d = b.get("decimal_odds")
    if d:
        return float(d)
    o = int(b.get("odds") or -110)
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / -o)


def _american(dec: float) -> int:
    b = dec - 1.0
    return int(round(b * 100)) if b >= 1.0 else int(round(-100.0 / b))


def _parlay(legs: list[dict], label: str) -> dict:
    dec = 1.0
    prob_model = 1.0   # blended/calibrated — our honest hit estimate
    prob_market = 1.0  # market no-vig implied
    prob_raw = 1.0     # raw model conviction (selection metric only)
    for l in legs:
        dec *= _decimal(l)
        prob_model *= float(l.get("model_prob") or _raw(l))
        prob_market *= float(l.get("novig_prob") or l.get("model_prob") or _raw(l))
        prob_raw *= _raw(l)
    return {
        "label": label,
        "n_legs": len(legs),
        "legs": [{"description": l.get("description", ""),
                  "market": l.get("market", ""),
                  "odds": int(l.get("odds") or -110),
                  "model_prob": round(float(l.get("model_prob") or 0), 4),
                  "model_prob_raw": round(_raw(l), 4),
                  "starters_confirmed": bool(l.get("starters_confirmed", True)),
                  "game_pk": l.get("game_pk")} for l in legs],
        "all_confirmed": all(l.get("starters_confirmed", True) for l in legs),
        "decimal_odds": round(dec, 3),
        "american_odds": _american(dec),
        # Headline hit probability uses the CALIBRATED (blended) leg probs —
        # raw conviction is overconfident and is only used to PICK legs.
        "model_prob": round(prob_model, 4),
        "market_prob": round(prob_market, 4),
        # EV under our calibrated estimate vs what the market implies. The gap
        # (ev_model − ev_market) is the claimed edge; ev_market ≈ −hold and is
        # what you should expect if the model has no real edge on these legs.
        "ev_per_dollar": round(prob_model * dec - 1.0, 4),
        "ev_market": round(prob_market * dec - 1.0, 4),
        "payout_per_dollar": round(dec - 1.0, 2),
    }


def build_parlays(bets: list[dict], sizes: tuple = (3, 4),
                  skill_markets: tuple | None = None,
                  min_conf: float = 0.55, pool: int = 6) -> list[dict]:
    """Build high-conviction parlays.

    `skill_markets`: which prop markets legs may come from. Pass None (default)
    for ALL_PROP_MARKETS — TB, HR, hits, K, runs, rbi, walks, plus all
    pitcher props. Pass `parlays.SKILL_MARKETS` to restrict to demonstrably
    skill-backed markets only (runs/rbi).

    Legs: raw model prob ≥ min_conf, one (highest-conf) leg per game so legs
    are from independent games. For each size we surface two parlays from the
    top `pool` legs: the highest combined (calibrated) probability ("Safest")
    and the highest combined EV ("Best value"). Combined hit prob and EV are
    shown alongside the market-implied EV so the claimed edge is visible.

    Note on min_conf=0.55: many one-sided overs (runs/rbi most of all) are
    inherently low-probability events, so 0.55 keeps the eligible pool
    populated; the parlay payout/odds reflect the conviction honestly.
    """
    allow = skill_markets if skill_markets is not None else ALL_PROP_MARKETS
    legs = [b for b in bets
            if b.get("market") in allow
            and _raw(b) >= min_conf
            and _side(b.get("description", "")) in ("OVER", "UNDER")]
    # one best leg per game (highest raw conviction)
    by_game: dict = {}
    for b in sorted(legs, key=lambda x: -_raw(x)):
        g = b.get("game_pk")
        if g not in by_game:
            by_game[g] = b
    cand = sorted(by_game.values(), key=lambda x: -_raw(x))[:pool]

    out: list[dict] = []
    seen: set = set()
    for size in sizes:
        if len(cand) < size:
            continue
        combos = list(combinations(cand, size))
        # Selection uses CALIBRATED leg probs (honest), even though eligibility
        # and the pool ranking use raw conviction (the "inside model CI" gate).
        def _mp(l):
            return float(l.get("model_prob") or _raw(l))
        variants = [
            ("Safest", lambda c: _prod(_mp(l) for l in c)),
            ("Best value", lambda c: _prod(_mp(l) for l in c) * _prod(_decimal(l) for l in c)),
        ]
        for vlabel, keyfn in variants:
            best = max(combos, key=keyfn)
            sig = frozenset(l.get("description", "") for l in best)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(_parlay(list(best), f"{size}-leg {vlabel}"))
    return out


def _prod(it) -> float:
    p = 1.0
    for x in it:
        p *= x
    return p
