"""Forecasting-skill scorecard: is our probability better than the market's?

ROI and W/L are high-variance estimators — they're dominated by payout luck,
so it takes hundreds of settled bets to separate skill from noise, and we
only have ~280. This script attacks the edge question with a far tighter
instrument: a PAIRED proper-scoring-rule comparison on EVERY priceable offer
in the odds history (not just the ~90/day that clear the betting floor).

For each offer we compute, against the SAME realized over/under outcome:
    Brier_model  = (P_model(over)  − outcome)²
    Brier_market = (P_market(over) − outcome)²
    skill        = Brier_market − Brier_model      (>0 ⇒ model beat market)

Because both Briers are graded on the same outcome, the outcome variance
cancels in the difference — `skill` has far lower variance than ROI, so its
mean converges with far fewer samples. mean(skill) > 0 with a CI excluding 0
is direct, low-noise evidence that our probability is a better forecast than
the market's no-vig probability — which is exactly the edge that has to exist
upstream of any positive CLV or ROI.

P_model is the RAW calibrated model forecast (calibrate_prop_prob ∘
prob_over_count) — NOT the market-blended price, so the comparison isn't
contaminated by the market it's being compared against.

Sample construction:
  - One pick per (game, player, market): the offered line CLOSEST to our
    projection (the most informative, near-50/50 forecast). Avoids the
    alt-line correlation that would fake-inflate n.
  - Two-sided offers de-vig honestly; one-sided "Yes" offers strip the same
    8% juice estimate the live pricer uses (reported separately — that novig
    is an estimate, not a measured median).
  - CIs are GAME-CLUSTERED bootstrap: resamples whole games, so within-game
    correlation (a high-scoring night lifts every batter) doesn't fake-narrow
    the interval.

Caveat: the prop calibration was fitted on a 2026 pool overlapping these
weeks, so P_model carries a mild in-sample edge. Read a model WIN with that
discount; a model LOSS despite the home-field advantage is strong.

Run:  python -m scripts.forecast_score
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import name_match, value
from scripts.replay_board import _select_snapshots, _build_indexes, _find_game, MKT

_ONE_SIDED_JUICE = 0.08
_ONE_SIDED_MAX_ODDS = 400


def _cluster_boot_ci(skills: list[float], games: list[int],
                     n_boot: int = 4000, level: float = 0.95,
                     seed: int = 0) -> tuple[float, float] | None:
    """Game-clustered bootstrap CI for mean(skill). Resamples whole games
    with replacement (size-weighted) so within-game correlation is respected."""
    if len(skills) < 10:
        return None
    groups: dict[int, list[float]] = defaultdict(list)
    for s, g in zip(skills, games):
        groups[g].append(s)
    keys = list(groups.keys())
    sums = np.array([sum(groups[k]) for k in keys], dtype=float)
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    G = len(keys)
    for i in range(n_boot):
        idx = rng.integers(0, G, size=G)
        means[i] = sums[idx].sum() / sizes[idx].sum()
    return (float(np.quantile(means, (1 - level) / 2)),
            float(np.quantile(means, (1 + level) / 2)))


def _score_offer(pp, proj, act, market, gpk, cal_fn=None) -> dict | None:
    """One Brier record for a single (player, market, LINE) offer, or None.

    cal_fn(raw_tail_prob, market) -> calibrated P(over). Defaults to the
    production calibrate_prop_prob; an out-of-sample variant can be injected
    to remove the in-sample calibration advantage."""
    try:
        line = float(pp.get("line") or 0)
    except (TypeError, ValueError):
        return None
    if act == line:
        return None  # push
    over = 1.0 if act > line else 0.0
    disp = value.get_dispersion(market, proj)
    _cal = cal_fn or value.calibrate_prop_prob
    p_model = _cal(value.prob_over_count(proj, line, disp), market)
    o_odds, u_odds = pp.get("over"), pp.get("under")
    if o_odds is not None and u_odds is not None:
        nv_o, _ = value.devig_two_way(value.american_to_prob(int(o_odds)),
                                      value.american_to_prob(int(u_odds)))
        two_sided = True
    elif o_odds is not None:
        if int(o_odds) > _ONE_SIDED_MAX_ODDS:
            return None
        nv_o = max(0.01, min(0.99,
                   value.american_to_prob(int(o_odds)) - _ONE_SIDED_JUICE / 2.0))
        two_sided = False
    else:
        return None
    return {"market": market, "two_sided": two_sided, "game_pk": gpk, "line": line,
            "dist": abs(proj - line), "p_model": p_model, "p_market": nv_o, "over": over,
            "brier_model": (p_model - over) ** 2, "brier_market": (nv_o - over) ** 2,
            "skill": (nv_o - over) ** 2 - (p_model - over) ** 2}


def _collect(nearest_only: bool = True,
             band: tuple = (0.15, 0.85), cal_fn=None) -> pd.DataFrame:
    """Build Brier records.

    nearest_only=True : one record per (game, player, market) — the line
        closest to our projection. The single-bit-per-game view.
    nearest_only=False: a record for EVERY offered line whose market-implied
        P(over) falls in `band` — the full implied-distribution view. Tail
        lines (everyone agrees ~0/1) carry no information and only add noise,
        so they're excluded. Game-clustered bootstrap keeps the CI honest:
        all lines of all players in a game resample together, so the extra
        within-game correlation does NOT fake-narrow the interval.
    cal_fn: optional (raw_tail_prob, market)->P(over) to override the
        production calibration (used for the out-of-sample variant).
    """
    snaps = _select_snapshots()
    games_by_date, by_game = _build_indexes()
    recs = []
    for day in sorted(snaps):
        bucket: dict[tuple, list] = defaultdict(list)
        for pp in snaps[day]["props"]:
            market = pp.get("market", "")
            if market not in MKT:
                continue
            g = _find_game(games_by_date, day, pp.get("game", ""))
            if g is None:
                continue
            kind, pcol, acol = MKT[market]
            rows = by_game[kind].get(int(g.game_pk), {})
            resolved = name_match.find_match(pp.get("player", ""), rows.keys())
            row = rows.get(resolved) if resolved else None
            if row is None:
                continue
            proj = getattr(row, pcol, None)
            act = getattr(row, acol, None)
            if proj is None or act is None or pd.isna(proj) or pd.isna(act) or float(proj) <= 0:
                continue
            rec = _score_offer(pp, float(proj), float(act), market, int(g.game_pk),
                               cal_fn=cal_fn)
            if rec is None:
                continue
            bucket[(int(g.game_pk), resolved, market)].append(rec)
        for items in bucket.values():
            if nearest_only:
                recs.append(min(items, key=lambda r: r["dist"]))
            else:
                recs.extend(r for r in items if band[0] <= r["p_market"] <= band[1])
    return pd.DataFrame(recs)


def _report(df: pd.DataFrame, title: str, focus: tuple = ()) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(f"  {'market':14s} {'n':>5s} {'games':>5s} {'Brier mdl':>9s} "
          f"{'Brier mkt':>9s} {'skill':>8s} {'CIhw':>7s} {'skill 95% CI':>20s}  {'mdl>mkt':>7s}")
    rows = [(mkt, sub) for mkt, sub in df.groupby("market")]
    rows.append(("ALL", df))
    # focus markets first (runs/rbi), then by size
    def _order(item):
        mkt = item[0]
        return (0 if mkt in focus else 1, -len(item[1]))
    for mkt, sub in sorted(rows, key=_order):
        n = len(sub)
        if n < 10:
            continue
        bm = sub["brier_model"].mean()
        bk = sub["brier_market"].mean()
        sk = sub["skill"].mean()
        ci = _cluster_boot_ci(sub["skill"].tolist(), sub["game_pk"].tolist())
        ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "—"
        hw = (ci[1] - ci[0]) / 2 if ci else float("nan")
        better = (sub["skill"] > 0).mean()
        ng = sub["game_pk"].nunique()
        flag = " *" if (ci and ci[0] > 0) else (" x" if (ci and ci[1] < 0) else "")
        mark = "»" if mkt in focus else " "
        print(f"{mark} {mkt:14s} {n:5d} {ng:5d} {bm:9.4f} {bk:9.4f} {sk:+8.4f} "
              f"{hw:7.4f} {ci_s:>20s}  {better:6.0%}{flag}")
    print("  * = CI excludes 0, model beats market   x = market beats market   "
          "CIhw = CI half-width (smaller = tighter)")


def main():
    # View 1: single nearest-line (one bit per game) — the original scorecard.
    single = _collect(nearest_only=True)
    # View 2: full implied-distribution (every informative line) — the new lever.
    multi = _collect(nearest_only=False)
    if single.empty:
        print("No priceable offers collected.")
        return

    FOCUS = ("runs", "rbi")
    for df, label in [(single, "SINGLE nearest-line (1 forecast/game)"),
                      (multi, "MULTI-LINE full implied-distribution (every informative line)")]:
        print("\n" + "#" * 88)
        print(f"# {label}")
        print(f"# offers: {len(df)}  (two-sided {int(df['two_sided'].sum())}, "
              f"one-sided {int((~df['two_sided']).sum())})  games {df['game_pk'].nunique()}")
        print("#" * 88)
        ts = df[df["two_sided"]]
        if len(ts) >= 10:
            _report(ts, "TWO-SIDED (honest de-vig)", focus=FOCUS)
        os_ = df[~df["two_sided"]]
        if len(os_) >= 10:
            _report(os_, "ONE-SIDED (8% juice-strip estimate)", focus=FOCUS)

    # Demonstrate the CI tightening on the focus markets head-to-head.
    print("\n" + "=" * 88)
    print("CI TIGHTENING — focus markets (runs, rbi): single-line vs multi-line")
    print("=" * 88)
    print(f"  {'market':10s} {'view':>6s} {'n':>6s} {'skill':>9s} {'CI half-width':>14s}")
    for mkt in FOCUS:
        for df, tag in [(single, "single"), (multi, "multi")]:
            sub = df[(df["market"] == mkt) & (~df["two_sided"])]
            if len(sub) < 10:
                continue
            sk = sub["skill"].mean()
            ci = _cluster_boot_ci(sub["skill"].tolist(), sub["game_pk"].tolist())
            hw = (ci[1] - ci[0]) / 2 if ci else float("nan")
            print(f"  {mkt:10s} {tag:>6s} {len(sub):6d} {sk:+9.4f} {hw:14.4f}")

    # Anchor pointed at runs/rbi: the betting-relevant "how much to trust the
    # model vs the market" is the Brier-optimal blend weight w in
    #   p = w·p_model + (1−w)·p_market   (current live default: 0.5).
    # Fit on the earlier 70% of game-days, evaluate on the last 30%. This is
    # the probability-space anchor (the count-anchor needs a two-sided central
    # line runs/rbi don't have). Read-only: we do NOT wire a market's weight
    # until its skill CI clears 0 — fitting on a signal whose CI still spans
    # zero would just chase noise (the trap we've avoided all along).
    print("\n" + "=" * 88)
    print("ANCHOR @ runs/rbi — Brier-optimal model-vs-market blend weight (read-only)")
    print("=" * 88)
    print(f"  {'market':10s} {'n':>6s} {'w*':>5s} {'Brier@0.5':>10s} "
          f"{'Brier@w*':>9s} {'gain':>7s}  note")
    ws = np.linspace(0.0, 1.0, 21)
    for mkt in FOCUS:
        sub = multi[(multi["market"] == mkt) & (~multi["two_sided"])].copy()
        if len(sub) < 50:
            continue
        sub = sub.sort_values("game_pk")
        # reconstruct p_model from brier_model + over (p_model = over ± sqrt)
        # we stored p_market and over but not p_model; recompute from brier.
        # brier_model = (p_model-over)^2 -> p_model = over ± sqrt(bm); pick the
        # root on the correct side using skill sign is fragile, so store p_model.
        gkeys = sub["game_pk"].to_numpy()
        cut_game = np.quantile(np.unique(gkeys), 0.7)
        tr = sub[sub["game_pk"] <= cut_game]
        te = sub[sub["game_pk"] > cut_game]
        if len(tr) < 30 or len(te) < 15:
            continue
        def _brier_at(w, d):
            p = w * d["p_model"].to_numpy() + (1 - w) * d["p_market"].to_numpy()
            return float(((p - d["over"].to_numpy()) ** 2).mean())
        w_star = min(ws, key=lambda w: _brier_at(w, tr))
        b_half = _brier_at(0.5, te)
        b_star = _brier_at(w_star, te)
        gain = b_half - b_star
        skill_ci = _cluster_boot_ci(sub["skill"].tolist(), sub["game_pk"].tolist())
        actionable = skill_ci and skill_ci[0] > 0
        note = "ACTIONABLE (skill CI>0)" if actionable else "hold (skill CI spans 0)"
        print(f"  {mkt:10s} {len(sub):6d} {w_star:5.2f} {b_half:10.4f} "
              f"{b_star:9.4f} {gain:+7.4f}  {note}")
    print("  w*>0.5 ⇒ trust model more than the live 0.5 default; w*<0.5 ⇒ less.")
    print("  Held until skill CI clears 0 so we tune on signal, not noise.")

    print("\n  NEW LEVER: scoring every informative line (not just one) uses the")
    print("  market's full implied distribution as the benchmark and the model's")
    print("  full distribution as the forecast — more information per game, so the")
    print("  skill CI tightens. Game-clustered bootstrap keeps it honest (lines")
    print("  within a game resample together). skill = Brier_mkt − Brier_mdl (>0 ⇒")
    print("  model better). Caveat: prop calibration overlaps these weeks in-sample.")


if __name__ == "__main__":
    main()
