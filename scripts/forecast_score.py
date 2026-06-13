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


def _collect() -> pd.DataFrame:
    snaps = _select_snapshots()
    games_by_date, by_game = _build_indexes()
    recs = []
    for day in sorted(snaps):
        # Per (game, player, market): keep the line closest to our projection.
        best: dict[tuple, tuple] = {}
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
            try:
                line = float(pp.get("line") or 0)
            except (TypeError, ValueError):
                continue
            key = (int(g.game_pk), resolved, market)
            dist = abs(float(proj) - line)
            if key not in best or dist < best[key][0]:
                best[key] = (dist, pp, float(proj), float(act), line,
                             int(g.game_pk), market)
        for (dist, pp, proj, act, line, gpk, market) in best.values():
            if act == line:
                continue  # push
            over = 1.0 if act > line else 0.0
            disp = value.get_dispersion(market, proj)
            p_model = value.calibrate_prop_prob(
                value.prob_over_count(proj, line, disp), market)
            o_odds, u_odds = pp.get("over"), pp.get("under")
            if o_odds is not None and u_odds is not None:
                nv_o, _ = value.devig_two_way(
                    value.american_to_prob(int(o_odds)),
                    value.american_to_prob(int(u_odds)))
                two_sided = True
            elif o_odds is not None:
                if int(o_odds) > _ONE_SIDED_MAX_ODDS:
                    continue
                nv_o = max(0.01, min(0.99,
                           value.american_to_prob(int(o_odds)) - _ONE_SIDED_JUICE / 2.0))
                two_sided = False
            else:
                continue
            bm = (p_model - over) ** 2
            bk = (nv_o - over) ** 2
            recs.append({"market": market, "two_sided": two_sided, "game_pk": gpk,
                         "p_model": p_model, "p_market": nv_o, "over": over,
                         "brier_model": bm, "brier_market": bk, "skill": bk - bm})
    return pd.DataFrame(recs)


def _report(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)
    print(f"  {'market':14s} {'n':>5s} {'games':>5s} {'Brier mdl':>9s} "
          f"{'Brier mkt':>9s} {'skill':>8s} {'skill 95% CI':>20s}  {'mdl better':>10s}")
    rows = []
    for mkt, sub in df.groupby("market"):
        rows.append((mkt, sub))
    rows.append(("ALL", df))
    for mkt, sub in sorted(rows, key=lambda x: -len(x[1])):
        n = len(sub)
        if n < 10:
            continue
        bm = sub["brier_model"].mean()
        bk = sub["brier_market"].mean()
        sk = sub["skill"].mean()
        ci = _cluster_boot_ci(sub["skill"].tolist(), sub["game_pk"].tolist())
        ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "—"
        better = (sub["skill"] > 0).mean()
        ng = sub["game_pk"].nunique()
        flag = " *" if (ci and ci[0] > 0) else (" x" if (ci and ci[1] < 0) else "")
        print(f"  {mkt:14s} {n:5d} {ng:5d} {bm:9.4f} {bk:9.4f} {sk:+8.4f} "
              f"{ci_s:>20s}  {better:9.0%}{flag}")
    print("  * = CI excludes 0, model beats market   x = market beats model")


def main():
    df = _collect()
    if df.empty:
        print("No priceable offers collected.")
        return
    print(f"Priceable forecasts: {len(df)}  "
          f"(two-sided: {int(df['two_sided'].sum())}, "
          f"one-sided: {int((~df['two_sided']).sum())})  "
          f"games: {df['game_pk'].nunique()}")
    ts = df[df["two_sided"]]
    if len(ts) >= 10:
        _report(ts, "TWO-SIDED OFFERS (honest de-vig) — model vs market forecast skill")
    os_ = df[~df["two_sided"]]
    if len(os_) >= 10:
        _report(os_, "ONE-SIDED OFFERS (8% juice-strip estimate) — model vs market skill")
    print("\n  skill = Brier_market − Brier_model per offer (>0 ⇒ model better).")
    print("  Paired on the same outcome, so far lower variance than ROI; runs on")
    print("  every offer, not just floor-clearers. Caveat: prop calibration was")
    print("  fitted on a pool overlapping these weeks (mild in-sample model edge).")


if __name__ == "__main__":
    main()
