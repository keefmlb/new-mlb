"""Ground-truth bias audit: sim batter means vs what actually happened.

The sim is a re-expression of the projections, so it is easy to check it
against itself and conclude everything is fine. This script instead grades the
sim's MEAN for each batter against the real boxscore line for that player in
that game, which is the only number that matters for a counting prop.

Reports, per stat:
  - sim mean vs actual mean (multiplicative bias)
  - the same for the raw projection, so we can see whether the sim ADDS bias
  - PA bias split out, since every counting stat scales with it

Usage: python -m scripts.sim_bias_audit [n_days]
"""
from __future__ import annotations
import sys
from collections import defaultdict

import pandas as pd

from src import predict_core, game_sim

STATS = [("pa", "expected_pa"), ("h", "proj_h"), ("tb", "proj_tb"),
         ("hr", "proj_hr"), ("doubles", "proj_2b"), ("triples", "proj_3b"),
         ("bb_b", "proj_bb"), ("k_b", "proj_k"),
         ("runs_b", "proj_runs"), ("rbi", "proj_rbi")]
# The sim tracks no 2B/3B split (it records tb), so those rows compare the
# PROJECTION against reality only; sim column is left at 0 and skipped.
SIMKEY = {"bb_b": "bb", "k_b": "k", "runs_b": "r"}
NO_SIM = {"doubles", "triples"}


def _gp(g) -> dict:
    return {
        "game_pk": g.game_pk, "home_team": g.home_team, "away_team": g.away_team,
        "pred_away_runs": g.pred_away_runs, "pred_home_runs": g.pred_home_runs,
        "away_batters": g.away_batters, "home_batters": g.home_batters,
        "away_starter": g.away_starter, "home_starter": g.home_starter,
        "away_sp_id": g.away_sp_id, "home_sp_id": g.home_sp_id,
        "home_sp_fip": g.home_sp_fip, "away_sp_fip": g.away_sp_fip,
        "home_bp_fip": g.home_bp_fip, "away_bp_fip": g.away_bp_fip,
        "home_sp_bb9": g.home_sp_bb9, "away_sp_bb9": g.away_sp_bb9,
        "home_bp_bb9": g.home_bp_bb9, "away_bp_bb9": g.away_bp_bb9,
    }


def main(n_days: int = 10, n_sims: int = 1500) -> None:
    box = pd.read_csv("data/games/box_2026.csv")
    box = box[box["pa"].fillna(0) > 0]
    dates = sorted(box["date"].dropna().unique())[-n_days:]
    # actual line per (game_pk, player_id)
    act = {(int(r.game_pk), int(r.player_id)): r for r in box.itertuples()}

    tot = defaultdict(lambda: [0.0, 0.0, 0.0, 0])   # stat -> [proj, sim, actual, n]
    for d in dates:
        try:
            slate = predict_core.predict_slate(str(d)[:10], fetch_odds=False)
        except Exception as e:                      # noqa: BLE001
            print(f"  {d}: slate failed ({e})")
            continue
        for g in slate.games:
            if len(g.away_batters or []) < 9 or len(g.home_batters or []) < 9:
                continue
            if (int(g.game_pk), None) is None:
                continue
            try:
                res = game_sim.simulate_game(_gp(g), n=n_sims, seed=g.game_pk)
            except Exception:                       # noqa: BLE001
                continue
            for proj9, simbox in ((g.away_batters[:9], res.box_away),
                                  (g.home_batters[:9], res.box_home)):
                for b, sb in zip(proj9, simbox):
                    pid = b.get("player_id")
                    if pid is None:
                        continue
                    a = act.get((int(g.game_pk), int(pid)))
                    if a is None:                   # scratched / didn't play
                        continue
                    for acol, pkey in STATS:
                        skey = SIMKEY.get(acol, acol)
                        av = getattr(a, acol, None)
                        if av is None or pd.isna(av):
                            continue
                        t = tot[acol]
                        t[0] += float(b.get(pkey) or 0.0)
                        t[1] += float(sb.get(skey) or 0.0)
                        t[2] += float(av)
                        t[3] += 1
        print(f"  {str(d)[:10]} done")

    n = tot["pa"][3]
    print(f"\nSIM BIAS AUDIT — {len(dates)} dates, {n} player-games graded\n")
    print(f"{'stat':6s} {'proj':>8s} {'sim':>8s} {'actual':>8s} "
          f"{'proj/act':>9s} {'sim/act':>8s}")
    for acol, _ in STATS:
        p, s, a, c = tot[acol]
        if not c or not a:
            continue
        simcol = "     n/a" if acol in NO_SIM else f"{s/a:8.3f}"
        print(f"{acol:7s} {p/c:8.3f} {s/c:8.3f} {a/c:8.3f} "
              f"{p/a:9.3f} {simcol}")
    ph, pt, pa_ = tot["h"][0], tot["tb"][0], tot["tb"][2]
    ah = tot["h"][2]
    if ph and ah:
        print(f"\nbases per hit:  projected {pt/ph:.3f}   actual {pa_/ah:.3f}   "
              f"({(pt/ph)/(pa_/ah)-1:+.1%})")
    print("\nsim/act > 1 means the simulation over-projects that stat for the "
          "NAMED batter — the number a prop actually settles on.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
