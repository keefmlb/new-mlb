"""Re-grade every K x TB parlay combination using the CURRENT simulator.

The combo grid in scripts/combo_grid.py reads sim_picks.json, which is a log of
what the simulator believed AT THE TIME. After the Aug 10-11 rework (HBP events,
mean-preserving bullpen split, hit-mix decomposition, substitution model) that
log describes a simulator that no longer exists.

This rebuilds the boards from scratch: real archived prices out of
closing_props.csv, probabilities from today's simulator, graded against final
boxscores. Same question, current model, and a longer window — the archive
carries ~46 dates against the ~26 tickets the live log had.

Usage: python -m scripts.replay_combo_grid [max_per_market] [n_sims]
"""
from __future__ import annotations
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict

import pandas as pd

from src import predict_core, game_sim

K, TB = "pitcher_k", "tb"
BOOST = 0.20
BOOT = 10000
# archived market -> (internal sim market, boxscore column)
MKT = {K: ("prop_pitcher_k", "k_p"), TB: ("prop_tb", "tb")}


def _dec(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o)) if o else None


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


def build_picks(n_sims: int = 1200) -> list[dict]:
    """Replay archived offers through the current simulator AND the real board.

    Ranking raw offers by simulated probability is NOT what the app shows. Doing
    that put pitcher-K legs at a 97-100% hit rate and a five-leg parlay at +9,
    because the highest-probability offer is always the lowest alt line
    ("over 0.5 K" at -769). Routing through build_sim_boards applies the board
    construction the picks actually pass through — the -400 price floor, line
    diversification, TB/hits dedupe and the pitcher-K adverse-selection cap —
    so the replay grades the board rather than the offer sheet.
    """
    props = pd.read_csv("data/odds/closing_props.csv")
    props = props[props["market"].isin(MKT)]
    box = pd.read_csv("data/games/box_2026.csv")
    actual: dict = {}
    for r in box.itertuples():
        try:
            actual[(int(r.game_pk), int(r.player_id))] = r
        except (TypeError, ValueError):
            continue

    picks: list[dict] = []
    for day in sorted(props["date"].dropna().unique()):
        day = str(day)[:10]
        sub = props[props["date"].astype(str).str[:10] == day]
        try:
            slate = predict_core.predict_slate(day, fetch_odds=False)
        except Exception:                                   # noqa: BLE001
            continue
        games = [g for g in slate.games
                 if len(g.away_batters or []) >= 9 and len(g.home_batters or []) >= 9]
        if not games:
            continue
        cache: dict = {}

        def sim_for(gp_obj, _cache=cache, _n=n_sims):
            if gp_obj is None:
                return None
            k = int(gp_obj.game_pk)
            if k not in _cache:
                try:
                    _cache[k] = game_sim.simulate_game(_gp(gp_obj), n=_n,
                                                       seed=gp_obj.game_pk)
                except Exception:                           # noqa: BLE001
                    _cache[k] = None
            return _cache[k]

        # Archived offers -> the offered_bets shape build_sim_boards expects.
        # The description must carry " OVER " because _side_of parses it.
        offers: list[dict] = []
        for r in sub.itertuples():
            try:
                gpk, pid, line = int(r.game_pk), int(r.player_id), float(r.line)
                odds = float(r.over)
            except (TypeError, ValueError):
                continue
            if not odds:
                continue
            mkt = MKT[r.market][0]
            offers.append({
                "game_pk": gpk, "player_id": pid, "market": mkt, "line": line,
                "odds": odds, "player": getattr(r, "player", ""),
                "description": f"{getattr(r, 'player', '?')} OVER {line} "
                               f"{'Pitcher K' if mkt.endswith('_k') else 'TB'}",
            })
        if not offers:
            continue
        try:
            boards = game_sim.build_sim_boards(games, sim_for, offers, top=12)
        except Exception as e:                              # noqa: BLE001
            print(f"  {day}: board build failed ({e})", flush=True)
            continue

        n_day = 0
        for mkt in ("prop_pitcher_k", "prop_tb"):
            col = "k_p" if mkt.endswith("_k") else "tb"
            # enumerate() IS the board position — build_sim_boards returns rows
            # already diversified and ordered. Re-sorting these on sim_hit later
            # rebuilds a board the app never showed; on the live log that error
            # inverted the entire K-vs-TB conclusion.
            for _rank, row in enumerate(
                    boards.get("by_market", {}).get(mkt) or [], start=1):
                pid = row.get("player_id")
                gpk = row.get("game_pk")
                if pid is None or gpk is None:
                    continue
                a = actual.get((int(gpk), int(pid)))
                if a is None:
                    continue
                av = getattr(a, col, None)
                if av is None or pd.isna(av):
                    continue
                try:
                    line = float(row.get("line"))
                except (TypeError, ValueError):
                    continue
                picks.append({
                    "date": day, "market": mkt, "line": line, "rank": _rank,
                    "sim_hit": float(row.get("sim_hit_raw")
                                     or row.get("sim_hit") or 0.0),
                    "odds": row.get("odds"),
                    "outcome": "W" if float(av) > line else "L",
                })
                n_day += 1
        print(f"  {day}: {len(games)} games, {len(offers)} offers -> "
              f"{n_day} board picks", flush=True)
    return picks


def _wilson(w, n):
    if not n:
        return (0.0, 0.0)
    p, z = w / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def grid(picks: list[dict], mx: int) -> None:
    by: dict = defaultdict(lambda: defaultdict(list))
    for e in picks:
        by[e["date"]][e["market"]].append(e)
    kk, tt = "prop_pitcher_k", "prop_tb"

    print(f"\n{'=' * 88}\nREPLAY GRID — current simulator, {len(by)} dates, "
          f"{len(picks)} graded offers\n{'=' * 88}\n")
    # leg rates first: this is the number that is actually measurable
    for m, lab in ((kk, "pitcher K"), (tt, "TB")):
        for rk in range(1, mx + 1):
            sel = []
            for d in by:
                s = sorted(by[d].get(m, []), key=lambda e: e.get("rank") or 999)
                if len(s) >= rk:
                    sel.append(s[rk - 1])
            if not sel:
                continue
            w = sum(1 for e in sel if e["outcome"] == "W")
            lo, hi = _wilson(w, len(sel))
            print(f"  {lab:10s} rank {rk}  n={len(sel):3d}  hit "
                  f"{w/len(sel)*100:5.1f}%  [{lo*100:4.1f}%, {hi*100:5.1f}%]")

    print(f"\n{'build':>9s} {'tix':>4s} {'W':>3s} {'hit%':>6s} {'median':>8s} "
          f"{'ROI':>9s} {'95% CI on ROI':>22s} {'P(>0)':>6s}")
    out = []
    for n_k in range(0, mx + 1):
        for n_tb in range(0, mx + 1):
            if n_k + n_tb == 0:
                continue
            rets, odds = [], []
            for d in sorted(by):
                day = (sorted(by[d].get(kk, []), key=lambda e: e.get("rank") or 999)[:n_k]
                       + sorted(by[d].get(tt, []), key=lambda e: e.get("rank") or 999)[:n_tb])
                if len(day) < n_k + n_tb:
                    continue
                dec, ok = 1.0, True
                for e in day:
                    dv = _dec(e["odds"])
                    if dv is None:
                        ok = False
                        break
                    dec *= dv
                if not ok:
                    continue
                odds.append(dec)
                rets.append(1.0 + (dec - 1.0) * (1.0 + BOOST)
                            if all(e["outcome"] == "W" for e in day) else 0.0)
            if not rets:
                continue
            n = len(rets)
            won = sum(1 for r in rets if r > 0)
            random.seed(7)
            b = sorted(sum(random.choice(rets) for _ in range(n)) / n - 1.0
                       for _ in range(BOOT))
            out.append((n_k, n_tb, n, won, sum(rets) / n - 1.0,
                        st.median(odds), b[int(.025*BOOT)], b[int(.975*BOOT)],
                        sum(1 for x in b if x > 0) / BOOT))
    for n_k, n_tb, n, won, roi, med, lo, hi, pp in out:
        print(f"{f'{n_k}K+{n_tb}TB':>9s} {n:4d} {won:3d} {won/n*100:5.1f}% "
              f"{'+' + str(int((med-1)*100)):>8s} {roi:+8.1%} "
              f"[{lo:+7.1%},{hi:+8.1%}] {pp:6.2f}")


def main(mx: int = 4, n_sims: int = 1200) -> None:
    picks = build_picks(n_sims)
    json.dump(picks, open("data/bets/replay_picks.json", "w"))
    grid(picks, mx)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 4, int(a[1]) if len(a) > 1 else 1200)
