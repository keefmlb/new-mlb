"""Measure (and tune) the SIMULATOR's calibration against real boxscores.

Why: the sim used to treat each batter's projected rates as known, so it only
produced sampling variance and came out 10-20pp overconfident between 55% and
90% (measured on 3,101 graded sim picks). `game_sim.RATE_SIGMA` now injects
per-game rate uncertainty (Gamma-mixed rates = the NegBin construction the
pricing path already uses). This script answers: did that fix the calibration,
and what sigma is best?

Method — no odds, no waiting for live picks:
  1. Take historical FINAL games from data/games/games_2026.csv.
  2. Rebuild each game's lineups + projections exactly as predict_slate does,
     using the leak-free weekly snapshot for that date.
  3. Simulate at a given rate_sigma; for every batter/stat emit the sim's
     P(over) at each standard line.
  4. Grade against the actual boxscore (box_2026.csv).
  5. Report realized win rate per confidence bracket -> the calibration curve.

A perfectly calibrated sim has gap ~0 in every bracket.

Run:  python -m scripts.sim_calibration_backtest            # default sigmas
      python -m scripts.sim_calibration_backtest 0 0.35 0.5 # explicit sweep
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import game_sim  # noqa: E402

GAMES = ROOT / "data" / "games" / "games_2026.csv"
BOX = ROOT / "data" / "games" / "box_2026.csv"

# sim stat -> boxscore column(s) (summed for hrr)
_BAT_COLS = {
    "h": ["h"], "hr": ["hr"], "tb": ["tb"], "rbi": ["rbi"], "r": ["runs_b"],
    "k": ["k_b"], "bb": ["bb_b"], "hrr": ["h", "runs_b", "rbi"],
}
_PIT_COLS = {
    "k": ["k_p"], "bb": ["bb_p"], "h": ["h_p"], "hr": ["hr_p"],
    "er": ["er"], "outs": ["outs"],
}
_BANDS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75),
          (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.001)]


def _hist_over(hist: dict, line: float) -> int:
    return sum(c for v, c in hist.items() if float(v) > line)


def _build_games(dates: list[str], max_games: int) -> list[dict]:
    """Rebuild historical game dicts ONCE (expensive); reused for every sigma."""
    from src import predict_core
    out: list[dict] = []
    for d in dates:
        if len(out) >= max_games:
            break
        try:
            slate = predict_core.predict_slate(d, fetch_odds=False)
        except Exception as exc:
            print(f"  ({d}: slate failed — {exc})")
            continue
        for g in slate.games:
            if len(out) >= max_games:
                break
            if len(g.away_batters or []) < 9 or len(g.home_batters or []) < 9:
                continue
            out.append({
                "game_pk": g.game_pk, "home_team": g.home_team,
                "away_team": g.away_team,
                "pred_away_runs": g.pred_away_runs,
                "pred_home_runs": g.pred_home_runs,
                "away_batters": g.away_batters, "home_batters": g.home_batters,
                "away_starter": g.away_starter, "home_starter": g.home_starter,
                "away_sp_id": g.away_sp_id, "home_sp_id": g.home_sp_id,
            })
        print(f"  built {len(out)} games (through {d})")
    return out


def _collect(gps: list[dict], sigma: float, nsim: int) -> list[tuple[float, int]]:
    """Return [(sim_prob, won)] for every predicted OVER across the games."""
    box = pd.read_csv(BOX)
    box = box.drop_duplicates(["game_pk", "player_id"]).set_index(
        ["game_pk", "player_id"])
    out: list[tuple[float, int]] = []
    for gp in gps:
        gpk = gp["game_pk"]
        try:
            res = game_sim.simulate_game(gp, n=nsim, seed=gpk,
                                         rate_sigma=sigma)
        except Exception:
            continue
        if True:
            for store, colmap in ((res.bat_hist, _BAT_COLS),
                                  (res.pit_hist, _PIT_COLS)):
                for pid, hists in store.items():
                    try:
                        brow = box.loc[(int(gpk), int(pid))]
                    except (KeyError, TypeError, ValueError):
                        continue
                    if isinstance(brow, pd.DataFrame):
                        brow = brow.iloc[0]
                    for stat, cols in colmap.items():
                        h = hists.get(stat)
                        if not h or any(c not in brow or pd.isna(brow[c])
                                        for c in cols):
                            continue
                        actual = float(sum(float(brow[c]) for c in cols))
                        mean = (sum(float(v) * c for v, c in h.items())
                                / max(1, sum(h.values())))
                        line = 0.5
                        while line <= mean + 1.0:
                            p = _hist_over(h, line) / res.n
                            if 0.0 < p < 1.0 and actual != line:
                                out.append((p, 1 if actual > line else 0))
                            line += 1.0
    return out


def _report(label: str, rows: list[tuple[float, int]]) -> float:
    if not rows:
        print(f"  {label}: no rows"); return float("nan")
    print(f"\n  === rate_sigma = {label}  (n={len(rows)}) ===")
    print(f"  {'band':>10} {'bets':>6} {'predicted':>10} {'actual':>7} {'gap':>7}")
    tot_w = 0.0
    tot_n = 0
    for lo, hi in _BANDS:
        sub = [(p, w) for p, w in rows if lo <= p < hi]
        if not sub:
            continue
        pred = sum(p for p, _ in sub) / len(sub)
        act = sum(w for _, w in sub) / len(sub)
        gap = (act - pred) * 100
        tot_w += abs(gap) * len(sub)
        tot_n += len(sub)
        lbl = f"{lo*100:.0f}-{min(hi,1.0)*100:.0f}%"
        print(f"  {lbl:>10} {len(sub):6d} {pred:9.0%} {act:6.0%} {gap:+6.1f}pp")
    mae = tot_w / tot_n if tot_n else float("nan")
    print(f"  weighted mean |gap| = {mae:.2f}pp   <-- lower is better")
    return mae


def main():
    sigmas = [float(a) for a in sys.argv[1:]] or [0.0, 0.25, 0.35, 0.50]
    nsim = 2000
    max_games = 40
    gdf = pd.read_csv(GAMES)
    gdf = gdf[gdf.get("is_final") == True]  # noqa: E712
    dates = sorted(gdf["date"].astype(str).unique())[-12:]
    print(f"Calibrating on up to {max_games} games from {len(dates)} recent "
          f"dates ({dates[0]}..{dates[-1]}), {nsim} sims each.")
    print("sigma=0 reproduces the OLD sim (rates treated as known).")
    gps = _build_games(dates, max_games)
    if not gps:
        print("No historical games could be rebuilt."); return
    print(f"Rebuilt {len(gps)} games; sweeping sigmas {sigmas}")
    best = None
    for s in sigmas:
        rows = _collect(gps, s, nsim)
        mae = _report(f"{s:g}", rows)
        if rows and (best is None or mae < best[1]):
            best = (s, mae)
    if best:
        print(f"\nBEST sigma = {best[0]:g}  (weighted mean |gap| {best[1]:.2f}pp)")
        print("Set game_sim.RATE_SIGMA to this value if it beats the current one.")


if __name__ == "__main__":
    main()
