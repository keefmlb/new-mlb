"""Backtest correlated same-game 2-leg parlays on REAL offered legs + actuals.

The question this answers: are same-game prop legs positively correlated, and
does betting the correlated ones beat the book pricing them independently?

Data (no simulation, no model needed — pure realized outcomes):
  - data/odds/closing_props.csv : the real Fanatics legs offered near each
    first pitch (game_pk, player_id, market, line, over/under American odds).
  - data/games/box_2026.csv     : actual player boxscores → grade each leg W/L.
  - data/games/games_2026.csv    : final flag / dates.

Method:
  1. Grade every closing-props leg (OVER/UNDER of the actual stat vs the line;
     whole-number ties are pushes and dropped).
  2. Within each (date, game), enumerate 2-leg combos of graded legs.
  3. For each combo measure:
       - realized JOINT win rate  (both legs hit)
       - INDEPENDENT product      (p_leg1 * p_leg2, from that market/side's own
                                    base rate across the sample)
       - correlation lift         (joint / independent)
       - realized flat-stake ROI at the MULTIPLIED price (d1*d2-1 if both win
         else -1).
  4. Report overall, and broken down by whether the two legs are SAME-TEAM
     (expected positive correlation among a lineup) vs OPPOSING.

Caveat: multiplied odds overstate real same-game-parlay payouts (books apply a
correlation haircut). ROI here is therefore an OPTIMISTIC bound — the
correlation LIFT is the assumption-free result.

Run:  python -m scripts.parlay_backtest
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLOSING = ROOT / "data" / "odds" / "closing_props.csv"
BOX = ROOT / "data" / "games" / "box_2026.csv"
GAMES = ROOT / "data" / "games" / "games_2026.csv"

# market -> boxscore column (batter markets only; pitcher props rarely pair up
# within the same game for a hitter-correlation test, but we include K/outs etc.)
_STAT_COL = {
    "hits": "h", "hr": "hr", "tb": "tb", "rbi": "rbi", "runs": "runs_b",
    "k": "k_b", "bb": "bb_b",
    "pitcher_k": "k_p", "pitcher_h": "h_p", "pitcher_er": "er",
    "pitcher_bb": "bb_p", "pitcher_hr": "hr_p", "pitcher_outs": "outs",
}


def _american_to_decimal(o) -> float | None:
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if not o:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / (-o))


def _grade_leg(actual: float, line: float, side: str) -> int | None:
    if actual == line:
        return None  # push
    if side == "over":
        return 1 if actual > line else 0
    return 1 if actual < line else 0


def _load() -> pd.DataFrame:
    cp = pd.read_csv(CLOSING)
    box = pd.read_csv(BOX)
    # actual stat per (game_pk, player_id)
    box = box.drop_duplicates(["game_pk", "player_id"])
    key = ["game_pk", "player_id"]
    rows = []
    bx = box.set_index(key)
    for _, r in cp.iterrows():
        mkt = str(r["market"])
        if mkt == "hrr":
            parts = ["h", "runs_b", "rbi"]
        else:
            col = _STAT_COL.get(mkt)
            parts = [col] if col else None
        if not parts:
            continue
        try:
            brow = bx.loc[(int(r["game_pk"]), int(r["player_id"]))]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(brow, pd.DataFrame):
            brow = brow.iloc[0]
        if any(p not in brow or pd.isna(brow[p]) for p in parts):
            continue
        actual = float(sum(float(brow[p]) for p in parts))
        try:
            line = float(r["line"])
        except (TypeError, ValueError):
            continue
        # team of this player (for same-team vs opposing split)
        team = brow.get("team_id")
        for side, odds_col in (("over", "over"), ("under", "under")):
            res = _grade_leg(actual, line, side)
            if res is None:
                continue
            dec = _american_to_decimal(r.get(odds_col))
            if dec is None:
                continue
            try:
                am = float(r.get(odds_col))
            except (TypeError, ValueError):
                am = 0.0
            rows.append({
                "date": r["date"], "game_pk": int(r["game_pk"]),
                "player_id": int(r["player_id"]), "team_id": team,
                "market": mkt, "side": side, "line": line,
                "win": res, "dec": dec, "abs_am": abs(am),
                "is_bat": not mkt.startswith("pitcher"),
                "key": f"{mkt}_{side}",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep only the MAIN line per (game, player, market, side): the one whose
    # American odds are closest to even. Alt lines (extreme odds) otherwise
    # explode the pair count and muddy the correlation signal.
    df = (df.sort_values("abs_am")
            .drop_duplicates(["game_pk", "player_id", "market", "side"]))
    return df


def main():
    if not (CLOSING.exists() and BOX.exists()):
        print("Need closing_props.csv and box_2026.csv.")
        return
    legs = _load()
    if legs.empty:
        print("No gradable closing-prop legs matched to boxscores yet.")
        return

    # Per market/side base win rate (the 'independent' factor).
    base = legs.groupby("key")["win"].mean().to_dict()
    print(f"Graded legs: {len(legs)}  across "
          f"{legs['game_pk'].nunique()} games, {legs['date'].nunique()} dates")

    combos = []
    for (_d, _g), sub in legs.groupby(["date", "game_pk"]):
        recs = sub.to_dict("records")
        for a, b in combinations(recs, 2):
            if a["player_id"] == b["player_id"]:
                continue  # never pair two legs on the same player
            joint = a["win"] * b["win"]
            indep = base.get(a["key"], 0.0) * base.get(b["key"], 0.0)
            sides = "".join(sorted((a["side"][0], b["side"][0])))  # oo/ou/uu
            combos.append({
                "joint": joint, "dec": a["dec"] * b["dec"], "indep": indep,
                "same_team": a["team_id"] == b["team_id"],
                "both_bat": a["is_bat"] and b["is_bat"],
                "sides": sides,
            })
    if not combos:
        print("No same-game leg pairs available.")
        return
    cdf = pd.DataFrame(combos)

    def _report(label, d):
        d = d[np.isfinite(d["dec"])]
        n = len(d)
        if not n:
            print(f"  {label:30s} (no pairs)"); return
        realized = d["joint"].mean()
        indep = d["indep"].mean()
        lift = realized / indep if indep > 0 else float("nan")
        roi = (d["joint"] * d["dec"] - 1.0).mean()
        print(f"  {label:30s} n={n:6d}  joint {realized:5.1%}  indep {indep:5.1%}"
              f"  lift {lift:4.2f}x  ROI(mult) {roi:+6.1%}")

    print("\n=== Correlation + ROI (ROI uses multiplied odds = optimistic) ===")
    _report("ALL pairs", cdf)
    bat = cdf[cdf["both_bat"]]
    print("  -- batter-vs-batter pairs by side & team --")
    _report("same-team OVER/OVER", bat[(bat.same_team) & (bat.sides == "oo")])
    _report("same-team UNDER/UNDER", bat[(bat.same_team) & (bat.sides == "uu")])
    _report("same-team OVER/UNDER", bat[(bat.same_team) & (bat.sides == "ou")])
    _report("opposing OVER/OVER", bat[(~bat.same_team) & (bat.sides == "oo")])
    _report("opposing UNDER/UNDER", bat[(~bat.same_team) & (bat.sides == "uu")])

    print("\nRead: lift > 1 = legs hit together MORE than independent pricing "
          "assumes\n(exploitable). Thesis: same-team OVER/OVER should lift ABOVE "
          "1 (a hot\noffense pushes many hitters over at once); opposing OVER/OVER "
          "and same-team\nOVER/UNDER should lift BELOW 1. ROI uses multiplied odds "
          "— real SGP prices\nare lower, so treat sign of the lift, not the ROI "
          "magnitude, as the result.")


if __name__ == "__main__":
    main()
