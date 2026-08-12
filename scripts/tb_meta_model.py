"""TB meta-model: line value, batting-order slot, opponent, park.

The generic screen (scripts/sim_meta_model.py) found a strong adverse-selection
lever for pitcher strikeouts but nothing for total bases. It was missing the one
feature most likely to matter for a hitter: WHERE HE BATS. Batting slot drives
plate appearances almost mechanically (a leadoff man gets ~0.8 more PA than the
9-hole), and TB is a counting stat — more PA, more chances.

Batting order is recoverable pre-game from games_2026.csv (home/away_lineup_ids
are pipe-separated player ids in order), so this joins it in properly along
with opponent quality and park/weather, then asks the same question: among
picks the sim liked, what separated the hits from the misses?

Only pre-game fields are used.

Run:  python -m scripts.tb_meta_model [market] [min_conf]
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bet_tracker import roi_ci  # noqa: E402

PICKS = ROOT / "data" / "bets" / "sim_picks.json"
GAMES = ROOT / "data" / "games" / "games_2026.csv"


def _prob(a):
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)


def _dec(a):
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))


def _usable(r):
    o = r.get("odds")
    return (r.get("outcome") in ("W", "L")
            and isinstance(o, (int, float)) and o and not (-100 < o < 100))


def _slot_map(games: pd.DataFrame) -> dict:
    """(game_pk, player_id) -> (batting_slot 1-9, is_home)."""
    out = {}
    for gpk, hrow, arow in zip(games.index, games["home_lineup_ids"],
                               games["away_lineup_ids"]):
        for ids, is_home in ((hrow, True), (arow, False)):
            if not isinstance(ids, str):
                continue
            for i, pid in enumerate(ids.split("|"), start=1):
                pid = pid.strip()
                if pid.isdigit():
                    out[(int(gpk), int(pid))] = (i, is_home)
    return out


def build(market: str, min_conf: float) -> pd.DataFrame:
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    games = pd.read_csv(GAMES).drop_duplicates("game_pk").set_index("game_pk")
    slots = _slot_map(games)
    rows = []
    for r in picks:
        if not _usable(r) or r.get("market") != market:
            continue
        raw = r.get("sim_hit_raw", r.get("sim_hit"))
        if raw is None or float(raw) < min_conf:
            continue
        gpk, pid = r.get("game_pk"), r.get("player_id")
        try:
            g = games.loc[int(gpk)]
        except Exception:
            continue
        slot, is_home = slots.get((int(gpk), int(pid or 0)), (None, None))
        if slot is None:
            continue
        opp_sp = "away_sp" if is_home else "home_sp"
        num = lambda k: pd.to_numeric(g.get(k), errors="coerce")  # noqa: E731
        imp = _prob(r["odds"])
        rows.append({
            "won": 1 if r["outcome"] == "W" else 0,
            "prof": (_dec(r["odds"]) - 1.0) if r["outcome"] == "W" else -1.0,
            "slot": float(slot),
            "top_of_order": 1.0 if slot <= 4 else 0.0,
            "sim_raw": float(raw),
            "rank": float(r.get("rank") or 10),
            "implied": imp,
            "edge": float(raw) - imp,
            "is_home": 1.0 if is_home else 0.0,
            "opp_sp_fip": num(f"{opp_sp}_fip"),
            "opp_sp_xfip": num(f"{opp_sp}_xfip"),
            "opp_sp_k9": num(f"{opp_sp}_k9"),
            "opp_bp_fip": num("away_bp_fip" if is_home else "home_bp_fip"),
            "park_pf_runs": num("park_pf_runs"),
            "park_pf_hr": num("park_pf_hr"),
            "park_elev": num("park_elev_ft"),
            "temp_f": num("temp_f"),
            "wind_to_cf": num("wind_to_cf_mph"),
            "runs_mult": num("runs_mult"),
        })
    return pd.DataFrame(rows)


def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "prop_tb"
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    df = build(market, min_conf)
    if df.empty:
        print("No rows (need lineups + graded picks)."); return
    hit, miss = df[df.won == 1], df[df.won == 0]
    print(f"{market}, sim raw >= {min_conf:.0%}   n={len(df)}  "
          f"HIT={len(hit)} ({len(hit)/len(df):.1%})  MISS={len(miss)}")

    print(f"\n{'feature':15s} {'HIT':>9} {'MISS':>9} {'gap':>9} {'z':>7}")
    print("-" * 54)
    scored = []
    for f in [c for c in df.columns if c not in ("won", "prof")]:
        a, b = hit[f].dropna(), miss[f].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if not se:
            continue
        scored.append((abs((a.mean() - b.mean()) / se), f, a.mean(), b.mean(),
                       (a.mean() - b.mean()) / se))
    scored.sort(reverse=True)
    for _, f, am, bm, z in scored:
        mark = "  <--" if abs(z) >= 2 else ""
        print(f"{f:15s} {am:9.3f} {bm:9.3f} {am-bm:+9.3f} {z:+7.2f}{mark}")

    print("\nBATTING SLOT — hit rate and ROI (the feature the first screen "
          "was missing)")
    for lo, hi, lab in ((1, 2, "1-2 (top)"), (3, 4, "3-4 (heart)"),
                        (5, 6, "5-6 (middle)"), (7, 9, "7-9 (bottom)")):
        s = df[(df.slot >= lo) & (df.slot <= hi)]
        if len(s) < 12:
            continue
        ci = roi_ci(list(s.prof))
        cis = f"[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
        print(f"  slot {lab:12s} n={len(s):3d}  hit {s.won.mean():5.1%}  "
              f"simSaid {s.sim_raw.mean():5.1%}  ROI {s.prof.mean():+7.1%} {cis}")

    print("\nTOP-5 PICKS ONLY, by slot")
    t5 = df[df["rank"] <= 5]
    for lo, hi, lab in ((1, 4, "1-4"), (5, 9, "5-9")):
        s = t5[(t5.slot >= lo) & (t5.slot <= hi)]
        if len(s) < 8:
            continue
        ci = roi_ci(list(s.prof))
        cis = f"[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
        print(f"  slot {lab:5s} n={len(s):3d}  hit {s.won.mean():5.1%}  "
              f"ROI {s.prof.mean():+7.1%} {cis}")

    print("\nCAVEAT: many features screened; |z|>=2 appears by chance. Act only "
          "on a lever\nthat is monotone AND has a mechanism.")


if __name__ == "__main__":
    main()
