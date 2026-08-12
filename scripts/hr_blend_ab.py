"""A/B the batter HR projection: ML-blended (production) vs pure analytical.

The ground-truth audit shows proj_hr running ~15% above actual, and HR feeds
total bases at weight 4, so it is the single largest remaining contributor to
TB over-projection. `train_props` tunes HR toward pure ML on a holdout MAE
criterion; MAE is bias-blind, so a model that is systematically high can still
win that contest. This grades both variants on the mean, which is what a
counting prop settles on.

Usage: python -m scripts.hr_blend_ab [n_days]
"""
from __future__ import annotations
import sys

import pandas as pd

from src import predict_core, projections


def main(n_days: int = 12) -> None:
    box = pd.read_csv("data/games/box_2026.csv")
    box = box[box["pa"].fillna(0) > 0]
    dates = sorted(box["date"].dropna().unique())[-n_days:]
    act = {(int(r.game_pk), int(r.player_id)): r for r in box.itertuples()}

    variants = {"production": None, "pure_analytical": 0.0, "pure_ml": 1.0}
    tot = {k: [0.0, 0.0, 0] for k in variants}     # [proj, actual, n]

    for d in dates:
        for name, w in variants.items():
            projections.BLEND_WEIGHT_OVERRIDE.pop("hr", None)
            if w is not None:
                projections.BLEND_WEIGHT_OVERRIDE["hr"] = w
            try:
                slate = predict_core.predict_slate(str(d)[:10], fetch_odds=False)
            except Exception:                       # noqa: BLE001
                continue
            for g in slate.games:
                for side in (g.away_batters or [], g.home_batters or []):
                    for b in side[:9]:
                        pid = b.get("player_id")
                        if pid is None:
                            continue
                        a = act.get((int(g.game_pk), int(pid)))
                        if a is None or pd.isna(getattr(a, "hr", None)):
                            continue
                        t = tot[name]
                        t[0] += float(b.get("proj_hr") or 0.0)
                        t[1] += float(a.hr)
                        t[2] += 1
        print(f"  {str(d)[:10]} done")
    projections.BLEND_WEIGHT_OVERRIDE.pop("hr", None)

    print(f"\nHR BLEND A/B — {len(dates)} dates\n")
    print(f"{'variant':18s} {'proj':>7s} {'actual':>7s} {'proj/act':>9s} {'n':>7s}")
    for name in variants:
        p, a, c = tot[name]
        if not c or not a:
            continue
        print(f"{name:18s} {p/c:7.4f} {a/c:7.4f} {p/a:9.3f} {c:7d}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
