"""Market-anchor experiment: is `book line + α·(projection − line)` a better
predictor than either alone?

The Jun 11 2026 benchmark showed the book's pitcher-K closing line out-
predicts our projection (MAE 2.07 vs 2.27) while our disagreement with the
line carries weakly-positive signal (r=+0.115, n=49). If that correlation is
real, the optimal predictor anchors on the line and adds a fraction α of our
disagreement:

    anchored = line + α · (proj − line)

α is fit per market by least squares of (actual − line) on (proj − line),
no intercept. α=0 means "the line is everything, our model adds nothing";
α=1 means "ignore the line". This script is the INSTRUMENT — it re-fits as
closing_props.csv accumulates and only writes data/models/market_anchor.json
once a market clears the evidence gate:

    n ≥ 150 two-sided rows AND anchored beats the line alone out-of-sample
    (last 30% of dates).

Nothing in live pricing consumes the JSON yet — wiring it in is a policy
change to make deliberately once a market passes the gate.

Data: closing_props.csv (TWO-SIDED rows only — one-sided "Yes" alt lines are
price points, not market medians) joined to props_{bat,pit}_2026.csv by
(game_pk, player_id) for the leak-free analytical projection and the actual.

Run:  python -m scripts.fit_market_anchor
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "models" / "market_anchor.json"
CLOSING = ROOT / "data" / "odds" / "closing_props.csv"

MKT = {
    "hits": ("bat", "proj_h", "h"), "hr": ("bat", "proj_hr", "hr"),
    "tb": ("bat", "proj_tb", "tb"), "rbi": ("bat", "proj_rbi", "rbi"),
    "runs": ("bat", "proj_runs", "runs_b"), "k": ("bat", "proj_k", "k_b"),
    "bb": ("bat", "proj_bb", "bb_b"),
    "pitcher_k": ("pit", "proj_k", "k_p"), "pitcher_bb": ("pit", "proj_bb", "bb_p"),
    "pitcher_h": ("pit", "proj_h", "h_p"), "pitcher_er": ("pit", "proj_er", "er"),
    "pitcher_hr": ("pit", "proj_hr", "hr_p"),
    "pitcher_outs": ("pit", "expected_outs", "outs"),
}

MIN_N = 150
HOLDOUT_FRAC = 0.30


def _build_rows() -> pd.DataFrame:
    cp = pd.read_csv(CLOSING)
    # two-sided rows only: both prices present
    cp = cp[cp["over"].notna() & cp["under"].notna()
            & (cp["over"].astype(str) != "") & (cp["under"].astype(str) != "")]
    bat = pd.read_csv(ROOT / "data" / "games" / "props_bat_2026.csv")
    pit = pd.read_csv(ROOT / "data" / "games" / "props_pit_2026.csv")
    ix = {"bat": bat.set_index(["game_pk", "player_id"]),
          "pit": pit.set_index(["game_pk", "player_id"])}
    rows = []
    for r in cp.itertuples():
        m = MKT.get(r.market)
        if not m:
            continue
        kind, pcol, acol = m
        try:
            pr = ix[kind].loc[(int(r.game_pk), int(r.player_id))]
        except (KeyError, ValueError, TypeError):
            continue
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        proj, act = pr.get(pcol), pr.get(acol)
        try:
            line = float(r.line)
        except (TypeError, ValueError):
            continue
        if pd.isna(proj) or pd.isna(act):
            continue
        rows.append({"date": str(r.date), "market": r.market,
                     "proj": float(proj), "act": float(act), "line": line})
    return pd.DataFrame(rows)


def main():
    if not CLOSING.exists():
        print("No closing_props.csv yet.")
        return
    df = _build_rows()
    print(f"two-sided closing rows joined to projections+actuals: {len(df)}")
    if df.empty:
        return

    out: dict = {}
    print(f"\n{'market':14s} {'n':>5s} {'corr':>6s} {'MAE line':>9s} {'MAE proj':>9s} "
          f"{'alpha':>6s} {'MAE anch OOS':>12s} {'MAE line OOS':>12s}  status")
    for mkt, sub in df.groupby("market"):
        sub = sub.sort_values("date")
        dis = sub["proj"] - sub["line"]
        res = sub["act"] - sub["line"]
        n = len(sub)
        corr = float(np.corrcoef(dis, res)[0, 1]) if n > 5 and dis.std() > 0 else float("nan")
        mae_line = float((sub["act"] - sub["line"]).abs().mean())
        mae_proj = float((sub["act"] - sub["proj"]).abs().mean())
        status = "accumulating"
        alpha = mae_anch_oos = mae_line_oos = None
        if n >= MIN_N:
            cut = int(n * (1 - HOLDOUT_FRAC))
            tr, te = sub.iloc[:cut], sub.iloc[cut:]
            d_tr = (tr["proj"] - tr["line"]).to_numpy()
            r_tr = (tr["act"] - tr["line"]).to_numpy()
            denom = float((d_tr ** 2).sum())
            if denom > 0:
                alpha = float((d_tr * r_tr).sum() / denom)
                anch_te = te["line"] + alpha * (te["proj"] - te["line"])
                mae_anch_oos = float((te["act"] - anch_te).abs().mean())
                mae_line_oos = float((te["act"] - te["line"]).abs().mean())
                if mae_anch_oos < mae_line_oos:
                    status = "PASSES GATE"
                    # final alpha on all data
                    d = dis.to_numpy(); rr = res.to_numpy()
                    alpha = float((d * rr).sum() / (d ** 2).sum())
                    out[mkt] = {"alpha": alpha, "n": n,
                                "mae_line_oos": mae_line_oos,
                                "mae_anchored_oos": mae_anch_oos,
                                "corr": corr}
                else:
                    status = "fails gate (line alone better OOS)"
        print(f"{mkt:14s} {n:5d} {corr:+6.2f} {mae_line:9.3f} {mae_proj:9.3f} "
              f"{alpha if alpha is not None else float('nan'):6.2f} "
              f"{mae_anch_oos if mae_anch_oos is not None else float('nan'):12.3f} "
              f"{mae_line_oos if mae_line_oos is not None else float('nan'):12.3f}  {status}")

    if out:
        out["method"] = ("alpha: LS of (act-line) on (proj-line), no intercept; "
                         "gate: n>=150 two-sided rows AND anchored beats line OOS "
                         "(last 30% of dates). NOT consumed by live pricing yet.")
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved gated markets -> {OUT}")
    else:
        print(f"\nNo market passes the gate yet (need {MIN_N}+ two-sided closes "
              "per market). Re-run as closing_props.csv accumulates — every "
              "slate run near first pitch adds a day.")


if __name__ == "__main__":
    main()
