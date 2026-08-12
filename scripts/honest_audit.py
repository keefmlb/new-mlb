"""Honest read: does the model beat the price, market by market, on CLEAN data.

Run after the Aug 3 2026 purge of corrupted-odds rows. Three questions per
market, in increasing order of what actually matters:

  1. CALIBRATION  — when the model says X%, does it happen X% of the time?
  2. SKILL        — is the model's probability a BETTER forecast than the
                    book's implied probability? (Brier, lower = better)
  3. MONEY        — realized flat-stake ROI with a bootstrap CI.

A market can be well calibrated and still unprofitable (the price already
knows). Only #2 winning AND #3 positive is a real edge. #1 alone is not.

CAVEAT: implied probability here is vig-inclusive (single-sided quotes), so it
is biased HIGH, which flatters the model in the Brier comparison. If the model
loses on Brier even with that handicap, the result is unambiguous.

Run:  python -m scripts.honest_audit
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bet_tracker import roi_ci, wilson_ci  # noqa: E402

SIM = ROOT / "data" / "bets" / "sim_picks.json"
BETS = ROOT / "data" / "bets" / "bet_log.json"


def _prob(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if not a or -100 < a < 100:
        return None
    return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)


def _dec(a):
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))


def _load(path, prob_key):
    rows = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return rows
    for r in data:
        if r.get("outcome") not in ("W", "L"):
            continue
        p = r.get(prob_key)
        imp = _prob(r.get("odds"))
        if p is None or imp is None:
            continue
        rows.append({
            "market": r.get("market", "?"),
            "p": float(p), "imp": imp,
            "won": 1 if r["outcome"] == "W" else 0,
            "profit": (_dec(r["odds"]) - 1.0) if r["outcome"] == "W" else -1.0,
        })
    return rows


def _report(title, rows, min_n=25):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    print(f"  {'market':20s} {'n':>5} {'hit':>6} {'model':>6} {'price':>6} "
          f"{'calGap':>7} {'BrierM':>7} {'BrierP':>7} {'skill':>7} {'ROI':>8}")
    by = {}
    for r in rows:
        by.setdefault(r["market"], []).append(r)
    by["ALL"] = rows
    for mkt, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        if n < min_n and mkt != "ALL":
            continue
        hit = sum(r["won"] for r in rs) / n
        mp = sum(r["p"] for r in rs) / n
        ip = sum(r["imp"] for r in rs) / n
        bm = sum((r["p"] - r["won"]) ** 2 for r in rs) / n
        bp = sum((r["imp"] - r["won"]) ** 2 for r in rs) / n
        roi = sum(r["profit"] for r in rs) / n
        skill = bp - bm                      # >0 => model better than price
        flag = "  <-- model better" if skill > 0 else ""
        print(f"  {mkt:20s} {n:5d} {hit:6.1%} {mp:6.1%} {ip:6.1%} "
              f"{(hit-mp)*100:+6.1f}p {bm:7.4f} {bp:7.4f} {skill:+7.4f} "
              f"{roi:+7.1%}{flag}")
    # bootstrap CI on the pooled ROI
    ci = roi_ci([r["profit"] for r in rows])
    w = wilson_ci(sum(r["won"] for r in rows), len(rows))
    print(f"\n  POOLED n={len(rows)}  hit CI [{w[0]:.1%},{w[1]:.1%}]"
          + (f"  ROI CI [{ci[0]:+.1%},{ci[1]:+.1%}]" if ci else ""))


def main():
    sim = _load(SIM, "sim_hit")
    bets = _load(BETS, "model_prob")
    if sim:
        _report("SIM LEADERBOARD PICKS — sim probability vs the price", sim)
    if bets:
        _report("LOGGED VALUE BETS — model probability vs the price", bets)
    print("\nREAD: calGap = actual hit minus what the model claimed (negative = "
          "overconfident).\nskill = Brier(price) - Brier(model); POSITIVE means "
          "the model forecasts better\nthan the price. Profit requires skill > 0 "
          "AND enough of it to clear the vig.")


if __name__ == "__main__":
    main()
