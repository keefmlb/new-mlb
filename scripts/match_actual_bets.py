"""Match the user's ACTUAL Pikkit legs against the simulation board's ranks.

Answers: of the legs actually bet, how many sat inside the board's top 3, and
where did the LOSING legs come from? If losses cluster below rank 3, the board
already knew and the discipline fix is simply to stop reaching deeper.

Legs transcribed from Pikkit screenshots (Jul 30 - Aug 10 2026), MLB only.
"""
from __future__ import annotations
from collections import defaultdict
import json

LEGS = [
    ("2026-07-30", "Yandy Diaz", "prop_tb", 1), ("2026-07-30", "Burleson", "prop_tb", 1),
    ("2026-07-30", "Jordan Walker", "prop_tb", 1), ("2026-07-30", "Gonzales", "prop_tb", 1),
    ("2026-07-30", "Arraez", "prop_tb", 0), ("2026-07-30", "Freeman", "prop_tb", 1),
    ("2026-07-31", "Antonacci", "prop_tb", 1), ("2026-07-31", "Yainer Diaz", "prop_tb", 1),
    ("2026-07-31", "Caglianone", "prop_tb", 0), ("2026-07-31", "Rumfield", "prop_tb", 1),
    ("2026-07-31", "Arraez", "prop_tb", 1), ("2026-07-31", "Tatis", "prop_tb", 1),
    ("2026-08-01", "Burleson", "prop_tb", 1), ("2026-08-01", "Otto Lopez", "prop_tb", 1),
    ("2026-08-01", "Alvarez", "prop_tb", 1), ("2026-08-01", "Rumfield", "prop_tb", 1),
    ("2026-08-01", "McCarthy", "prop_tb", 1),
    ("2026-08-02", "Liberatore", "prop_pitcher_k", 1), ("2026-08-02", "Jax", "prop_pitcher_k", 1),
    ("2026-08-02", "Alcantara", "prop_pitcher_k", 1), ("2026-08-02", "Urena", "prop_pitcher_k", 1),
    ("2026-08-02", "Misiorowski", "prop_pitcher_k", 1), ("2026-08-02", "Roupp", "prop_pitcher_k", 1),
    ("2026-08-02", "Bennett", "prop_pitcher_k", 1),
    ("2026-08-03", "Burleson", "prop_tb", 1), ("2026-08-03", "Jung Hoo Lee", "prop_tb", 1),
    ("2026-08-03", "Ohtani", "prop_tb", 1), ("2026-08-03", "Freeman", "prop_tb", 1),
    ("2026-08-03", "McCarthy", "prop_tb", 1), ("2026-08-03", "Tatis", "prop_tb", 0),
    ("2026-08-04", "Rafaela", "prop_tb", 1), ("2026-08-04", "Freeman", "prop_tb", 1),
    ("2026-08-04", "Simpson", "prop_tb", 1), ("2026-08-04", "Aranda", "prop_tb", 1),
    ("2026-08-04", "Tatis", "prop_tb", 1), ("2026-08-04", "Marte", "prop_tb", 0),
    ("2026-08-05", "Christian Scott", "prop_pitcher_k", 1), ("2026-08-05", "Irvin", "prop_pitcher_k", 1),
    ("2026-08-05", "Warren", "prop_pitcher_k", 1), ("2026-08-05", "Harrison", "prop_pitcher_k", 1),
    ("2026-08-05", "Cameron", "prop_pitcher_k", 1), ("2026-08-05", "Skenes", "prop_pitcher_k", 1),
    ("2026-08-05", "Woo", "prop_pitcher_k", 1),
    ("2026-08-05", "Pena", "prop_tb", 0), ("2026-08-05", "Freeman", "prop_tb", 1),
    ("2026-08-05", "Jung Hoo Lee", "prop_tb", 1), ("2026-08-05", "Nimmo", "prop_tb", 1),
    ("2026-08-05", "Rumfield", "prop_tb", 0), ("2026-08-05", "Otto Lopez", "prop_tb", 0),
    ("2026-08-06", "Barnett", "prop_pitcher_k", 1), ("2026-08-06", "Ashcraft", "prop_pitcher_k", 1),
    ("2026-08-06", "Valdez", "prop_pitcher_k", 1), ("2026-08-06", "Suarez", "prop_pitcher_k", 0),
    ("2026-08-06", "Junk", "prop_pitcher_k", 0), ("2026-08-06", "Ober", "prop_pitcher_k", 0),
    ("2026-08-06", "Buehler", "prop_pitcher_k", 1),
    ("2026-08-06", "Meckler", "prop_tb", 1), ("2026-08-06", "Tommy White", "prop_tb", 1),
    ("2026-08-06", "Gonzales", "prop_tb", 1), ("2026-08-06", "Mangum", "prop_tb", 1),
    ("2026-08-06", "Otto Lopez", "prop_tb", 1), ("2026-08-06", "Tatis", "prop_tb", 1),
    ("2026-08-07", "Cavalli", "prop_pitcher_k", 1), ("2026-08-07", "Phillips", "prop_pitcher_k", 1),
    ("2026-08-07", "Perkins", "prop_pitcher_k", 0), ("2026-08-07", "Messick", "prop_pitcher_k", 1),
    ("2026-08-07", "Schultz", "prop_pitcher_k", 0), ("2026-08-07", "Baz", "prop_pitcher_k", 1),
    ("2026-08-07", "Rasmussen", "prop_pitcher_k", 1),
    ("2026-08-07", "Gonzales", "prop_tb", 1), ("2026-08-07", "Abrams", "prop_tb", 1),
    ("2026-08-07", "Hoerner", "prop_tb", 0), ("2026-08-07", "Burleson", "prop_tb", 0),
    ("2026-08-07", "Jordan Walker", "prop_tb", 1), ("2026-08-07", "Jung Hoo Lee", "prop_tb", 1),
    ("2026-08-09", "Rafaela", "prop_tb", 0), ("2026-08-09", "Otto Lopez", "prop_tb", 1),
    ("2026-08-09", "Meckler", "prop_tb", 1), ("2026-08-09", "Witt", "prop_tb", 0),
    ("2026-08-09", "Burleson", "prop_tb", 1), ("2026-08-09", "Jung Hoo Lee", "prop_tb", 0),
    ("2026-08-10", "Gray", "prop_pitcher_k", 1), ("2026-08-10", "Elder", "prop_pitcher_k", 1),
    ("2026-08-10", "Gore", "prop_pitcher_k", 1), ("2026-08-10", "Detmers", "prop_pitcher_k", 1),
    ("2026-08-10", "Tidwell", "prop_pitcher_k", 1), ("2026-08-10", "Cameron", "prop_pitcher_k", 0),
    ("2026-08-10", "Rafaela", "prop_tb", 1), ("2026-08-10", "Burleson", "prop_tb", 1),
    ("2026-08-10", "Herrera", "prop_tb", 1), ("2026-08-10", "Yandy Diaz", "prop_tb", 1),
    ("2026-08-10", "Jung Hoo Lee", "prop_tb", 1), ("2026-08-10", "Alvarez", "prop_tb", 1),
]


def main() -> None:
    picks = json.load(open("data/bets/sim_picks.json", encoding="utf-8"))
    board = defaultdict(list)
    for e in picks:
        if e.get("market") in ("prop_tb", "prop_pitcher_k"):
            board[(e.get("date"), e.get("market"))].append(e)

    def rank_of(date, player, market):
        """Use the STORED rank — that is the position the app actually showed.

        Re-deriving rank by sorting on sim_hit is wrong: _diversify_lines
        reorders and caps the board before it is logged, so a raw-probability
        sort produces positions the user never saw. That error put a leg the
        user did bet at 'rank 25' when he has never picked past 6.
        """
        key = player.lower()
        best = None
        for e in board.get((date, market), []):
            if key in (e.get("description") or "").lower():
                r = e.get("rank")
                if r is None:
                    continue
                if best is None or r < best:
                    best = int(r)
        return best

    res = defaultdict(lambda: [0, 0])
    unmatched = []
    losers = []
    for d, p, m, w in LEGS:
        r = rank_of(d, p, m)
        if r is None:
            unmatched.append((d, p, m, w))
            continue
        bucket = "top3" if r <= 3 else ("4-6" if r <= 6 else "7+")
        res[(m, bucket)][0] += w
        res[(m, bucket)][1] += 1
        if not w:
            losers.append((d, p, m, r))

    print("LOSING LEGS and where they ranked on the board\n")
    for d, p, m, r in sorted(losers, key=lambda x: x[3]):
        tag = "TOP 3" if r <= 3 else ("rank 4-6" if r <= 6 else f"rank {r}")
        print(f"  {d}  {p[:20]:20s} {m.replace('prop_',''):10s} rank {r:3d}   {tag}")

    print(f"\nHIT RATE BY BOARD RANK BUCKET (legs actually bet)\n")
    print(f"{'market':12s} {'bucket':7s} {'W':>3s} {'n':>3s}   hit%")
    for (m, b), (w, n) in sorted(res.items()):
        print(f"{m.replace('prop_',''):12s} {b:7s} {w:3d} {n:3d}   {w/n*100:5.1f}%")

    tot_t3 = [sum(res[(m,'top3')][i] for m in ('prop_tb','prop_pitcher_k')) for i in (0,1)]
    tot_out = [sum(res[(m,b)][i] for m in ('prop_tb','prop_pitcher_k')
                   for b in ('4-6','7+')) for i in (0,1)]
    if tot_t3[1]:
        print(f"\n  TOP 3 overall : {tot_t3[0]}/{tot_t3[1]} = {tot_t3[0]/tot_t3[1]*100:.1f}%")
    if tot_out[1]:
        print(f"  BELOW TOP 3   : {tot_out[0]}/{tot_out[1]} = {tot_out[0]/tot_out[1]*100:.1f}%")
    print(f"\n  unmatched (not on that day's board at all): {len(unmatched)}/{len(LEGS)}")
    for d, p, m, w in unmatched[:12]:
        print(f"    {d} {p[:20]:20s} {m.replace('prop_','')}  {'W' if w else 'L'}")


if __name__ == "__main__":
    main()
