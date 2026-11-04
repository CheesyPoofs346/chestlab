"""Two players, 60 s each, one chest never taken twice: both start together (within GAP_M of each other, they land at the same
place), then go different ways to take the most chests between them. Quiet places only (see fast_routes.py).
    python tools/pair_routes.py [budget_s [gap_m]]   -> assets/pair_routes.json   (draw with: python tools/render_route.py <rank> pair_routes.json)"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chestlab import route as R
from fast_routes import load_pool

BUDGET, GAP_M = 60, 30.0
QUICK, LONG, FINALISTS, SEEDS, TOP = 40, 1500, 20, (0, 1, 2), 3


def main(budget=BUDGET, gap_m=GAP_M):
    kind, pts, P, v, pois, allp, names = load_pool()
    print(len(P), "chests in the pool", flush=True)
    quick = []
    for s in range(len(P)):
        d = np.hypot(*(P - P[s]).T)
        close = [t for t in np.argsort(d)[1:] if d[t] <= gap_m]
        if not close:
            continue
        for t in sorted({close[0], close[len(close) // 2], close[-1]}):             # a near, a middle and a far partner chest
            quick.append((R.best_pair(P, v, s, int(t), budget, iters=QUICK)[0], s, int(t)))
    quick.sort(reverse=True)
    print("stage 1:", len(quick), "start pairs, best", round(quick[0][0], 1), flush=True)
    finals = []
    for _, s, t in quick[:FINALISTS]:
        finals.append(max((R.best_pair(P, v, s, t, budget, iters=LONG, seed=sd, rounds=4) for sd in SEEDS), key=lambda r: round(r[0], 6)))
    finals.sort(key=lambda r: -r[0])
    out, seen = [], []
    for val, a, b in finals:
        c = P[a + b].mean(axis=0)
        if any(np.hypot(*(c - c2)) < 150 for c2 in seen):
            continue
        seen.append(c)
        players = []
        for o in (a, b):
            walk = np.hypot(*(P[o][1:] - P[o][:-1]).T)
            players.append(dict(chests=len(o), guaranteed=int((kind[o] != "chest").sum()), expected=round(float(v[o].sum()), 1), seconds=round(R.route_time(P, o) - R.OVERHEAD_S, 1),
                                walk_m=round(float(walk.sum())), start_units=[round(float(x), 2) for x in pts[o[0]]],
                                path=[dict(n=i + 1, kind=str(kind[j]), unit=[round(float(x), 2) for x in pts[j]], leg_m=round(float(walk[i - 1]), 1) if i else 0.0) for i, j in enumerate(o)]))
        near = int(np.argmin(np.hypot(*(allp - c).T)))
        t = max(p["seconds"] for p in players)
        out.append(dict(budget_s=budget, scope="pair", chests=sum(p["chests"] for p in players), guaranteed=sum(p["guaranteed"] for p in players), expected=round(float(val), 1),
                        seconds=t, nearest_place=names[near], nearest_place_m=round(float(np.hypot(*(allp[near] - c).T))),
                        nearest_named_m=round(float(min(np.hypot(*(pois - p).T).min() for p in P[a + b]))), players=players,
                        start_gap_m=round(float(np.hypot(*(P[a[0]] - P[b[0]])))), chests_per_min_with_reset=round(60 * float(val) / (t + 40), 1)))
        o = out[-1]
        print("#%d: %d chests together (%d always spawn, %.1f expected) = %.1f each | P1 %d/%d sure %d m, P2 %d/%d sure %d m | start gap %d m | near %s (%d m), named POI %d m | %.1f/min incl. 40 s leave+rejoin" % (
            len(out), o["chests"], o["guaranteed"], o["expected"], o["expected"] / 2, players[0]["chests"], players[0]["guaranteed"], players[0]["walk_m"],
            players[1]["chests"], players[1]["guaranteed"], players[1]["walk_m"], o["start_gap_m"], o["nearest_place"], o["nearest_place_m"], o["nearest_named_m"], o["chests_per_min_with_reset"]), flush=True)
        if len(out) == TOP:
            break
    with open(os.path.join(ROOT, "assets", "pair_routes.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:3]))
