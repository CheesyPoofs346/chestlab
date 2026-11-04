"""The most chests you can collect in a fixed time (60 / 75 / 90 / 120 / 180 s of walking and opening, no vaults, no digging) in
quiet places only: the pool is just the chests that are >= 100 m from every named location, so every route stays out of them.
Two stages: every chest of the pool as the start with a quick local search, then the best 25 starts with a long one.
    python tools/fast_routes.py [quiet_m]   -> assets/fast_routes.json"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chestlab import route as R
from chestlab.worldmap import KINDS
from lowkey_routes import places

KEEP = ("chest_sure", "rare", "sprite", "chest")          # no vaults, no buried chests (digging)
BUDGETS = (60, 75, 90, 120, 180)
QUIET_M = 100.0                                          # every chest of a route at least this far from every named location (they are ~150 m wide, so this is outside them)
QUICK, LONG, FINALISTS, SEEDS = 150, 3000, 25, (0, 1, 2)


def load_pool(quiet_m=QUIET_M):
    """The chests a quiet route may use: (kind, map units, metres, expected value per chest, named locations in metres, every
    place in metres, place names). Sets up the planner (terrain, no extra seconds)."""
    data = json.load(open(os.path.join(ROOT, "assets", "world_chests.json")))
    kind = np.array([k for k in KINDS if k in KEEP for _ in data[k]])
    pts = np.array([p for k in KINDS if k in KEEP for p in data[k]], float) / 16.0
    R.EXTRA = None
    R.use_terrain()
    pl = places()
    pois = np.array([[x, y] for n, k, x, y in pl if k == "poi"]) * R.MPU
    allp = np.array([[x, y] for n, k, x, y in pl]) * R.MPU
    names = [n for n, k, x, y in pl]
    far = np.array([np.hypot(*(pois - p).T).min() for p in R.metres(pts)]) >= quiet_m
    kind, pts = kind[far], pts[far]
    return kind, pts, R.metres(pts), np.array([R.VALUE[k] for k in kind]), pois, allp, names


def main(quiet_m=QUIET_M):
    kind, pts, P, v, pois, allp, names = load_pool(quiet_m)
    print(len(P), "chests are >= %d m from every named location" % quiet_m, flush=True)
    out = []
    for B in BUDGETS:
        quick = sorted(((R.best_budget(P, v, s, B, iters=QUICK), s) for s in range(len(P))), key=lambda t: -t[0][0])
        finals = [R.best_budget(P, v, s, B, iters=LONG, seed=sd) for _, s in quick[:FINALISTS] for sd in SEEDS] + [q for q, _ in quick[:FINALISTS]]
        val, o, t = max(finals, key=lambda r: (round(r[0], 6), int((kind[r[1]] != "chest").sum()), -r[2]))
        c = P[o].mean(axis=0)
        dpoi = float(min(np.hypot(*(pois - p).T).min() for p in P[o]))        # the route's closest chest to any named location
        near = int(np.argmin(np.hypot(*(allp - c).T)))
        walk = np.hypot(*(P[o][1:] - P[o][:-1]).T)
        path = [dict(n=i + 1, kind=str(kind[j]), unit=[round(float(x), 2) for x in pts[j]], leg_m=round(float(walk[i - 1]), 1) if i else 0.0) for i, j in enumerate(o)]
        out.append(dict(budget_s=B, scope="quiet", chests=len(o), guaranteed=int((kind[o] != "chest").sum()), expected=round(float(val), 1), seconds=round(t, 1),
                        walk_m=round(float(walk.sum())), nearest_place=names[near], nearest_place_m=round(float(np.hypot(*(allp[near] - c).T))),
                        nearest_named_m=round(dpoi), start_units=path[0]["unit"], path=path,
                        chests_per_min_with_reset=round(60 * val / (t + 40), 1)))
        print("%3d s: %2d chests (%2d always spawn, %4.1f expected), %5.1f s, walk %3d m | near %s (%d m), nearest named POI %d m | %.1f/min incl. 40 s leave+rejoin" % (
            B, len(o), out[-1]["guaranteed"], val, t, out[-1]["walk_m"], names[near], out[-1]["nearest_place_m"], dpoi, out[-1]["chests_per_min_with_reset"]), flush=True)
    with open(os.path.join(ROOT, "assets", "fast_routes.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:2]))
