"""Best chest routes in quiet places: far from every named location (where everyone lands), ranked by chests per minute.
    python tools/lowkey_routes.py [min_distance_from_named_pois_m [min_units_from_map_centre]]   -> assets/lowkey_routes.json, out/lowkey_*.png"""
import json, os, re, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import route as R
from chestlab.worldmap import KINDS


def places():
    """fortnite.gg named locations (big POIs) and landmarks (small ones): [(name, kind, x_unit, y_unit)]."""
    t = open(os.path.join(ROOT, "assets", "gg", "en.js"), encoding="utf-8").read()
    i = t.index('"poi":{"name":"Locations"')
    lm = t.index('"landmarks"')
    out = []
    for m in re.finditer(r'\{"name":"([^"]+)","coords":\[(-?[\d.]+),(-?[\d.]+)\]\}', t[i:]):
        out.append((m.group(1).title(), "poi" if i + m.start() < lm else "landmark", float(m.group(3)), -float(m.group(2))))   # (lat, lng) -> x = lng, y = -lat
    return out


def main(min_poi_m=300.0, min_edge=0.0):
    data = json.load(open(os.path.join(ROOT, "assets", "world_chests.json")))
    kind = np.array([k for k in KINDS for _ in data[k]])
    pts = np.array([p for k in KINDS for p in data[k]], float) / 16.0
    P = R.metres(pts)
    v = np.array([R.VALUE[k] for k in kind])
    R.EXTRA = np.where(kind == "buried", R.DIG_S, np.where(kind == "vault", R.VAULT_S, 0.0))
    R.use_terrain()                                  # swimming costs time
    pl = places()
    pois = np.array([[x, y] for n, k, x, y in pl if k == "poi"]) * R.MPU
    allp = np.array([[x, y] for n, k, x, y in pl]) * R.MPU
    names = [n for n, k, x, y in pl]
    print(len(pl), "places;", int(sum(k == "poi" for _, k, _, _ in pl)), "named locations")
    found = []
    for s in range(len(P)):
        r, o = R.best_from(P, v, s)
        if len(o) >= 6:
            found.append((r, o))
    found.sort(key=lambda t: -t[0])
    out, seen = [], []
    for r, o in found:
        c = P[o].mean(axis=0)
        dpoi = float(np.hypot(*(pois - c).T).min())
        edge0 = float(np.hypot(*(pts[o].mean(axis=0) - 128.0)))
        if dpoi < min_poi_m or edge0 < min_edge or any(np.hypot(*(c - c2)) < 250 for c2 in seen):
            continue
        seen.append(c)
        near = int(np.argmin(np.hypot(*(allp - c).T)))
        edge = float(np.hypot(*(pts[o].mean(axis=0) - 128.0)))                 # map units from the centre (the map is 256 across)
        path = P[o]
        legs = np.hypot(*(path[1:] - path[:-1]).T)
        out.append(dict(rank=len(out) + 1, chests_per_min=round(60 * r, 1), chests=len(o), expected=round(float(v[o].sum()), 1), guaranteed=int(np.isin(kind[o], ("chest_sure", "rare", "sprite")).sum()), vault=int((kind[o] == "vault").sum()),
                        no_vault_per_min=round(60 * float(v[o][kind[o] != "vault"].sum() / R.route_time(P, o)), 1),
                        buried=int((kind[o] == "buried").sum()), seconds=round(R.route_time(P, o)), walk_m=round(float(legs.sum())),
                        nearest_place=names[near], nearest_place_m=round(float(np.hypot(*(allp[near] - c))), 0), nearest_named_m=round(dpoi),
                        from_centre_units=round(edge, 1), start_units=[round(float(x), 2) for x in pts[o[0]]],
                        path=[dict(n=k + 1, kind=str(kind[i]), unit=[round(float(x), 2) for x in pts[i]], leg_m=round(float(legs[k - 1]), 1) if k else 0.0) for k, i in enumerate(o)]))
        if len(out) == 8:
            break
    with open(os.path.join(ROOT, "assets", "lowkey_routes.json"), "w") as f:
        json.dump(out, f, indent=1)
    for o in out:
        print("#%d %4.1f chests/min | %2d chests (%d guaranteed, %d buried) | %3d s | walk %4d m | near %s (%d m) | nearest named POI %d m | %.0f units from centre" % (
            o["rank"], o["chests_per_min"], o["chests"], o["guaranteed"], o["buried"], o["seconds"], o["walk_m"], o["nearest_place"], o["nearest_place_m"], o["nearest_named_m"], o["from_centre_units"]))
    return out


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:3]))
