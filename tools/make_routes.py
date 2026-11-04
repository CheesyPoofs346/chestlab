"""Best chest routes over the whole map -> assets/routes.json + out/routes.png.   python tools/make_routes.py [sure|all]
sure = only chests that always spawn (100% spawn, rare, sprite); all = every chest, weighted by how likely it exists."""
import json, os, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import route as R
from chestlab.worldmap import KINDS, px_to_uv

mode = (sys.argv[1] if len(sys.argv) > 1 else "all")
with open(os.path.join(ROOT, "assets", "world_chests.json")) as f:
    data = json.load(f)
kinds = [k for k in KINDS if (mode == "all" or k in ("chest_sure", "rare", "sprite"))]
pts = np.array([p for k in kinds for p in data[k]], float) / 16.0         # z4 px -> map units
kind = np.array([k for k in kinds for _ in data[k]])
P = R.metres(pts)
v = np.array([R.VALUE[k] for k in kind])
R.EXTRA = np.where(kind == "buried", R.DIG_S, np.where(kind == "vault", R.VAULT_S, 0.0))
R.use_terrain()                                  # swimming costs time
routes = R.best_routes(P, v)
out = []
for r, o in routes:
    t = R.route_time(P, o)
    length = float(np.hypot(*(P[o[1:]] - P[o[:-1]]).T).sum())
    out.append(dict(chests_per_min=round(60 * r, 2), chests=len(o), expected=round(float(v[o].sum()), 1), seconds=round(t, 1), metres=round(length),
                    kinds={k: int((kind[o] == k).sum()) for k in kinds if (kind[o] == k).any()},
                    start_units=[round(float(x), 2) for x in pts[o[0]]], path_units=[[round(float(x), 2) for x in pts[i]] for i in o],
                    path_uv=[[round(float(x), 1) for x in px_to_uv(pts[i] * 16, 4)] for i in o]))
    print("%5.1f chests/min | %2d chests (%.0f expected) | %3.0f s | %4d m | start at map unit (%.1f, %.1f) | %s" % (
        60 * r, len(o), v[o].sum(), t, length, pts[o[0]][0], pts[o[0]][1], out[-1]["kinds"]))
with open(os.path.join(ROOT, "assets", "routes_%s.json" % mode), "w") as f:
    json.dump(out, f)
img = cv2.imread(os.path.join(ROOT, "assets", "gg", "full_z4.png"))
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
for k, c in (("chest", (150, 150, 150)), ("chest_sure", (60, 160, 255)), ("rare", (200, 90, 160)), ("sprite", (90, 200, 80)), ("buried", (60, 190, 250))):
    if k in data:
        for x, y in data[k]:
            cv2.circle(img, (int(x / 2), int(y / 2)), 2, c, -1, cv2.LINE_AA)
for i, rt in enumerate(out):
    poly = np.array([[x * 8, y * 8] for x, y in rt["path_units"]], np.int32)          # unit * 16 px / 2
    cv2.polylines(img, [poly], False, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.polylines(img, [poly], False, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, str(i + 1), (int(poly[0][0]) - 6, int(poly[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, str(i + 1), (int(poly[0][0]) - 6, int(poly[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
cv2.imwrite(os.path.join(ROOT, "out", "routes_%s.png" % mode), img)
