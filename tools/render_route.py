"""Draw one of assets/lowkey_routes.json on the real map (zoom-6 tiles) and write the step list.   python tools/render_route.py <rank> [routes file: lowkey_routes.json | fast_routes.json]
-> out/route_<rank>.png and out/route_<rank>.txt"""
import json, math, os, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from build_world import get          # noqa  (tools/ is on the path when run as a script)

MPU, Z, T = 8.75, 6, 256
COL = {"vault": (200, 140, 60), "chest_sure": (60, 190, 255), "rare": (220, 90, 190), "sprite": (110, 220, 90), "chest": (235, 235, 235), "buried": (60, 160, 230)}
NAME = {"vault": "VAULT chest (needs a key or bomb: skip it without one)", "chest_sure": "100% chest", "rare": "rare chest", "sprite": "sprite chest", "chest": "chest (may not spawn)", "buried": "buried chest"}
DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def tiles(x0, y0, x1, y1):
    """Stitched zoom-6 pixels of the map-unit box (x0..x1, y0..y1) -> (image, origin unit)."""
    per = T / 2 ** Z * 1.0                        # 4 map units per tile
    tx0, ty0, tx1, ty1 = int(x0 // per), int(y0 // per), int(x1 // per), int(y1 // per)
    img = np.zeros(((ty1 - ty0 + 1) * T, (tx1 - tx0 + 1) * T, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = cv2.imdecode(np.frombuffer(get("https://fortnite.gg/maps/43/%d/%d/%d.webp" % (Z, tx, ty)), np.uint8), cv2.IMREAD_COLOR)
            if t is not None:
                img[(ty - ty0) * T:(ty - ty0) * T + T, (tx - tx0) * T:(tx - tx0) * T + T] = t[:T, :T]
    return img, (tx0 * per, ty0 * per)


def steps(path, label=""):
    lines = []
    for k, p in enumerate(path):
        if k == 0:
            lines.append("%s1. START: %s at map position (%.1f, %.1f)" % (label, NAME[p["kind"]], *p["unit"]))
        else:
            a, b = path[k - 1]["unit"], p["unit"]
            brg = (math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) + 360) % 360
            lines.append("%s%d. %s %.0f m to the %s: %s" % (label, k + 1, "walk" if p["leg_m"] > 3 else "step", p["leg_m"], DIRS[int((brg + 22.5) // 45) % 8], NAME[p["kind"]]))
    return lines


def main(rank, name="lowkey_routes.json"):
    r = json.load(open(os.path.join(ROOT, "assets", name)))[rank - 1]
    pair = "players" in r
    tag = ("pair%d_%d" % (r["budget_s"], rank)) if pair else ("fast%d_" % r["budget_s"] + r["scope"]) if "budget_s" in r else str(rank)
    paths = [p["path"] for p in r["players"]] if pair else [r["path"]]
    LINE = [(255, 255, 255), (255, 60, 255)]                                    # player 1 white, player 2 magenta
    pts = np.array([q["unit"] for path in paths for q in path])
    m = 4.0
    x0, y0, x1, y1 = pts[:, 0].min() - m, pts[:, 1].min() - m, pts[:, 0].max() + m, pts[:, 1].max() + m
    img, (ox, oy) = tiles(x0, y0, x1, y1)
    cx0, cy0, cx1, cy1 = (int(v * 2 ** Z) for v in (x0 - ox, y0 - oy, x1 - ox, y1 - oy))
    img = img[max(cy0, 0):cy1, max(cx0, 0):cx1].copy()
    ox, oy = ox + max(cx0, 0) / 2 ** Z, oy + max(cy0, 0) / 2 ** Z
    sc = 1600 / max(img.shape[:2]) if max(img.shape[:2]) < 1600 else 1.0
    img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for n, path in enumerate(paths):
        P = [(int(round((q["unit"][0] - ox) * 2 ** Z * sc)), int(round((q["unit"][1] - oy) * 2 ** Z * sc))) for q in path]
        cv2.polylines(img, [np.array(P, np.int32)], False, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.polylines(img, [np.array(P, np.int32)], False, LINE[n], 3, cv2.LINE_AA)
    for n, path in enumerate(paths):
        P = [(int(round((q["unit"][0] - ox) * 2 ** Z * sc)), int(round((q["unit"][1] - oy) * 2 ** Z * sc))) for q in path]
        for k, (p, q) in enumerate(zip(P, path)):
            cv2.circle(img, p, 17, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p, 14, COL[q["kind"]], -1, cv2.LINE_AA)
            if q["kind"] != "chest" or pair:
                cv2.circle(img, p, 17, LINE[n], 2, cv2.LINE_AA)
            t = str(k + 1)
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(img, t, (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        for text, at, dx in (("START" if not pair else "P%d START" % (n + 1), P[0], -34), ("END" if not pair else "P%d END" % (n + 1), P[-1], -20)):
            y = at[1] - 26 - (34 if pair and n == 1 and text.endswith("START") else 0)
            cv2.putText(img, text, (at[0] + dx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, text, (at[0] + dx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE[n] if pair else (255, 255, 255), 2, cv2.LINE_AA)
    bar = int(round(50 * sc * 2 ** Z / MPU))                                                    # 50 m scale bar
    cv2.line(img, (20, img.shape[0] - 24), (20 + bar, img.shape[0] - 24), (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(img, "50 m   north is up", (30 + bar, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(ROOT, "out", "route_%s.png" % tag), img)
    if pair:
        lines = ["Two-player %d-second route near %s (%d m away, quiet): %d chests between you, %d of them always spawn (the rest are about 50/50), no vaults. Start together (the two start chests are %d m apart), then split: player 1 (white line) and player 2 (magenta line) never take the same chest. With ~40 s to leave and rejoin that is %.1f chests a minute for the two of you." % (
            r["budget_s"], r["nearest_place"], r["nearest_place_m"], r["chests"], r["guaranteed"], r["start_gap_m"], r["chests_per_min_with_reset"])]
        for n, pl in enumerate(r["players"]):
            lines += ["", "PLAYER %d: %d chests, %d always spawn, %.0f s, %d m of walking" % (n + 1, pl["chests"], pl["guaranteed"], pl["seconds"], pl["walk_m"])] + steps(pl["path"], "")
    elif "budget_s" in r:
        lines = ["%d-second route near %s (%d m away, %s): %d chests, %d of them always spawn (the rest are about 50/50), %.0f s of walking and opening, %d m of walking, no vaults. With ~40 s to leave and rejoin that is %.1f chests a minute." % (
            r["budget_s"], r["nearest_place"], r["nearest_place_m"], r["scope"], r["chests"], r["guaranteed"], r["seconds"], r["walk_m"], r["chests_per_min_with_reset"]), ""] + steps(r["path"])
    else:
        lines = ["Route %d near %s (%d m away): %d chests, %d of them 100%% spawn, about %d s a run incl. %d s to leave/rejoin/land, %d m of walking, %.1f chests a minute." % (
            rank, r["nearest_place"], r["nearest_place_m"], r["chests"], r["guaranteed"], r["seconds"], 40, r["walk_m"], r["chests_per_min"]), ""] + steps(r["path"])
    open(os.path.join(ROOT, "out", "route_%s.txt" % tag), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]), *(sys.argv[2:3]))
