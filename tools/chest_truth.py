"""Ground truth from your screenshots: where a real golden chest is on screen vs where the pipeline's nearest marker puts it."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.live import Pipeline, DEFAULTS
# approximate screen position (2000x1125 frame) of the visible golden chest in each shot, picked from the contact sheet
HINT = {14: (1155, 625), 15: (1075, 604), 16: (1022, 718), 17: (950, 667), 18: (700, 542), 20: (375, 571), 21: (1743, 700), 25: (1440, 730), 26: (1022, 659)}

def find_chest(img, hint, r=110):
    x0, y0 = max(hint[0] - r, 0), max(hint[1] - r, 0)
    roi = img[y0:hint[1] + r, x0:hint[0] + r]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 120, 170), (34, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = stats[i]
    return (x0 + x + w / 2, y0 + y + h / 2, x0 + x + w / 2, y0 + y + h, int(a))   # centre x,y ; base x,y ; area

if __name__ == "__main__":
    p = Pipeline(dict(DEFAULTS))
    for k, hint in HINT.items():
        f = os.path.join("real", "shot%d.png" % k)
        img = cv2.imread(f); p.tracker.filt.reset()
        st = p.process(img, force_ocr=True)
        c = find_chest(img, hint)
        if c is None or not st["locked"]:
            print(k, "no chest blob" if c is None else "no lock"); continue
        ms = [m for m in st["markers"] if m["front"]]
        best = min(ms, key=lambda m: np.hypot(m["x"] - c[0], m["y"] - c[3]))
        print("shot%d chest centre (%.0f,%.0f) base (%.0f,%.0f) area %d | nearest marker #%d at (%.0f,%.0f) %.1f m | err dx %+.0f dy %+.0f px | heading %s" % (
            k, c[0], c[1], c[2], c[3], c[4], best["id"], best["x"], best["y"], best["dist"], best["x"] - c[0], best["y"] - c[3], st.get("compass")))


def derive(p, st, c, W, H):
    """Back-project the chest's base pixel to the ground: where (map px) does the camera model say it is?"""
    from chestlab.geometry import Camera
    cfg = p.cfg
    cam = Camera(W, H, cfg["fov"]); yaw = st["heading"] + cfg["yaw_off"]
    pos = np.asarray(st["pos"], float)
    eye = cam.eye(p.world.uv_to_enu(pos, 0.0)[0], yaw, cfg["pitch"])
    ray = cam.ray(c[2], c[3] - cfg["v_off"] * H / 1125.0, yaw, cfg["pitch"])
    if ray[2] >= 0:
        return None
    s = (cfg["chest_z"] - eye[2]) / ray[2]
    enu = eye + ray * s
    return p.world.enu_to_uv(enu[None, :2])[0], float(np.hypot(*(enu[:2] - p.world.uv_to_enu(pos, 0.0)[0][:2])))


def survey():
    p = Pipeline(dict(DEFAULTS))
    for k, hint in HINT.items():
        img = cv2.imread(os.path.join("real", "shot%d.png" % k)); p.tracker.filt.reset()
        st = p.process(img, force_ocr=True); c = find_chest(img, hint)
        if c is None or not st["locked"]:
            continue
        d = derive(p, st, c, img.shape[1], img.shape[0])
        if d is None:
            continue
        uv, dist_m = d
        gaps = np.hypot(*(p.world.chests - uv).T) * p.world.mpp
        j = int(np.argmin(gaps))
        near_player = np.hypot(*(p.world.chests - np.asarray(st["pos"])).T) * p.world.mpp
        print("shot%d: chest is %.1f m ahead of you | nearest LIST chest to it: #%d, %.1f m away | nearest list chest to YOU: %.1f m" % (
            k, dist_m, j + 1, gaps[j], near_player.min()))


if __name__ == "__main__" and "--survey" in sys.argv:
    survey()
