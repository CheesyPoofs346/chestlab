"""Fit camera + scale from real screenshots: the same golden chest seen from different spots must back-project to the
same ground point. Unknowns: fov, pitch, mpp (m per map px), yaw offset. python tools/calibrate_from_shots.py"""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import least_squares
from chestlab.geometry import Camera
from chestlab.live import Pipeline, DEFAULTS
from tools.chest_truth import HINT, find_chest

GROUPS = {"A": (14, 20, 25), "B": (15, 21), "C": (16, 17)}     # shots that show the same chest (checked by the survey)


def collect():
    p = Pipeline(dict(DEFAULTS)); rows = []
    for g, shots in GROUPS.items():
        for k in shots:
            img = cv2.imread(os.path.join("real", "shot%d.png" % k)); p.tracker.filt.reset()
            st = p.process(img, force_ocr=True); c = find_chest(img, HINT[k])
            if c is None or not st["locked"]:
                print("skip", k); continue
            rows.append(dict(g=g, k=k, pos=np.asarray(st["pos"], float), head=float(st["compass"]), px=c[2], py=c[3], W=img.shape[1], H=img.shape[0]))
    return rows


def hit(r, fov, pitch, mpp, yaw_off, chest_z=0.4, dist=2.8, pivot=1.5, shoulder=0.5):
    cam = Camera(r["W"], r["H"], fov, dist=dist, shoulder=shoulder, pivot_h=pivot)
    yaw = r["head"] + yaw_off
    me = np.array([r["pos"][0] * mpp, -r["pos"][1] * mpp, 0.0])
    eye = cam.eye(me, yaw, pitch)
    ray = cam.ray(r["px"], r["py"], yaw, pitch)
    s = (chest_z - eye[2]) / ray[2] if ray[2] < 0 else np.nan
    return (eye + ray * s)[:2]


def residuals(th, rows):
    fov, pitch, mpp, yaw_off = th
    pts = {g: [] for g in GROUPS}
    for r in rows:
        pts[r["g"]].append(hit(r, fov, pitch, mpp, yaw_off))
    out = []
    for g, v in pts.items():
        if len(v) > 1:
            v = np.array(v) / mpp * 0.085; out += list((v - v.mean(0)).ravel())   # spread in map px (x0.085: reported as 'metres at the default scale')
    return np.nan_to_num(np.array(out), nan=50.0)


if __name__ == "__main__":
    rows = collect()
    th0 = [80.0, -4.0, 0.085, 0.0]
    r0 = residuals(th0, rows)
    print("defaults: rms spread %.2f m over %d residuals" % (np.sqrt(np.mean(r0 ** 2)), len(r0)))
    best = None
    for fov in (70, 80, 90, 100, 112):          # several starts: FOV / scale can trade off
        for mpp in (0.06, 0.085, 0.12):
            s = least_squares(residuals, [fov, -4.0, mpp, 0.0], args=(rows,), bounds=([60, -15, 0.03, -15], [115, 10, 0.4, 15]))
            if best is None or s.cost < best.cost:
                best = s
            print("start fov %3d mpp %.3f -> fov %.1f pitch %.1f mpp %.4f yaw_off %.1f  rms %.2f m" % (fov, mpp, *s.x, np.sqrt(np.mean(s.fun ** 2))))
    print("BEST fov %.1f pitch %.1f mpp %.4f yaw_off %.1f rms %.2f m" % (*best.x, np.sqrt(np.mean(best.fun ** 2))))
