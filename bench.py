"""chestlab benchmark: drives the whole pipeline (minimap -> position, heading -> projected chest markers) against
the simulator, where the ground truth is known, and checks error budgets. Run: python bench.py"""
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from chestlab.calib import ChestRecorder, fit_fov_yaw
from chestlab.estimate import ArrowHeading, HorizonPitch, SilhouetteFitter, Tracker
from chestlab.geometry import Camera, wrap180
from chestlab.render import draw_markers, render_minimap, render_view
from chestlab.world import World

ZOOM, MPP = 0.0698, 0.21   # ZOOM = the real game's measured minimap zoom
WORLD = World(mpp=MPP)
CAM = Camera(2560, 1440, 80.0)
RESULTS = []


def check(name, value, limit, unit="", lower=True):
    ok = bool(value <= limit if lower else value >= limit)
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} {value:9.2f} {unit:3s} (limit {'<=' if lower else '>='} {limit})")


def head(t):
    print(f"\n== {t}")


def terrain(uv):
    """Smooth hills the estimator knows nothing about (it assumes a flat island)."""
    uv = np.asarray(uv, float)
    return 1.2 * np.sin(uv[..., 0] / 90) * np.cos(uv[..., 1] / 110)


def hard(rng, uv, heading, seed=None):
    """Nasty minimap: storm wall, route line + pins, JPEG, noise, blur, and an arrow that differs from the sprite."""
    return render_minimap(WORLD, uv, heading, ZOOM, rng, storm=0.8, noise=6, jpeg=35, blur=0.9, scene_seed=seed,
                          arrow=(rng.uniform(15, 19), rng.uniform(0.25, 0.45)))


def pct(x, q):
    return float(np.percentile(x, q))


# ---------------------------------------------------------------------------------------------------------
head("S1 position + heading from a clean minimap (zoom known)")
rng = np.random.default_rng(10)
fitter, arrow = SilhouetteFitter(WORLD), ArrowHeading()
fitter.zoom = ZOOM
pe, he = [], []
for _ in range(100):
    uv, hd = rng.uniform([150, 100], [800, 780]), rng.uniform(0, 360)
    mm = render_minimap(WORLD, uv, hd, ZOOM, rng, clutter=False)
    f = fitter.fit(mm)
    pe.append(math.hypot(f.u - uv[0], f.v - uv[1]) * MPP if f else np.nan)
    he.append(abs(float(wrap180(arrow.estimate(mm)[0] - hd))))
check("clean: median position error", np.nanmedian(pe), 1.0, "m")
check("clean: p95 position error", np.nanpercentile(pe, 95), 2.5, "m")
check("clean: p95 heading error", pct(he, 95), 1.0, "deg")

head("S2 re-acquisition when the minimap zoom is off by up to 4% (snaps to the constant zoom)")
rng = np.random.default_rng(11)
zerr, perr, fails = [], [], 0
for _ in range(12):
    z, uv = rng.uniform(0.0670, 0.0726), rng.uniform([200, 150], [750, 720])
    f = SilhouetteFitter(WORLD).acquire(render_minimap(WORLD, uv, 0, z, rng, clutter=False))
    if f is None:
        fails += 1
        continue
    zerr.append(abs(f.zoom / z - 1) * 100)
    perr.append(math.hypot(f.u - uv[0], f.v - uv[1]) * MPP)
check("acquire: failures (of 12)", fails, 0)
# zoom is no longer estimated: it is a constant in the game (REAL_ZOOM); the scan only snaps to it within +-4%
check("acquire: p95 position error (zoom off by up to 4%)", pct(perr, 95), 4.0, "m")
check("acquire: median position error", float(np.median(perr)), 2.0, "m")

head("S3 hard minimap (storm, route line, pins, JPEG q35, noise, blur, different arrow)")
rng = np.random.default_rng(12)
pe, he, miss = [], [], 0
for i in range(150):
    uv, hd = rng.uniform([150, 100], [800, 780]), rng.uniform(0, 360)
    mm = hard(rng, uv, hd, seed=1000 + i)
    f = fitter.fit(mm)
    if f is None:
        miss += 1
        continue
    pe.append(math.hypot(f.u - uv[0], f.v - uv[1]) * MPP)
    he.append(abs(float(wrap180(arrow.estimate(mm)[0] - hd))))
print(f"      refused {miss}/150 frames (storm wall covers most of the minimap; refusing beats guessing)")
check("hard: refusal rate", miss / 150 * 100, 45.0, "%")
check("hard: median position error", float(np.median(pe)), 2.5, "m")
check("hard: p95 position error", pct(pe, 95), 6.0, "m")
check("hard: median heading error", float(np.median(he)), 3.0, "deg")
check("hard: p95 heading error", pct(he, 95), 10.0, "deg")

# ---------------------------------------------------------------------------------------------------------
head("S4 a 20 s walk along the beach at 30 fps (storm + noise, Kalman on)")
rng = np.random.default_rng(13)
N, FPS = 600, 30


def make_path(n, fps, rng, speed=5.5):
    c = WORLD.chests.mean(0)
    pts = WORLD.chests[np.argsort(np.arctan2(WORLD.chests[:, 1] - c[1], WORLD.chests[:, 0] - c[0]))]
    pts = np.vstack([c + (pts - c) * 0.93, c + (pts[:1] - c) * 0.93])
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    s = (rng.uniform(0, cum[-1]) + np.arange(n) * speed / MPP / fps) % cum[-1]
    uv = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)
    d = np.gradient(uv, axis=0)
    t = np.arange(n) / fps
    yaw = (np.degrees(np.arctan2(d[:, 0], -d[:, 1])) + 35 * np.sin(2 * np.pi * t / 9)) % 360
    return uv, yaw, -5 + 8 * np.sin(2 * np.pi * t / 7)


uv_t, yaw_t, pitch_t = make_path(N, FPS, rng)
tr = Tracker(WORLD)
raw_e, flt_e, yaw_raw_e, yaw_flt_e, ms, est = [], [], [], [], [], []
for i in range(N):
    mm = render_minimap(WORLD, uv_t[i], yaw_t[i], ZOOM, rng, storm=0.6, noise=4, jpeg=50, blur=0.6, scene_seed=77)
    t0 = time.perf_counter()
    r = tr.step(mm)
    ms.append((time.perf_counter() - t0) * 1000)
    est.append(r)
    if r["pos"] is not None and i >= 30:
        raw_e.append(math.hypot(r["raw"][0] - uv_t[i][0], r["raw"][1] - uv_t[i][1]) * MPP)
        flt_e.append(math.hypot(r["pos"][0] - uv_t[i][0], r["pos"][1] - uv_t[i][1]) * MPP)
        yaw_raw_e.append(abs(float(wrap180(r["raw_heading"] - yaw_t[i]))))
        yaw_flt_e.append(abs(float(wrap180(r["heading"] - yaw_t[i]))))
tracked = len(flt_e) / (N - 30)
rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
print(f"      raw RMSE {rmse(raw_e):.2f} m -> filtered {rmse(flt_e):.2f} m | heading raw {rmse(yaw_raw_e):.2f} deg -> filtered {rmse(yaw_flt_e):.2f} deg")
check("walk: frames tracked", tracked * 100, 98.0, "%", lower=False)
check("walk: filtered position RMSE", rmse(flt_e), 2.5, "m")
check("walk: filter beats raw (ratio)", rmse(flt_e) / rmse(raw_e), 1.0)
check("walk: filtered heading RMSE", rmse(yaw_flt_e), 3.0, "deg")
check("walk: median ms per frame", float(np.median(ms)), 25.0, "ms")
check("walk: p95 ms per frame", pct(ms, 95), 60.0, "ms")

# ---------------------------------------------------------------------------------------------------------
head("S5 where the chest markers land on a 2560x1440 screen (px off the true projection)")
cam_small = CAM.with_size(480, 270)
hp = HorizonPitch()
bins = [(5, 20), (20, 50), (50, 100), (100, 200)]
err = {k: {b: ([], []) for b in bins} for k in ("level", "horizon")}
demo = None
for i in range(40, N, 10):
    if est[i]["pos"] is None:
        continue
    P = WORLD.uv_to_enu(uv_t[i], terrain(uv_t[i]))[0]
    eye = CAM.eye(P, yaw_t[i], pitch_t[i])
    C = WORLD.uv_to_enu(WORLD.chests, terrain(WORLD.chests) + 0.5)
    xy, zc, ok = CAM.project(C, eye, yaw_t[i], pitch_t[i])
    view = render_view(WORLD, cam_small, cam_small.eye(P, yaw_t[i], pitch_t[i]), yaw_t[i], pitch_t[i])
    for variant, p_est in (("level", 0.0), ("horizon", hp.estimate(view, cam_small))):
        Pe = WORLD.uv_to_enu(est[i]["pos"], 0.0)[0]
        eye_e = CAM.eye(Pe, est[i]["heading"], p_est)
        xye, zce, oke = CAM.project(WORLD.uv_to_enu(WORLD.chests, 0.5), eye_e, est[i]["heading"], p_est)
        vis = ok & oke & (xy[:, 0] > -100) & (xy[:, 0] < CAM.w + 100) & (xy[:, 1] > -100) & (xy[:, 1] < CAM.h + 100)
        for j in np.where(vis)[0]:
            for b in bins:
                if b[0] <= zc[j] < b[1]:
                    err[variant][b][0].append(abs(xye[j, 0] - xy[j, 0]))
                    err[variant][b][1].append(abs(xye[j, 1] - xy[j, 1]))
        if variant == "horizon" and demo is None and vis.sum() >= 3:
            demo = (i, view, xy / (CAM.w / cam_small.w), xye / (CAM.w / cam_small.w), vis)
print("      distance    n   horizontal p50 / p90 px    vertical p50 px (level pitch -> horizon pitch)")
for b in bins:
    hx, vy = err["horizon"][b][0], err["horizon"][b][1]
    vy0 = err["level"][b][1]
    if hx:
        print(f"      {b[0]:3d}-{b[1]:3d} m  {len(hx):3d}   {np.median(hx):7.0f} / {pct(hx, 90):6.0f}           {np.median(vy0):6.0f} -> {np.median(vy):6.0f}")
mid_h = err["horizon"][(20, 50)][0] + err["horizon"][(50, 100)][0]
mid_v = err["horizon"][(20, 50)][1] + err["horizon"][(50, 100)][1]
mid_v0 = err["level"][(20, 50)][1] + err["level"][(50, 100)][1]
check("markers 20-100 m: horizontal p50 (about 8 deg)", float(np.median(mid_h)), 210.0, "px")
check("markers 20-100 m: vertical p50 with horizon pitch", float(np.median(mid_v)), 150.0, "px")
check("horizon pitch beats assuming level (ratio)", float(np.median(mid_v) / np.median(mid_v0)), 1.0)

# ---------------------------------------------------------------------------------------------------------
head("S6 chest-map calibration: map is wrong by noise 12 px + offset (9,-6) px, then 'press a key at each chest'")
rng = np.random.default_rng(14)
true = WORLD.chests.copy()
wrong = true + rng.normal(0, 12, true.shape) + [9, -6]
rec = ChestRecorder(wrong)
for i in range(len(true)):
    got = []
    for k in range(20):
        pos = true[i] + rng.normal(0, 2, 2)  # standing there, shuffling about
        f = fitter.fit(hard(rng, pos, rng.uniform(0, 360), seed=2000 + i))
        if f and not f.adaptive:   # as the app does: storm-mode fixes are too rough to correct a chest with
            got.append((f.u, f.v))
    if got:
        rec.press(i, got)
done = rec.n > 0
before = float(np.sqrt(np.mean(np.sum((wrong[done] - true[done]) ** 2, 1)))) * MPP
after = float(np.sqrt(np.mean(np.sum((rec.chests[done] - true[done]) ** 2, 1)))) * MPP
print(f"      recorded {int(done.sum())}/{len(true)} chests (rest: storm hid the minimap, press again later); RMS error {before:.2f} m -> {after:.2f} m")
check("calibration: chests recorded", int(done.sum()), len(true) * 0.5, "", lower=False)
check("calibration: error after / before", after / before, 0.4)
check("calibration: error after", after, 1.5, "m")

head("S7 recover FOV and yaw offset from hand-marked chest positions on screen (sigma 4 px)")
rng = np.random.default_rng(15)
rel = rng.uniform(-36, 36, 9)
sx = CAM.w / 2 + (CAM.w / 2) / math.tan(math.radians(88 / 2)) * np.tan(np.radians(rel + 2.5)) + rng.normal(0, 4, 9)
fov, off = fit_fov_yaw(rel, sx, CAM.w)
print(f"      true FOV 88.0 / yaw offset 2.50  ->  fitted {fov:.2f} / {off:.2f}")
check("fov error", abs(fov - 88), 1.0, "deg")
check("yaw offset error", abs(off - 2.5), 0.5, "deg")

head("S8 'am I on the island?' (single-frame decision, hard minimap)")
rng = np.random.default_rng(16)
land = (cv2.erode((tr.fitter.full > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
ys, xs = np.nonzero(land)
correct = total = refused = 0
for k in range(120):
    inside = k % 2 == 0
    if inside:
        j = rng.integers(len(xs))
        uv = np.array([xs[j], ys[j]], float)
    else:
        while True:
            uv = rng.uniform([-700, -700], [1650, 1550])
            if not tr.on_land(*uv) and (uv[0] < -60 or uv[0] > 1000 or uv[1] < -60 or uv[1] > 940):
                break
    tr.filt.reset()
    tr.fitter.zoom = ZOOM
    r = tr.step(hard(rng, uv, rng.uniform(0, 360), seed=3000 + k))
    if inside and r["raw"] is None:
        refused += 1
        continue
    correct += int(r["on_island"] == inside)
    total += 1
print(f"      {refused} inside-island frames refused (storm); decided {total}")
check("on-island decision accuracy (when it decides)", correct / total * 100, 95.0, "%", lower=False)

# ---------------------------------------------------------------------------------------------------------
if demo is not None:
    i, view, tru, est_xy, vis = demo
    left = cv2.cvtColor(WORLD.img.copy(), cv2.COLOR_BGR2RGB)[:, :, ::-1].copy()
    pts = np.round(uv_t).astype(int)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(left, tuple(a), tuple(b), (0, 200, 0), 3)
    for r_, p in zip(est[30:], range(30, N)):
        if r_["pos"] is not None:
            cv2.circle(left, tuple(np.round(r_["pos"]).astype(int)), 2, (255, 200, 0), -1)
    for p, h in ((uv_t[i], yaw_t[i]),):
        a = math.radians(h)
        cv2.arrowedLine(left, tuple(np.round(p).astype(int)), tuple(np.round(p + 90 * np.array([math.sin(a), -math.cos(a)])).astype(int)), (0, 0, 255), 4)
    left = cv2.resize(left, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    v = cv2.resize(view, None, fx=1.6, fy=1.6)
    s = 1.6
    draw_markers(v, tru[vis] * s, (0, 255, 0), r=9)
    draw_markers(v, est_xy[vis] * s, (0, 0, 255), r=6, labels=[str(k + 1) for k in np.where(vis)[0]])
    for a, b in zip(tru[vis] * s, est_xy[vis] * s):
        cv2.line(v, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(v, "green = true chest   red = estimated marker", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    h = max(left.shape[0], v.shape[0])
    canvas = np.zeros((h, left.shape[1] + v.shape[1], 3), np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:v.shape[0], left.shape[1]:] = v
    cv2.imwrite(os.path.join(ROOT, "out", "report.png"), canvas)
    print("\nreport image: out\\report.png")

print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(RESULTS) else 1)
