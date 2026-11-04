"""A storm wall closing over a real minimap, 0% -> 100% at 30 fps, through the whole Tracker (gating, coasting, Kalman).
Per coverage band: share of frames locked, position error against the clear-weather fit (m), biggest frame-to-frame jump."""
import math, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.estimate import SilhouetteFitter, Tracker
from chestlab.live import crop_minimap
from chestlab.world import World

VIOLET = np.array([200, 18, 150], np.float32)
MPP = 0.0875


def storm_frame(mm, cover, angle, ramp, rng):
    s = mm.shape[0]
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    proj = (xx - s / 2) * math.cos(angle) + (yy - s / 2) * math.sin(angle)
    thr = s / 2 * (1 - 2 * cover)                       # wall position along the sweep direction
    w = np.clip((proj - thr) / ramp + 0.5, 0, 1)[..., None]
    st = 0.4 * mm.astype(np.float32) + 0.6 * VIOLET
    out = mm * (1 - w) + st * w + rng.normal(0, 2.0, mm.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def sweep(world, shot, angle, frames=240, ramp=6.0, seed=0):
    rng = np.random.default_rng(seed)
    mm = crop_minimap(cv2.imread(os.path.join("real", "shot%d.png" % shot)))
    truth = SilhouetteFitter(world).fit(mm)
    tr = Tracker(world)
    rows = []
    prev = None
    for i in range(frames):
        cover = i / (frames - 1)
        r = tr.step(storm_frame(mm, cover, angle, ramp, rng), 1 / 30)
        pos = r["pos"] if r["locked"] else None
        err = None if pos is None else math.hypot(pos[0] - truth.u, pos[1] - truth.v) * MPP
        jump = None if pos is None or prev is None else math.hypot(pos[0] - prev[0], pos[1] - prev[1]) * MPP
        prev = pos
        rows.append((cover, r["locked"], bool(r.get("fit") and r["fit"].adaptive), err, jump))
    return rows


if __name__ == "__main__":
    world = World()
    bands = [(0, .25), (.25, .5), (.5, .75), (.75, .95), (.95, 1.01)]
    allrows = []
    for shot in (15, 20, 24):
        for angle in (0.4, 2.2, 4.0):
            allrows += sweep(world, shot, angle)
    print("coverage   frames  locked  storm-mode  err median / p95 / max (m)   max frame jump (m)")
    for lo, hi in bands:
        r = [x for x in allrows if lo <= x[0] < hi]
        e = np.array([x[3] for x in r if x[3] is not None])
        j = [x[4] for x in r if x[4] is not None]
        print(f"{lo:4.0%}-{min(hi, 1):4.0%} {len(r):8d} {np.mean([x[1] for x in r]):7.1%} {np.mean([x[2] for x in r]):10.1%}   "
              f"{np.median(e):.2f} / {np.percentile(e, 95):.2f} / {e.max():.2f}                 {max(j):.2f}")
