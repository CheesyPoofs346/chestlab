"""Storm accuracy: take real clean minimaps, put a synthetic storm on them (colour model fitted to real storm frames:
0.4*original + 0.6*violet), and compare the fit against the fit of the clean frame (ground truth)."""
import glob, math, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.estimate import SilhouetteFitter, REAL_ZOOM
from chestlab.live import crop_minimap
from chestlab.world import World

VIOLET = np.array([200, 18, 150], np.float32)

def storm(mm, cover, rng):
    """cover in 0..1: fraction of the minimap under the storm, wall = straight edge at a random angle."""
    s = mm.shape[0]; yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    a = rng.uniform(0, 2 * math.pi); proj = (xx - s / 2) * math.cos(a) + (yy - s / 2) * math.sin(a)
    thr = np.quantile(proj, 1 - cover) if cover < 1 else -1e9
    m = (proj >= thr)[..., None]
    st = 0.4 * mm.astype(np.float32) + 0.6 * VIOLET
    return np.where(m, st, mm.astype(np.float32)).astype(np.uint8)

if __name__ == "__main__":
    W = World(0.21); rng = np.random.default_rng(7)
    files = [f for f in sorted(glob.glob("real/shot*.png")) if int(f.split("shot")[1][:-4]) <= 24 and int(f.split("shot")[1][:-4]) != 23]
    clean = SilhouetteFitter(W, min_score=0.8)
    rows = {c: [] for c in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)}; fails = {c: 0 for c in rows}
    for f in files:
        mm = crop_minimap(cv2.imread(f)); t = clean.fit(mm)
        if t is None: continue
        for c in rows:
            for rep in range(4 if c not in (0.0, 1.0) else 1):
                fit = SilhouetteFitter(W, min_score=0.8).fit(storm(mm, c, rng)) if c > 0 else clean.fit(mm)
                if fit is None: fails[c] += 1; continue
                rows[c].append(math.hypot(fit.u - t.u, fit.v - t.v))
    print("covered | n   fail | median / p90 / max error  (map px; x0.085 = metres)")
    for c, e in rows.items():
        if not e: print(f"{c:7.0%} | no fits, {fails[c]} fails"); continue
        e = np.array(e); print(f"{c:7.0%} | {len(e):3d} {fails[c]:4d} | {np.median(e):5.1f} / {np.percentile(e,90):5.1f} / {e.max():5.1f}   = {np.median(e)*.085:.1f} m / {np.percentile(e,90)*.085:.1f} m")
