"""Frame cost of the HUD renderer on a real scene, pure numpy (no window). python tools/hud_profile.py [real/shot24.png]"""
import ctypes, os, sys, time
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.hud import Hud, hex_bgr
from chestlab.live import Pipeline, DEFAULTS
f = sys.argv[1] if len(sys.argv) > 1 else "real/shot24.png"
p = Pipeline(dict(DEFAULTS)); img = cv2.imread(f); st = p.process(img, force_ocr=True)
for W, H in ((2000, 1125), (2560, 1440)):
    k = W / img.shape[1]
    s2 = dict(st); s2["markers"] = [dict(m, x=m["x"] * k, y=m["y"] * k, top=(m["top"][0] * k, m["top"][1] * k), ring=[(a * k, b * k) for a, b in m["ring"]] if m["ring"] else None) for m in st["markers"]]
    hud, cv = Hud(W, H), np.zeros((H, W, 4), np.uint8)
    ts = []
    for i in range(40):
        t0 = time.perf_counter(); hud.render(cv, s2, "heading 148 | 60 fps | scale 0.0875", hex_bgr("#3dff8a"), "", t=i * 0.016); ts.append((time.perf_counter() - t0) * 1000)
    print("%dx%d markers on screen %d: render median %.1f ms p95 %.1f ms (clear included)" % (W, H, sum(1 for m in s2["markers"] if m["front"]), np.median(ts[5:]), np.percentile(ts[5:], 95)))
