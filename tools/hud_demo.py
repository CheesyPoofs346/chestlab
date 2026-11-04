"""Run the live pipeline on saved screenshots and render the HUD over them -> out/hud_<name>.png (what the overlay shows).
python tools/hud_demo.py [real/shot*.png]"""
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.hud import Hud, hex_bgr
from chestlab.live import Pipeline, DEFAULTS


def composite(frame, canvas):
    a = canvas[..., 3:4].astype(np.float32) / 255.0
    return np.clip(canvas[..., :3].astype(np.float32) + frame.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    files = sorted(sys.argv[1:] or glob.glob(os.path.join("real", "shot*.png")))
    p = Pipeline(dict(DEFAULTS))
    for f in files:
        p.tracker.filt.reset()
        frame = cv2.imread(f)
        st = p.process(frame, force_ocr=True)
        H, W = frame.shape[:2]
        hud, cv = Hud(W, H), np.zeros((H, W, 4), np.uint8)
        t0 = time.perf_counter()
        hud.render(cv, st, "chests: heading %s | 30 fps | scale 0.085 m/px" % st.get("compass"), hex_bgr("#3dff8a"), "chest #2 dug (was 3.1 m away)", t=0.3)
        ms = (time.perf_counter() - t0) * 1000
        print(os.path.basename(f), "locked", st["locked"], "render %.1f ms" % ms)
        cv2.imwrite(os.path.join("out", "hud_" + os.path.basename(f)), composite(frame, cv))
