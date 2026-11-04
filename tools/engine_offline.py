"""The live engine (compass thread + minimap thread + snapshot) running on real screenshots served by a fake capture
source. No window, no screen access. python tools/engine_offline.py [real/shot*.png]"""
import ctypes, glob, os, sys, time
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # below normal: be nice to a running game
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.engine import Engine, FrameSource
from chestlab.live import Pipeline, DEFAULTS

OCR = {13: 308, 14: 310, 15: 170, 16: 138, 17: 138, 18: 53, 19: 189, 20: 150, 21: 302, 22: 235, 23: 125, 24: 148, 26: 260, 27: 260}
if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join("real", "shot*.png")), key=lambda f: int(f[-6:-4]))
    ok = 0
    for f in files:
        img = cv2.imread(f); H, W = img.shape[:2]; k = int(f[-6:-4])
        pipe = Pipeline(dict(DEFAULTS)); eng = Engine(pipe, dict(left=0, top=0, width=W, height=H), grabber=FrameSource(img)); eng.start()
        t0 = time.perf_counter(); first = None
        while time.perf_counter() - t0 < 3.0:
            time.sleep(0.05); st = eng.snapshot(time.perf_counter(), W, H)
            if first is None and st["locked"] and st["heading"] is not None: first = time.perf_counter() - t0
        eng.stop = True; time.sleep(0.15)
        st = eng.snapshot(time.perf_counter(), W, H)
        he = None if st["heading"] is None or k not in OCR else (st["heading"] - OCR[k] + 180) % 360 - 180
        print("shot%d locked=%s island=%s heading=%s (truth %s, err %s) src=%s first fix %.2fs | compass %.0f Hz, map %.0f Hz | markers %d | err=%s" % (
            k, st["locked"], st["on_island"], None if st["heading"] is None else round(st["heading"], 1), OCR.get(k), None if he is None else round(he, 1),
            eng.heading_src, first or -1, eng.rate_c.v, eng.rate_p.v, len(st["markers"]), eng.err))
