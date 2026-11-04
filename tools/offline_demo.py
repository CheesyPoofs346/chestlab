"""Run the live pipeline on saved screenshots and draw where the overlay would put the chest markers.
python tools/offline_demo.py [real/shot*.png]  ->  out/demo_<name>.png"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.live import Pipeline, DEFAULTS


def draw(frame, st, n_chests):
    out = frame.copy()
    for m in st["markers"]:
        if m["dug"] or not m["front"]:
            continue
        x, y = int(m["x"]), int(m["y"]); tx, ty = (int(v) for v in m["top"])
        if not (-50 < x < out.shape[1] + 50 and -300 < y < out.shape[0] + 50):
            continue
        w = int(max(2, min(7, 260 / max(m["dist"], 10))))
        if m["ring"]:
            cv2.polylines(out, [np.int32(m["ring"])], True, (0, 255, 255), w, cv2.LINE_AA)
        cv2.line(out, (x, y), (tx, ty), (0, 255, 255), w, cv2.LINE_AA)
        r = int(max(5, min(16, 420 / max(m["dist"], 12))))
        cv2.drawMarker(out, (tx, ty), (0, 255, 255), cv2.MARKER_DIAMOND, 2 * r, 3)
        cv2.putText(out, "#%d %dm" % (m["id"], round(m["dist"])), (tx - 30, ty - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "heading %s  pos %s" % (st.get("compass"), None if st["pos"] is None else tuple(round(v) for v in st["pos"])),
                (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    files = sorted(sys.argv[1:] or glob.glob(os.path.join("real", "shot*.png")))
    p = Pipeline(dict(DEFAULTS))
    for f in files:
        p.tracker.filt.reset()
        frame = cv2.imread(f)
        st = p.process(frame, force_ocr=True)
        vis = [m for m in st["markers"] if m["front"] and 0 < m["x"] < frame.shape[1] and 0 < m["y"] < frame.shape[0]]
        print(os.path.basename(f), "locked", st["locked"], "island", st["on_island"], "heading", st.get("compass"),
              "pos", None if st["pos"] is None else tuple(round(v) for v in st["pos"]),
              "on-screen:", [(m["id"], round(m["dist"])) for m in vis])
        cv2.imwrite(os.path.join("out", "demo_" + os.path.basename(f)), draw(frame, st, 0))
