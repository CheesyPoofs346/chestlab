"""Reading which spawns have a chest this match from the full in-game map (F11)."""
import os, sys, tempfile, unittest
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import live
from chestlab.mapscan import MapScanner, ScanResult

SC = MapScanner()


def make_frame(present, scale, icon_scale, off, size=(1920, 1080), noise=4.0, seed=0, dim=0.9):
    """The island map as the game would show it: scaled, panned, other icons than island.png's, UI clutter, noise, JPEG."""
    rng = np.random.default_rng(seed)
    hole = np.zeros(SC.img.shape[:2], np.uint8)
    for x, y in SC.spawns[SC.listed]:
        cv2.circle(hole, (int(round(x)), int(round(y))), 19, 255, -1)
    img = cv2.inpaint(SC.img, hole, 5, cv2.INPAINT_TELEA)                   # island.png without its icons
    W, H = size
    M = np.float32([[scale, 0, off[0]], [0, scale, off[1]]])
    frame = np.full((H, W, 3), (60, 40, 25), np.uint8)
    cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT, dst=frame)
    s2 = int(round(SC.tpl.shape[0] * icon_scale))
    t = cv2.resize(SC.tpl, (s2, s2), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(cv2.resize(SC.mask[..., 0], (s2, s2)), (0, 0), 0.8)[..., None]
    for i, (x, y) in enumerate(SC.spawns):
        if present[i]:
            cx, cy = M @ np.array([x, y, 1.0])
            x0, y0 = int(round(cx - s2 / 2)), int(round(cy - s2 / 2))
            frame[y0:y0 + s2, x0:x0 + s2] = (frame[y0:y0 + s2, x0:x0 + s2] * (1 - a) + t * a).astype(np.uint8)
    cv2.rectangle(frame, (30, 30), (520, 160), (30, 30, 30), -1)
    cv2.putText(frame, "MAP  Buried Treasure", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 3)
    frame = np.clip(frame.astype(np.float32) * dim + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)


class Scan(unittest.TestCase):
    def test_the_reference_map_reads_as_its_own_11_icons(self):
        r = SC.scan(SC.img.copy())
        self.assertTrue(r.ok)
        self.assertTrue(np.array_equal(r.present, SC.listed))

    def test_any_subset_any_zoom_any_icon_size(self):
        rng = np.random.default_rng(3)
        for k, (scale, icon) in enumerate([(1.0, 1.0), (1.0, 1.0 * 1.3), (1.15, 1.15), (0.9, 0.9), (1.1, 0.8)]):
            present = rng.random(len(SC.spawns)) < 0.5
            present[k] = True
            f = make_frame(present, scale, icon, ((1920 - 935 * scale) / 2, (1080 - 874 * scale) / 2), seed=k)
            r = SC.scan(f)
            self.assertTrue(r.ok, r.message)
            self.assertTrue(np.array_equal(r.present, present), (k, np.round(r.scores, 2), present.astype(int)))
            self.assertAlmostEqual(r.scale, scale, delta=0.05)

    def test_something_that_is_not_the_map_is_refused(self):
        for f in (cv2.imread(os.path.join(ROOT, "real", "shot15.png")), np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)):
            self.assertFalse(SC.scan(f).ok)

    def test_the_map_without_any_icon_is_refused(self):
        f = make_frame(np.zeros(len(SC.spawns), bool), 1.0, 1.0, ((1920 - 935) / 2, (1080 - 874) / 2))
        r = SC.scan(f)
        self.assertFalse(r.ok)
        self.assertIn("no chest icons", r.message)


class Apply(unittest.TestCase):
    def setUp(self):
        self.saved = (live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS)
        d = tempfile.mkdtemp()
        live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS = d, os.path.join(d, "c.json"), os.path.join(d, "l.json"), os.path.join(d, "h.json")
        self.p = live.Pipeline(dict(live.DEFAULTS), reader=object())

    def tearDown(self):
        live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS = self.saved

    def test_scan_sets_the_active_spawns_and_keeps_learned_ones(self):
        self.p.world.chests = np.vstack([self.p.world.chests, [5.0, 5.0]])             # a learned spawn
        self.p.active = np.append(self.p.active, True)
        self.p.dug = np.append(self.p.dug, False)
        self.p.kind = np.append(self.p.kind, "buried")
        self.p.chest_h = np.append(self.p.chest_h, 0.0)
        present = np.zeros(self.p.n_island, bool)
        present[[2, 14]] = True
        msg = self.p.apply_scan(ScanResult(True, "map scan: 2 of 19", present))
        self.assertIn("#3", msg), self.assertIn("#15", msg)
        self.assertEqual(self.p.active[:self.p.n_island].sum(), 2)
        self.assertTrue(self.p.active[-1])
        ms = self.p.project(1600, 900, tuple(self.p.world.chests[2]), 0.0)
        self.assertTrue(ms[2]["sure"] and not ms[0]["sure"])

    def test_a_failed_scan_changes_nothing(self):
        before = self.p.active.copy()
        msg = self.p.apply_scan(ScanResult(False, "map scan: can't see the island map - open the full map (M) and try again"))
        self.assertIn("open the full map", msg)
        self.assertTrue(np.array_equal(before, self.p.active))

    def test_new_match_resets_to_your_original_map(self):
        self.p.apply_scan(ScanResult(True, "x", np.ones(self.p.n_island, bool)))
        self.assertTrue(self.p.active[:self.p.n_island].all())
        self.p.reset_dug()
        self.assertTrue(np.array_equal(self.p.active, self.p.on_map))


if __name__ == "__main__":
    unittest.main()
