"""The full-map overlay (M): registering a frame of the open map onto fortnite.gg's map, following pan/zoom, drawing chests on it.
Offline only: frames are cut from the gg map itself, nothing touches the screen."""
import os
import sys
import time
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import fullmap, hud, live

W, H = 2560, 1440
REF = cv2.imread(os.path.join(ROOT, "assets", "gg", "full_z4.png"))


def cam(s, cx, cy, w=W, h=H):
    """Camera showing ref point (cx, cy) in the middle at s screen px per ref px: ref -> frame (2x3)."""
    return np.array([[s, 0, w / 2 - s * cx], [0, s, h / 2 - s * cy]])


def shoot(T, w=W, h=H):
    """A frame of the gg map seen through camera T (screen outside the map is dark), with some junk drawn over it like our own icons."""
    fr = cv2.warpAffine(REF, T, (w, h), flags=cv2.INTER_AREA if T[0, 0] < 1 else cv2.INTER_LINEAR, borderValue=(28, 28, 28))
    rng = np.random.default_rng(5)
    for _ in range(120):
        cv2.circle(fr, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, 12)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1, cv2.LINE_AA)
    return fr


def err_px(T, est, w=W, h=H):
    """Largest disagreement (screen px) between two ref->frame transforms over the screen."""
    pts = np.float32([[0.2 * w, 0.2 * h], [0.8 * w, 0.2 * h], [0.5 * w, 0.5 * h], [0.2 * w, 0.8 * h]])
    inv = cv2.invertAffineTransform(T)
    ref = pts @ inv[:, :2].T + inv[:, 2]
    return float(np.hypot(*((ref @ est[:, :2].T + est[:, 2]) - pts).T).max())


if not os.path.isdir(os.path.join(ROOT, "tests", "data")):
    raise unittest.SkipTest("tests/data (real screenshots) is not part of the repository")
DATA = os.path.join(ROOT, "tests", "data")
REAL_MAP = cv2.imread(os.path.join(DATA, "map_open.webp"))          # the user's own screenshot of the open in-game map (2000 x 1125)
STORM_MAP = cv2.imread(os.path.join(DATA, "map_storm.webp"))         # the map during a storm (everything purple)
GAMEPLAY = cv2.imread(os.path.join(DATA, "gameplay.webp"))          # ... and one of the game while playing


def warp(img, z, ox, oy):
    """The same screen after the player zoomed by z and panned: new = z * old + (ox, oy)."""
    M = np.array([[z, 0, ox], [0, z, oy]])
    return M, cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA if z < 1 else cv2.INTER_CUBIC, borderValue=(60, 30, 10))


class Registration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.finder = fullmap.MapFinder()
        cls.real = cls.finder.register(REAL_MAP)

    def test_finds_the_real_in_game_map(self):
        self.assertIsNotNone(self.real)
        T, score, method = self.real
        self.assertEqual(method, "hue")
        self.assertGreater(score, 0.8)
        self.assertTrue(0.25 < fullmap.scale_of(T) < 0.36)              # the island fills ~730 px of a 1125 px screen: 0.3 screen px per z4 px

    def test_real_map_stays_consistent_under_pan_and_zoom(self):
        T0 = self.real[0]
        for z, ox, oy in ((1.4, 100, 0), (1.2, -100, 50), (0.8, -40, 30), (0.6, 200, 100)):
            M, fr = warp(REAL_MAP, z, ox, oy)
            r = self.finder.register(fr)
            self.assertIsNotNone(r, z)
            self.assertLess(err_px(fullmap.compose(M, T0), r[0], 2000, 1125), 5.0, z)

    def test_finds_the_storm_tinted_map(self):
        """A storm tints the whole map purple: the sea is not blue any more, only the shape of the land is left."""
        r = self.finder.register(STORM_MAP)
        self.assertIsNotNone(r)
        self.assertGreater(r[1], 0.7)
        self.assertTrue(0.22 < fullmap.scale_of(r[0]) < 0.36)
        M, fr = warp(STORM_MAP, 1.3, -80, -40)                           # and it stays consistent when moved
        r2 = self.finder.register(fr)
        self.assertIsNotNone(r2)
        self.assertLess(err_px(fullmap.compose(M, r[0]), r2[0], 2000, 1125), 14.0)       # colour-free fits are looser than the colour rule

    def test_finds_a_storm_map_with_our_overlay_and_the_safe_zone_on_it(self):
        """The user's own capture: purple sea, pink land, a clear-coloured safe circle, our chest icons all over it. Was 0.50 (not found), now ~0.9."""
        f = cv2.imread(os.path.join(DATA, "map_storm_ovl.webp"))
        r = self.finder.register(f)
        self.assertIsNotNone(r)
        self.assertGreater(r[1], 0.8)
        self.assertAlmostEqual(fullmap.scale_of(r[0]), 0.295, delta=0.015)

    def test_gameplay_is_not_a_map(self):
        self.assertIsNone(self.finder.register(GAMEPLAY))
        rng = np.random.default_rng(1)
        noise = cv2.GaussianBlur(rng.integers(0, 255, (1125, 2000, 3), dtype=np.uint8), (0, 0), 6)
        self.assertIsNone(self.finder.register(noise))

    def test_gg_frames_at_many_zooms(self):
        for s, cx, cy in ((0.32, 2048, 2048), (0.5, 1500, 2400), (1.0, 2600, 1500)):
            T = cam(s, cx, cy)
            r = self.finder.register(shoot(T))
            self.assertIsNotNone(r, s)
            self.assertLess(err_px(T, r[0]), 4.0, s)

    def test_track_follows_pan_and_zoom(self):
        a = cv2.cvtColor(cv2.resize(REAL_MAP, (1000, 562)), cv2.COLOR_BGR2GRAY)
        M, fr = warp(REAL_MAP, 1.1, -30, 20)
        b = cv2.cvtColor(cv2.resize(fr, (1000, 562)), cv2.COLOR_BGR2GRAY)
        D = fullmap.track(a, b)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D[0, 0], 1.1, delta=0.01)
        self.assertAlmostEqual(D[0, 2], -15.0, delta=1.0)                # half-size frame: the offset halves too


class Watch(unittest.TestCase):
    def test_finds_follows_then_gives_up_when_the_map_closes(self):
        script = dict(M=np.array([[1.0, 0, 0], [0, 1.0, 0]]), game=False)
        cache = {}

        def make_grab():
            def grab():
                time.sleep(0.03)
                if script["game"]:
                    return cv2.cvtColor(GAMEPLAY, cv2.COLOR_BGR2BGRA)
                key = script["M"].tobytes()
                if key not in cache:
                    cache.clear()
                    cache[key] = cv2.cvtColor(warp(REAL_MAP, script["M"][0, 0], script["M"][0, 2], script["M"][1, 2])[1], cv2.COLOR_BGR2BGRA)
                return cache[key]
            return grab

        w = fullmap.MapWatch(make_grab, (2000, 1125))
        w.GIVE_UP_S = 1.5
        w.start()
        try:
            t0 = time.time()
            while w.state() is None and time.time() - t0 < 20:
                time.sleep(0.05)
            st = w.state()
            self.assertIsNotNone(st, "never found the map")
            T0 = st["M"].copy()
            script["M"] = np.array([[1.25, 0, -120.0], [0, 1.25, -60.0]])      # the player zooms in and pans
            time.sleep(2.0)
            st = w.state()
            self.assertIsNotNone(st)
            self.assertLess(err_px(fullmap.compose(script["M"], T0), st["M"], 2000, 1125), 6.0)
            script["game"] = True                                             # map closed: gameplay
            t0 = time.time()
            while w.active and time.time() - t0 < 12:
                time.sleep(0.1)
            self.assertFalse(w.active)
            self.assertIsNone(w.state())
        finally:
            w.stop()


class ZoomFollow(unittest.TestCase):
    """The player zooms smoothly while OUR OWN overlay (drawn from the watcher's last answer, in its own thread) is part of what the watcher sees:
    the chests must stay on the map the whole way, not trail behind it or get dragged by their own icons."""

    def test_smooth_zoom_with_our_icons_in_the_capture(self):
        import threading
        pipe = live.Pipeline(dict(live.DEFAULTS))
        items = pipe.map_items()
        H0, W0 = REAL_MAP.shape[:2]
        hd = hud.Hud(W0, H0)
        state = dict(t0=None, snap=np.zeros((H0, W0, 4), np.uint8), canvas=np.zeros((H0, W0, 4), np.uint8))
        DUR, Z1, PIV = 1.2, 2.0, (1300.0, 400.0)
        ref = {}

        def Z_at(t):
            if state["t0"] is None:
                return np.array([[1.0, 0, 0], [0, 1.0, 0]])
            x = float(np.clip((t - state["t0"]) / DUR, 0, 1))
            z = 1 + (Z1 - 1) * x * x * (3 - 2 * x)
            return np.array([[z, 0, PIV[0] * (1 - z)], [0, z, PIV[1] * (1 - z)]])

        def make_grab():
            def grab():
                Z = Z_at(time.time())
                fr = cv2.warpAffine(REAL_MAP, Z, (W0, H0), flags=cv2.INTER_LINEAR if Z[0, 0] >= 1 else cv2.INTER_AREA, borderValue=(60, 30, 10))
                snap = state["snap"]
                sel = snap[..., 3] > 0
                fr[sel] = snap[..., :3][sel]
                state["canvas"] = snap
                time.sleep(0.015)
                return cv2.cvtColor(fr, cv2.COLOR_BGR2BGRA)
            return grab

        stop = threading.Event()

        def overlay():
            cv = np.zeros((H0, W0, 4), np.uint8)
            while not stop.is_set():
                t = time.time()
                st = ref["w"].state()
                cv[:] = 0
                if st is not None:
                    hd.render(cv, dict(markers=[], fullmap=dict(M=st["M"], items=items)), "", hud.hex_bgr("#3dff8a"), "", t=0.1)
                state["snap"] = cv.copy()
                time.sleep(max(0.0, 1 / 60 - (time.time() - t)))

        w = fullmap.MapWatch(make_grab, (W0, H0), mask=lambda: np.where(state["canvas"][::2, ::2, 3] == 0, 255, 0).astype(np.uint8))
        ref["w"] = w
        w.start()
        threading.Thread(target=overlay, daemon=True).start()
        try:
            t0 = time.time()
            while w.state() is None and time.time() - t0 < 20:
                time.sleep(0.02)
            st = w.state()
            self.assertIsNotNone(st, "never found the map")
            T0 = st["M"].copy()
            time.sleep(0.6)
            state["t0"] = time.time()
            errs = []
            while time.time() < state["t0"] + DUR + 1.2:
                now = time.time()
                s = w.state()
                if s is not None:
                    errs.append((now - state["t0"], err_px(fullmap.compose(Z_at(now), T0), s["M"], W0, H0)))
                time.sleep(0.02)
        finally:
            w.stop()
            stop.set()
        errs = np.array(errs)
        moving = errs[errs[:, 0] < DUR][:, 1]
        still = errs[errs[:, 0] > DUR + 0.7][:, 1]
        self.assertLess(float(moving.mean()), 25.0)           # was ~75 px (max 147) before the flow ignored its own icons and the answer was moved on to the display time
        self.assertLess(float(moving.max()), 80.0)
        self.assertLess(float(still.max()), 6.0)


class Drawing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = live.Pipeline(dict(live.DEFAULTS))

    def render(self, T, route=None):
        pipe = self.pipe
        pipe.select_route(route)
        # chests are in ref px (z4); the overlay maps ref -> screen with T
        st = dict(markers=[], fullmap=dict(M=T, items=pipe.map_items()))
        h, cv = hud.Hud(W, H), np.zeros((H, W, 4), np.uint8)
        h.render(cv, st, "", hud.hex_bgr("#3dff8a"), "", t=0.1)
        return cv, st["fullmap"]["items"]

    def test_icons_land_on_their_chests(self):
        T = cam(2.0, 2400, 1900)
        cv, it = self.render(T)
        xy = it["xy"] @ T[:, :2].T + T[:, 2]
        on = np.nonzero((xy[:, 0] > 60) & (xy[:, 0] < W - 60) & (xy[:, 1] > 60) & (xy[:, 1] < H - 60))[0]
        self.assertGreater(len(on), 5)
        hit = sum(cv[int(xy[i, 1]), int(xy[i, 0]), 3] > 0 or cv[int(xy[i, 1]) - 4:int(xy[i, 1]) + 5, int(xy[i, 0]) - 4:int(xy[i, 0]) + 5, 3].any() for i in on)
        self.assertEqual(hit, len(on))

    def test_zoomed_out_is_thinned(self):
        T = cam(0.32, 2048, 2048)
        cv, it = self.render(T)
        self.assertGreater(int((cv[..., 3] > 0).sum()), 1000)
        t0 = time.perf_counter()
        for _ in range(5):
            self.render(T)
        self.assertLess((time.perf_counter() - t0) / 5, 0.25)

    def test_dug_chest_is_not_drawn(self):
        T = cam(3.0, 2400, 1900)
        cv, it = self.render(T)
        xy = it["xy"] @ T[:, :2].T + T[:, 2]
        i = int(np.argmin(np.hypot(*(xy - [W / 2, H / 2]).T)))
        self.assertTrue(cv[int(xy[i, 1]) - 3:int(xy[i, 1]) + 4, int(xy[i, 0]) - 3:int(xy[i, 0]) + 4, 3].any())
        self.pipe.dug[i] = True
        try:
            cv2_, _ = self.render(T)
        finally:
            self.pipe.dug[i] = False
        self.assertFalse(cv2_[int(xy[i, 1]) - 3:int(xy[i, 1]) + 4, int(xy[i, 0]) - 3:int(xy[i, 0]) + 4, 3].any())

    def test_route_mode_numbers_and_line(self):
        self.assertTrue(self.pipe.routes)
        cv_off, _ = self.render(cam(0.4, 2048, 2048))
        cv_on, it = self.render(cam(0.4, 2048, 2048), "Solo 60")
        self.assertGreater(int(it["step"].max()), 5)
        self.assertEqual(int(it["next"]), self.pipe.routes[self.pipe.route]["idx"][0])
        self.assertGreater(int((cv_on[..., 3] > 0).sum()), 0.9 * int((cv_off[..., 3] > 0).sum()))      # every chest is still there
        self.pipe.select_route(None)


if __name__ == "__main__":
    unittest.main()


class Confirm(unittest.TestCase):
    """A map found from scratch with a weak fit or at a big zoom must be found twice, in the same place, before it is believed."""

    def watch(self):
        return fullmap.MapWatch(lambda: None, (2000, 1125))

    def test_strong_ordinary_find_is_taken_at_once(self):
        self.assertTrue(self.watch()._confirmed((fullmap.matrix(0.3, 10, 20), 0.9, "hue"), 1.0, 1125))

    def test_weak_or_zoomed_in_needs_a_second_look_that_agrees(self):
        for T, sc in ((fullmap.matrix(0.3, 10, 20), 0.75), (fullmap.matrix(1.9, 10, 20), 0.95)):
            w = self.watch()
            self.assertFalse(w._confirmed((T, sc, "hue"), 1.0, 1125))
            self.assertTrue(w._confirmed((fullmap.matrix(T[0, 0] * 1.01, T[0, 2] + 3, T[1, 2] - 2), sc, "hue"), 1.4, 1125))

    def test_a_second_look_somewhere_else_or_too_late_starts_over(self):
        w = self.watch()
        T = fullmap.matrix(1.9, 10, 20)
        self.assertFalse(w._confirmed((T, 0.9, "hue"), 1.0, 1125))
        self.assertFalse(w._confirmed((fullmap.matrix(1.9, 400, 20), 0.9, "hue"), 1.3, 1125))      # another place
        self.assertFalse(w._confirmed((fullmap.matrix(1.9, 400, 20), 0.9, "hue"), 4.0, 1125))      # too long after


class StaticPanels(unittest.TestCase):
    """The game's own panels (stats, legend, buttons) stand still while the map zooms. They are all corners, so the optical flow used to follow
    them and report 'no motion': the map was lost on every zoom. The flow now only looks between the panels."""

    @classmethod
    def setUpClass(cls):
        ui = cv2.imread(os.path.join(DATA, "map_zoomed_storm.webp"))
        finder = fullmap.MapFinder()
        cls.ui, cls.T = ui, finder.register(ui)[0]
        ref = cv2.imread(fullmap.REF)
        cls.map = cv2.warpAffine(ref, cls.T, (ui.shape[1], ui.shape[0]), borderValue=(60, 30, 10))
        cls.panels = np.zeros(ui.shape[:2], bool)
        for x0, y0, x1, y1 in ((0, 0, 2000, 75), (58, 188, 478, 712), (1598, 186, 1956, 590), (930, 1045, 1115, 1105), (1800, 1045, 1960, 1105)):
            cls.panels[y0:y1, x0:x1] = True

    def frame(self, z):
        Z = np.array([[z, 0, 1300 * (1 - z)], [0, z, 400 * (1 - z)]])
        f = cv2.warpAffine(self.map, Z, (2000, 1125), flags=cv2.INTER_LINEAR, borderValue=(60, 30, 10))
        f[self.panels] = self.ui[self.panels]
        return f

    def motion(self, region):
        w = fullmap.MapWatch(lambda: None, (2000, 1125))
        w.FLOW_REGION = region
        a, _, ma = w._flow_frame(self.frame(1.0))
        b, _, mb = w._flow_frame(self.frame(1.07))
        return fullmap.track(a, b, ma, mb)

    def test_flow_follows_the_zoom_with_panels_on_screen(self):
        D = self.motion(fullmap.MapWatch.FLOW_REGION)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(fullmap.scale_of(D), 1.07, delta=0.005)

    def test_without_the_region_the_panels_win(self):
        D = self.motion(None)                                                # (documents why the region exists)
        self.assertTrue(D is None or abs(fullmap.scale_of(D) - 1.07) > 0.03)
