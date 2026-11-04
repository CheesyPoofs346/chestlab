import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.calib import ChestRecorder, fit_fov_yaw, metres_per_pixel
from chestlab.estimate import ArrowHeading, PoseFilter, SilhouetteFitter, classify
from chestlab.geometry import Camera, basis, wrap180
from chestlab.render import render_minimap
from chestlab.world import World

ZOOM = 0.0712
WORLD = World()


class Geometry(unittest.TestCase):
    cam = Camera(2560, 1440, 80.0)
    zero = np.zeros(3)

    def test_wrap(self):
        self.assertAlmostEqual(float(wrap180(190)), -170)
        self.assertAlmostEqual(float(wrap180(-190)), 170)
        self.assertAlmostEqual(float(wrap180(360)), 0)

    def test_straight_ahead_is_screen_centre(self):
        xy, _, ok = self.cam.project([[0, 10, 0]], self.zero, 0, 0)
        self.assertTrue(ok[0])
        np.testing.assert_allclose(xy[0], [1280, 720], atol=1e-9)

    def test_fov_edge_lands_on_screen_edge(self):
        a = math.radians(40)
        xy, _, _ = self.cam.project([[10 * math.sin(a), 10 * math.cos(a), 0]], self.zero, 0, 0)
        self.assertAlmostEqual(xy[0, 0], 2560, places=6)

    def test_yaw_rotates_world(self):
        xy, _, _ = self.cam.project([[10, 0, 0]], self.zero, 90, 0)  # east point, facing east
        np.testing.assert_allclose(xy[0], [1280, 720], atol=1e-9)

    def test_behind_camera_rejected(self):
        self.assertFalse(self.cam.project([[0, -5, 0]], self.zero, 0, 0)[2][0])

    def test_horizon_row_matches_projection_of_far_point(self):
        for p in (-15, 0, 12):
            xy, _, _ = self.cam.project([[0, 1e7, 0]], self.zero, 0, p)
            self.assertAlmostEqual(xy[0, 1], self.cam.horizon_row(p), delta=0.01)

    def test_ray_roundtrip(self):
        eye, yaw, pitch = np.array([3.0, -2.0, 1.7]), 37.0, -8.0
        pt = np.array([30.0, 41.0, 0.4])
        (sx, sy), _, _ = (lambda r: (r[0][0], r[1], r[2]))(self.cam.project(pt, eye, yaw, pitch))
        d = self.cam.ray(sx, sy, yaw, pitch)
        want = pt - eye
        np.testing.assert_allclose(d / np.linalg.norm(d), want / np.linalg.norm(want), atol=1e-9)

    def test_basis_orthonormal(self):
        f, r, u = basis(123, 17)
        for a, b in ((f, r), (f, u), (r, u)):
            self.assertAlmostEqual(float(a @ b), 0, places=12)
        self.assertAlmostEqual(float(u[2]), math.cos(math.radians(17)), places=12)


class Estimators(unittest.TestCase):
    def test_classify_basic_colours(self):
        img = np.zeros((1, 3, 3), np.uint8)
        img[0, 0] = (60, 200, 90)       # grass
        img[0, 1] = (190, 160, 30)      # teal water
        img[0, 2] = (255, 255, 255)     # white overlay
        obs, known = classify(img)
        self.assertGreater(obs[0, 0], 0.9)
        self.assertLess(obs[0, 1], -0.9)
        self.assertLess(abs(float(obs[0, 2])), 0.05)
        self.assertEqual(list(known[0]), [True, True, False])

    def test_island_fit_recovers_position_noise_free(self):
        f = SilhouetteFitter(WORLD)
        f.zoom = ZOOM
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(12):
            uv = (rng.uniform(150, 800), rng.uniform(100, 780))
            fit = f.fit(render_minimap(WORLD, uv, rng.uniform(0, 360), ZOOM, rng, clutter=False))
            self.assertIsNotNone(fit)
            errs.append(math.hypot(fit.u - uv[0], fit.v - uv[1]))
        self.assertLess(float(np.median(errs)), 9.0, errs)   # sanity only (map px; 1 minimap px = 14); bench.py measures it properly

    def test_ecc_sign_convention(self):
        """A pure sub-pixel shift must be recovered with the right sign (guards the origin/ECC convention)."""
        f = SilhouetteFitter(WORLD)
        f.zoom = ZOOM
        rng = np.random.default_rng(2)
        base = (480.0, 420.0)
        a = f.fit(render_minimap(WORLD, base, 0, ZOOM, rng, clutter=False))
        b = f.fit(render_minimap(WORLD, (base[0] + 14, base[1] - 9), 0, ZOOM, rng, clutter=False))
        self.assertAlmostEqual(b.u - a.u, 14, delta=4)
        self.assertAlmostEqual(b.v - a.v, -9, delta=4)

    def test_arrow_heading_clean(self):
        ah, rng = ArrowHeading(), np.random.default_rng(3)
        for h in range(0, 360, 37):
            est, _ = ah.estimate(render_minimap(WORLD, (480, 420), h, ZOOM, rng, clutter=False))
            self.assertLess(abs(float(wrap180(est - h))), 3.0, (h, est))

    def test_arrow_heading_survives_route_line(self):
        ah, rng = ArrowHeading(), np.random.default_rng(4)
        errs = [abs(float(wrap180(ah.estimate(render_minimap(WORLD, (480, 420), h, ZOOM, rng, scene_seed=h))[0] - h)))
                for h in range(5, 360, 31)]
        self.assertLess(float(np.median(errs)), 4.0, errs)
        self.assertLess(float(np.percentile(errs, 90)), 15.0, errs)

    def test_kalman_beats_raw(self):
        rng = np.random.default_rng(5)
        pf, raw, filt = PoseFilter(), [], []
        for i in range(300):
            z = np.array([500 + 0.9 * i, 400.0]) + rng.normal(0, 3, 2)
            pf.predict(1 / 30)
            pf.update(z[0], z[1], 10.0, 1 / 30)
            if i > 60:
                raw.append(np.hypot(z[0] - (500 + 0.9 * i), z[1] - 400))
                filt.append(np.hypot(pf.pos[0] - (500 + 0.9 * i), pf.pos[1] - 400))
        self.assertLess(np.mean(filt), 0.7 * np.mean(raw))

    def test_angle_filter_wraps(self):
        pf = PoseFilter()
        pf.update(0, 0, 359.0, 1 / 30)
        for _ in range(30):
            pf.predict(1 / 30)
            pf.update(0, 0, 1.0, 1 / 30)
        self.assertLess(abs(float(wrap180(pf.th - 1.0))), 2.0)


class Calibration(unittest.TestCase):
    def test_fov_yaw_solver_exact(self):
        fov, off, W = 92.0, 3.0, 2560
        rel = np.linspace(-35, 35, 9)
        sx = W / 2 + (W / 2) / math.tan(math.radians(fov / 2)) * np.tan(np.radians(rel + off))
        got = fit_fov_yaw(rel, sx, W)
        self.assertAlmostEqual(got[0], fov, places=3)
        self.assertAlmostEqual(got[1], off, places=3)

    def test_chest_recorder_replaces_wrong_position(self):
        rec = ChestRecorder([[100, 100], [200, 200]])
        rec.press(1, [[300, 310], [302, 308], [298, 312]])
        np.testing.assert_allclose(rec.chests[1], [300, 310], atol=1e-9)
        np.testing.assert_allclose(rec.chests[0], [100, 100])

    def test_metres_per_pixel(self):
        self.assertAlmostEqual(metres_per_pixel(52.0, (0, 0), (200, 0)), 0.26)


if __name__ == "__main__":
    unittest.main(verbosity=2)
