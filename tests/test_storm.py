"""Minimap fit under the closing storm. The storm tint model (seen = 0.4 * original + violet) is applied to real clear
minimaps, so the clear fit is the ground truth; shots 25-27 are real storm frames."""
import math, os, sys, unittest
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import chestlab.estimate as E
from chestlab.estimate import SilhouetteFitter
from chestlab.live import crop_minimap
from chestlab.world import World
from storm_test import storm

MPP = 0.0875
WORLD = World()
CLEAR = (15, 16, 20, 21, 24)


def minimap(k):
    return crop_minimap(cv2.imread(os.path.join(ROOT, "real", "shot%d.png" % k)))


class Destorm(unittest.TestCase):
    def test_recovers_the_clear_classification(self):
        rng = np.random.default_rng(1)
        for k in (15, 20):
            mm = minimap(k)
            clear, known_c = E.classify(mm)
            obs, known = E.classify_mixed(storm(mm, 1.0, rng))
            both = known_c & known
            self.assertGreater(both.mean(), 0.9)
            self.assertLess(float(((np.sign(clear) != np.sign(obs)) & both).mean()), 0.005)

    def test_clear_minimap_is_untouched(self):
        obs, known = E.classify(minimap(15))
        obs2, known2 = E.classify_mixed(minimap(15))
        self.assertTrue(np.array_equal(obs, obs2) and np.array_equal(known, known2))


class FitInStorm(unittest.TestCase):
    def test_partial_and_full_storm_stay_within_a_few_metres_of_the_clear_fit(self):
        rng = np.random.default_rng(4)
        clear = SilhouetteFitter(WORLD)
        errs = {c: [] for c in (0.3, 0.5, 0.7, 1.0)}
        for k in CLEAR:
            mm = minimap(k)
            truth = clear.fit(mm)
            self.assertIsNotNone(truth)
            for c in errs:
                fit = SilhouetteFitter(WORLD).fit(storm(mm, c, rng))
                self.assertIsNotNone(fit, "no fit at %d%% storm on shot %d" % (c * 100, k))
                errs[c].append(math.hypot(fit.u - truth.u, fit.v - truth.v) * MPP)
        for c, e in errs.items():
            self.assertLess(max(e), 3.0, "storm %d%%: worst error %.2f m" % (c * 100, max(e)))
            self.assertLess(float(np.median(e)), 1.0, "storm %d%%: median %.2f m" % (c * 100, np.median(e)))

    def test_real_storm_frames_lock(self):
        for k in (25, 26):
            fit = SilhouetteFitter(WORLD).fit(minimap(k))
            self.assertIsNotNone(fit, "shot %d" % k)
            self.assertTrue(fit.adaptive)
            self.assertGreater(fit.score, 0.8)
            self.assertGreater(fit.cc, 0.95)


class ClosingStorm(unittest.TestCase):
    def test_lock_and_position_hold_while_the_wall_closes_in(self):
        """0% -> 100% cover at 30 fps through the whole tracker: never loses lock, no jumps when the mode changes."""
        import storm_sweep
        for shot, angle in ((15, 2.2), (20, 4.0)):
            rows = storm_sweep.sweep(WORLD, shot, angle, frames=150)
            self.assertEqual([round(r[0], 2) for r in rows if not r[1]], [], "covers at which the lock was lost")
            errs = np.array([r[3] for r in rows])
            self.assertLess(float(np.percentile(errs, 95)), 1.2)
            self.assertLess(max(r[4] for r in rows if r[4] is not None), 0.5)
            self.assertTrue(any(r[2] for r in rows))                      # and the storm-aware classifier really did run


class EccStability(unittest.TestCase):
    def test_refinement_does_not_wander_with_tiny_input_changes(self):
        """ECC with a 5-px internal Gaussian did not converge: a 0.02 noise change moved the fix by 0.5-0.8 px (~1 m)."""
        rng = np.random.default_rng(0)
        f = SilhouetteFitter(WORLD)
        for k in (15, 20):
            obs, known = f._observe(minimap(k), (False, True))
            _, tx0, ty0 = f._coarse(obs, known, f.zoom)
            base = f._refine(obs, known, f.zoom, tx0, ty0)
            for _ in range(5):
                tx, ty, _ = f._refine(obs + rng.normal(0, 0.02, obs.shape).astype(np.float32), known, f.zoom, tx0, ty0)
                self.assertLess(math.hypot(tx - base[0], ty - base[1]), 0.1)


if __name__ == "__main__":
    unittest.main()
