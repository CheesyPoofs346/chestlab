"""On-screen text fits the screen, and the overlay only tells Windows about the pixels that changed (offline: renders into numpy canvases)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import hud as H_

HINT = "M map  ·  F12 route  ·  F2 skip chest  ·  F5 new match  ·  F6 hide  ·  F10 quit"
LONG = "route 2 players 60 s #1: player 2 14/14 done: leave and rejoin | rough fix (storm or off the island) | floor +1 | heading 123 | 60 fps | map 30 Hz | scale 0.0875 | pitch -4.0"


def render(W, H, status="", toast="", **st):
    hud, cv = H_.Hud(W, H), np.zeros((H, W, 4), np.uint8)
    hud.render(cv, dict(markers=[], **st), status, H_.hex_bgr("#3dff8a"), toast, t=0.1)
    return hud, cv


def cols(cv, y0=0, y1=None):
    a = np.nonzero(cv[y0:y1, :, 3].any(axis=0))[0]
    return (int(a.min()), int(a.max())) if len(a) else (0, -1)


class Fit(unittest.TestCase):
    def test_toast_wraps_inside_the_screen_at_any_size(self):
        long = " ".join([HINT] * 3)
        for W, H in ((1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)):
            _, cv = render(W, H, toast=long)
            lo, hi = cols(cv)
            self.assertGreater(hi, lo)
            self.assertGreaterEqual(lo, 0.12 * W, (W, H))
            self.assertLessEqual(hi, 0.88 * W, (W, H))
            rows = np.nonzero(cv[..., 3].any(axis=1))[0]
            self.assertLess(rows.max(), H)                                    # not cut off at the bottom

    def test_the_default_hint_is_one_line_that_fits_a_small_screen(self):
        _, cv = render(1280, 720, toast=HINT)
        lo, hi = cols(cv)
        self.assertGreaterEqual(lo, 0.1 * 1280)
        self.assertLessEqual(hi, 0.9 * 1280)
        rows = np.nonzero(cv[..., 3].any(axis=1))[0]
        self.assertLess(rows.max() - rows.min(), 0.06 * 720)                  # a single line

    def test_a_long_status_shrinks_instead_of_running_off(self):
        for W, H in ((1280, 720), (2560, 1440)):
            _, cv = render(W, H, status=LONG)
            lo, hi = cols(cv, int(0.9 * H))
            self.assertGreaterEqual(lo, 0)
            self.assertLessEqual(hi, 0.5 * W, (W, H))


class Changed(unittest.TestCase):
    def marker(self, brg, hd):
        return dict(id=1, x=-999.0, y=-999.0, top=(-999.0, -999.0), ring=None, dist=hd, dz=0.0, rel=0.0, front=False, dug=False, sure=True,
                    kind="chest_sure", show=True, step=0, next=False, brg=brg, hd=hd)

    def test_changed_box_covers_every_pixel_that_differs(self):
        W, H = 2560, 1440
        hud, cv = H_.Hud(W, H), np.zeros((H, W, 4), np.uint8)
        frames = [dict(markers=[self.marker(40, 30)], yaw=0.0, fov=78.0, radar_m=60.0, radar_up=True),
                  dict(markers=[self.marker(50, 33)], yaw=12.0, fov=78.0, radar_m=60.0, radar_up=True),
                  dict(markers=[], yaw=None)]
        prev = cv.copy()
        for k, st in enumerate(frames * 2):
            hud.render(cv, st, "hello status" if k % 2 else "", H_.hex_bgr("#3dff8a"), "toast" if k == 3 else "", t=0.1 * k)
            diff = np.nonzero((cv != prev).any(axis=2))
            x0, y0, x1, y1 = hud.changed
            if len(diff[0]):
                self.assertGreaterEqual(int(diff[1].min()), x0, k)
                self.assertLessEqual(int(diff[1].max()), x1 - 1, k)
                self.assertGreaterEqual(int(diff[0].min()), y0, k)
                self.assertLessEqual(int(diff[0].max()), y1 - 1, k)
            prev = cv.copy()

    def test_radar_only_frame_is_a_small_part_of_the_screen_and_an_idle_frame_is_empty(self):
        W, H = 2560, 1440
        hud, cv = H_.Hud(W, H), np.zeros((H, W, 4), np.uint8)
        st = dict(markers=[self.marker(40, 30)], yaw=0.0, fov=78.0, radar_m=60.0, radar_up=True, radar_at="left")      # (the status dot sits bottom-left: the radar next to it keeps the changed box small)
        hud.render(cv, st, "", H_.hex_bgr("#3dff8a"), "", t=0.1)
        hud.render(cv, st, "", H_.hex_bgr("#3dff8a"), "", t=0.2)
        x0, y0, x1, y1 = hud.changed
        self.assertLess((x1 - x0) * (y1 - y0), 0.10 * W * H)
        hud.render(cv, dict(markers=[], yaw=None), "", H_.hex_bgr("#3dff8a"), "", t=0.3)     # radar leaves: the old pixels are wiped...
        hud.render(cv, dict(markers=[], yaw=None), "", H_.hex_bgr("#3dff8a"), "", t=0.4)
        x0, y0, x1, y1 = hud.changed                                                         # ...and then there is nothing to send
        self.assertLessEqual(x1 - x0, 0)


if __name__ == "__main__":
    unittest.main()


class StaleRadar(unittest.TestCase):
    def test_stale_radar_is_dimmed_not_gone(self):
        W, H = 1600, 900
        mk = Changed().marker(40, 30)
        st = dict(markers=[mk], yaw=0.0, fov=78.0, radar_m=60.0, radar_up=True)
        sums = []
        for extra in ({}, {"stale": True}):
            hud, cv = H_.Hud(W, H), np.zeros((H, W, 4), np.uint8)
            hud.render(cv, dict(st, **extra), "", H_.hex_bgr("#3dff8a"), "", t=0.1)
            sums.append(int(cv[..., 3].sum()))
        self.assertGreater(sums[1], 0.3 * sums[0])
        self.assertLess(sums[1], 0.7 * sums[0])


class PushesWholeBuffer(unittest.TestCase):
    """The window must really change on screen: UpdateLayeredWindowIndirect with a dirty rect 'succeeded' on the user's PC but the window kept
    showing its first frame (start-up hint and 'no minimap lock' stayed for minutes while the radar ran). Only UpdateLayeredWindow is used."""

    def test_every_changed_frame_is_pushed_with_updatelayeredwindow(self):
        from unittest import mock
        from chestlab import overlay
        ov = overlay.Overlay.__new__(overlay.Overlay)
        ov.w, ov.h, ov.left, ov.top = 800, 450, 0, 0
        ov.canvas = np.zeros((450, 800, 4), np.uint8)
        ov.hud = H_.Hud(800, 450)
        ov.full, ov.indirect, ov.hwnd, ov.dc = False, True, 1, 2
        ov.blend = overlay.BLENDFUNCTION(0, 0, 255, 1)
        fake = mock.MagicMock()
        fake.UpdateLayeredWindow.return_value = 1
        with mock.patch.object(overlay, "u32", fake):
            ov.draw(dict(markers=[]), "no minimap lock", "", "#ffffff")
            ov.draw(dict(markers=[]), "", "", "#ffffff")                          # the status disappears: that has to reach the screen too
            n = fake.UpdateLayeredWindow.call_count
        self.assertGreaterEqual(n, 2)
        fake.UpdateLayeredWindowIndirect.assert_not_called()
