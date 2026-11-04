"""The settings window's logic, on a Tk root that is never shown (withdrawn before anything is mapped)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tkinter as tk

from chestlab import settings, hud, live


class Panel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as e:
            raise unittest.SkipTest("no Tk: %s" % e)
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def make(self, **cfg):
        self.sets, self.did = [], []
        c = dict(live.DEFAULTS)
        c.update(cfg)
        self.cfg = c
        p = settings.SettingsPanel(self.root, c, ["Solo 60 s", "Pair 90 s"], 2,
                                   lambda k, v: (self.sets.append((k, v)), "saved")[1], lambda a: (self.did.append(a), "did " + a)[1])
        return p

    def test_shows_the_current_values(self):
        p = self.make(radar_at="top-left", fps=120, fullmap=False, radar_up=False, radar_m=40.0)
        self.assertEqual(p.vars["radar_at"].get(), "Top left")
        self.assertEqual(p.vars["fps"].get(), "120")
        self.assertFalse(p.vars["fullmap"].get())
        self.assertEqual(p.vars["radar_up"].get(), "North is up")
        self.assertEqual(p.vars["radar_m"].get(), "40 m")
        self.assertEqual(p.route_var.get(), "Pair 90 s")

    def test_choices_report_their_values(self):
        p = self.make()
        p.vars["radar_at"].set("Bottom right")
        p.on_choice("radar_at", settings.RADAR_AT)
        p.vars["fps"].set("144")
        p.on_choice("fps", settings.FPS)
        p.vars["fullmap"].set(False)
        p.on_check("fullmap")
        self.assertEqual(self.sets, [("radar_at", "bottom-right"), ("fps", 144), ("fullmap", False)])
        self.assertEqual(p.status.get(), "saved")

    def test_every_radar_spot_in_the_window_exists_in_the_hud(self):
        for _, v in settings.RADAR_AT:
            self.assertIn(v, hud.RADAR_SPOTS)

    def test_every_setting_key_is_a_config_key(self):
        for k in ("radar_at", "radar_m", "radar_up", "fps", "fullmap", "markers_3d"):
            self.assertIn(k, live.DEFAULTS, k)

    def test_route_and_actions(self):
        p = self.make()
        p.route_var.set("No route: mark every chest")
        p.on_route()
        p.route_var.set("Solo 60 s")
        p.on_route()
        p.on_action("skip")
        p.on_action("quit")
        self.assertEqual(self.did, ["route:0", "route:1", "skip", "quit"])


@unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "tests", "data")), "tests/data (real screenshots) is not part of the repository")
class Wiring(unittest.TestCase):
    """The callbacks live_app gives the window: they change the running app's config and call the right actions (config file write mocked)."""

    def test_set_value_and_do_on_a_headless_run(self):
        import cv2
        from unittest import mock
        import live_app
        from chestlab.engine import FrameSource
        from chestlab.overlay import NullOverlay
        img = cv2.imread(os.path.join(ROOT, "tests", "data", "ground53.webp"))
        H, W = img.shape[:2]
        cfg = dict(live.DEFAULTS)
        with mock.patch.object(live_app, "save_config") as sv:
            out = live_app.main(overlay=NullOverlay, grabber=FrameSource(img), hotkeys=False, rect=dict(left=0, top=0, width=W, height=H), seconds=1.0, cfg=cfg)
            self.assertEqual(out["set_value"]("radar_at", "left"), "Saved.")
            self.assertEqual(cfg["radar_at"], "left")
            self.assertTrue(sv.called)
            out["set_value"]("fps", 120)
            self.assertEqual(cfg["fps"], 120)
            self.assertIn("route", out["do"]("route:1").lower())
            self.assertEqual(cfg["route"], "1")
            self.assertIn("route off", out["do"]("route:0"))
            self.assertEqual(cfg["route"], "")
            out["do"]("new_match")
            self.assertEqual(out["do"]("quit"), "Quitting...")
