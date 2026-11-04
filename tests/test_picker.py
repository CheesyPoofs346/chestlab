"""The start-up route picker: drawing, hit-testing, keys, and how live_app uses it. The Tk window itself is never opened here."""
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import live, picker


class PickerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = live.Pipeline(dict(live.DEFAULTS))
        cls.info = picker.describe(cls.pipe)

    def test_every_route_has_a_tab_a_title_and_a_thumbnail(self):
        p = picker.Picker(self.info)
        self.assertEqual(sum(len(i) for i in p.tab_ids), len(self.info))
        self.assertTrue(all(len(i) <= 8 for i in p.tab_ids))                   # a tab's cards fit the window without scrolling
        for r in self.info:
            self.assertTrue(picker.title_of(r))
            th = picker.thumb(r["pts"])
            self.assertEqual(th.shape, (picker.THUMB[1] * picker.S, picker.THUMB[0] * picker.S, 3))
            self.assertGreater(th.std(), 5)

    def test_titles(self):
        by = {r["name"]: picker.title_of(r) for r in self.info}
        self.assertEqual(by["Solo 60 s, quiet"], "60 second run")
        self.assertEqual(by["2 players 60 s #1: player 2"], "Option 1 · Player 2")
        self.assertIn("Fortune's Fort", by["Quiet #2 near Fortune'S Fort"])

    def test_render_every_tab(self):
        p = picker.Picker(self.info, last=1)
        thumbs = {i: picker.thumb(r["pts"]) for i, r in enumerate(self.info)}
        for t in range(len(picker.TABS)):
            p.set_tab(t)
            im = picker.render(p, thumbs)
            self.assertEqual(im.shape, (picker.H, picker.W, 3))
            self.assertGreater(im.std(), 10)

    def test_clicks_keys_and_result(self):
        p = picker.Picker(self.info)
        self.assertEqual((p.tab, picker.TABS[p.tab]), (0, "Saved"))                 # opens on your own route when there is one
        self.assertEqual(p.sel, p.ids[0])
        p.set_tab(1)
        self.assertEqual(p.sel, p.ids[0])
        lay = picker.layout(len(p.ids))
        x0, y0, x1, y1 = lay["cards"][2]
        p.click((x0 + x1) // 2, (y0 + y1) // 2)
        self.assertEqual(p.sel, p.ids[2])
        self.assertEqual(p.result, "unset")
        p.key("Return")
        self.assertEqual(p.result, str(p.ids[2] + 1))
        p = picker.Picker(self.info)
        tx0, ty0, tx1, ty1 = picker.layout(1)["tabs"][2]
        p.click((tx0 + tx1) // 2, (ty0 + ty1) // 2)
        self.assertEqual(p.tab, 2)
        self.assertIn(p.sel, p.ids)                                            # switching group moves the selection into it
        p.key("Right")
        self.assertEqual(p.sel, p.ids[1])
        p.key("Down")
        self.assertEqual(p.sel, p.ids[3])
        p.key("Down")
        self.assertEqual(p.sel, p.ids[3])                                      # stops at the last card
        p.key("Tab")
        self.assertEqual(p.tab, 3)
        p.key("2")
        self.assertEqual(p.tab, 1)
        p.key("Escape")
        self.assertEqual(p.result, "0")
        p = picker.Picker(self.info)
        sx0, sy0, sx1, sy1 = picker.layout(1)["none"]
        p.click(sx0 + 5, sy0 + 5)
        self.assertEqual(p.result, "0")

    def test_your_screenshot_route_is_in_the_menu(self):
        mine = [r for r in self.pipe.routes if r["group"] == "saved"]
        self.assertEqual(len(mine), 1)
        self.assertEqual((mine[0]["name"], len(mine[0]["idx"]), len(set(mine[0]["idx"]))), ("Your 60 s route", 15, 15))
        self.assertEqual(self.pipe.routes[-1]["group"], "saved")                  # last, so every older menu number is unchanged
        self.assertTrue(55 < mine[0]["seconds"] < 65)

    def test_remembers_the_last_route(self):
        p = picker.Picker(self.info, last=8)
        self.assertEqual(p.sel, 8)
        self.assertIn(8, p.ids)
        self.assertEqual(picker.Picker(self.info, last=999).last, None)

    def test_pick_route_uses_the_window_and_saves_the_choice(self):
        import live_app
        cfg = dict(live.DEFAULTS)
        with mock.patch.object(live_app, "save_config") as sv:
            live_app.pick_route(self.pipe, cfg, [], True, window=lambda: "2")
            self.assertEqual(self.pipe.routes[self.pipe.route]["name"], "Solo 75 s, quiet")
            self.assertEqual(cfg["route"], "2")
            sv.assert_called()
            live_app.pick_route(self.pipe, cfg, [], True, window=lambda: "0")
            self.assertIsNone(self.pipe.route)
            self.assertEqual(cfg["route"], "")

    def test_a_broken_window_falls_back_to_the_last_route(self):
        import live_app
        cfg = dict(live.DEFAULTS, route="3")

        def boom():
            raise RuntimeError("no display")
        with mock.patch.object(live_app, "save_config"), mock.patch.object(live_app.sys, "stdin", None):
            live_app.pick_route(self.pipe, cfg, [], True, window=boom)
        self.assertEqual(self.pipe.routes[self.pipe.route]["name"], "Solo 90 s, quiet")


if __name__ == "__main__":
    unittest.main()
