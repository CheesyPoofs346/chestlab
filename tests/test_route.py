"""Chest-farming routes: the planner, and the overlay's route mode (F12)."""
import json, os, sys, unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from chestlab import live, route as R
from chestlab.hud import Hud, hex_bgr


class Planner(unittest.TestCase):
    def setUp(self):
        self.saved = (R.OVERHEAD_S, R.EXTRA)
        R.EXTRA = None
        g = np.arange(6) * 12.0                                       # a 6x6 grid of chests 12 m apart, and a lone chest far away
        self.P = np.vstack([np.array([[x, y] for x in g for y in g]), [[900.0, 900.0]]])
        self.v = np.ones(len(self.P))

    def tearDown(self):
        R.OVERHEAD_S, R.EXTRA = self.saved

    def test_time_is_overhead_walking_and_opening(self):
        R.OVERHEAD_S = 10.0
        t = R.route_time(self.P, [0, 6])                              # two chests 12 m apart
        self.assertAlmostEqual(t, 10.0 + 12.0 / R.WALK_MPS + 2 * R.OPEN_S)
        R.EXTRA = np.zeros(len(self.P)); R.EXTRA[6] = 4.0
        self.assertAlmostEqual(R.route_time(self.P, [0, 6]), t + 4.0)

    def test_it_collects_the_cluster_and_never_the_lone_chest(self):
        R.OVERHEAD_S = 40.0
        r, o = R.best_from(self.P, self.v, 0)
        self.assertGreaterEqual(len(o), 30)
        self.assertNotIn(len(self.P) - 1, o)
        self.assertEqual(len(set(o)), len(o))

    def test_longer_overhead_means_longer_routes(self):
        R.OVERHEAD_S = 1.0
        short = len(R.best_from(self.P, self.v, 0)[1])
        R.OVERHEAD_S = 300.0
        long_ = len(R.best_from(self.P, self.v, 0)[1])
        self.assertGreaterEqual(long_, short)

    def test_unlikely_chests_are_worth_less(self):
        R.OVERHEAD_S = 5.0
        v = self.v.copy(); v[:18] = 0.05                              # half the grid almost never exists
        o = R.best_from(self.P, v, 35)[1]
        self.assertTrue(sum(i >= 18 for i in o) > sum(i < 18 for i in o))

    def test_a_time_budget_is_respected_and_more_time_never_means_less(self):
        P, v = self.P, self.v
        got = {}
        for B in (10.0, 30.0, 60.0):
            val, o, t = R.best_budget(P, v, 0, B)
            self.assertLessEqual(t, B + 1e-6)
            self.assertEqual(o[0], 0)
            self.assertEqual(len(set(o)), len(o))
            self.assertAlmostEqual(val, v[o].sum())
            got[B] = val
        self.assertLessEqual(got[10.0], got[30.0])
        self.assertLessEqual(got[30.0], got[60.0])
        self.assertGreaterEqual(got[10.0], 1.0)                       # at least the chest you start at

    def test_local_search_never_does_worse_than_the_greedy_route_and_keeps_the_rules(self):
        rng = np.random.default_rng(3)
        P = rng.uniform(0, 120, (60, 2)); v = rng.choice([1.0, 0.5], 60)
        for s in (0, 7, 23):
            g = R.best_budget(P, v, s, 45.0)
            val, o, t = R.best_budget(P, v, s, 45.0, iters=120, seed=1)
            self.assertGreaterEqual(val, g[0] - 1e-9)
            self.assertLessEqual(t, 45.0 + 1e-6)
            self.assertEqual(o[0], s)
            self.assertEqual(len(set(o)), len(o))
            self.assertAlmostEqual(val, v[o].sum())

    def test_two_players_never_share_a_chest_and_beat_one_player_alone(self):
        rng = np.random.default_rng(5)
        P = rng.uniform(0, 100, (70, 2)); v = rng.choice([1.0, 0.5], 70)
        s = 0; t = int(np.argsort(np.hypot(*(P - P[0]).T))[1])
        val, a, b = R.best_pair(P, v, s, t, 40.0, iters=40, seed=2)
        self.assertEqual((a[0], b[0]), (s, t))
        self.assertFalse(set(a) & set(b))
        self.assertAlmostEqual(val, v[a].sum() + v[b].sum())
        for r in (a, b):
            self.assertLessEqual(R.route_time(P, r) - R.OVERHEAD_S, 40.0 + 1e-6)
        self.assertGreater(val, R.best_budget(P, v, s, 40.0)[0])

    def test_zero_value_chests_are_never_collected(self):
        v = self.v.copy(); v[1:] = 0.0                                 # only the start chest counts
        val, o, t = R.best_budget(self.P, v, 0, 100.0)
        self.assertEqual(o, [0])

    def test_two_opt_never_lengthens_a_path(self):
        rng = np.random.default_rng(1)
        P = rng.uniform(0, 100, (14, 2)); o = list(range(14))
        before = R.route_time(P, o)
        self.assertLessEqual(R.route_time(P, R.two_opt(P, o)), before + 1e-9)


class Overlay(unittest.TestCase):
    def setUp(self):
        self.p = live.Pipeline(dict(live.DEFAULTS), reader=object())

    def test_built_routes_exist_and_are_sorted_by_the_planner(self):
        self.assertGreaterEqual(len(self.p.routes), 8)
        for r in self.p.routes:
            self.assertEqual(len(set(r["idx"])), len(r["idx"]))

    def test_every_kind_of_route_we_made_is_in_the_menu_and_none_repeats_a_chest(self):
        groups = {r["group"] for r in self.p.routes}
        self.assertEqual(groups, {"solo", "pair", "quiet", "loop", "saved"})
        self.assertEqual([r["name"] for r in self.p.routes[:5]], ["Solo %d s, quiet" % b for b in (60, 75, 90, 120, 180)])
        self.assertEqual(len(self.p.route_menu()), len(self.p.routes) + 1)
        for r in self.p.routes:
            self.assertEqual(len(set(r["idx"])), len(r["idx"]))
        pair = [r for r in self.p.routes if r["group"] == "pair"]
        self.assertFalse(set(pair[0]["idx"]) & set(pair[1]["idx"]))                  # the two players never share a chest

    def test_a_route_is_picked_by_number_or_name_and_the_next_chest_can_be_skipped(self):
        self.assertIn("Solo 90 s", self.p.select_route(3))
        self.assertEqual(self.p.route, 2)
        self.assertIn("2 players 60 s #1: player 2", self.p.select_route("player 2"))
        self.assertIn("no route", self.p.select_route("nonsense"))
        self.assertIn("no route", self.p.select_route(999))
        self.assertEqual(self.p.route, [r["name"] for r in self.p.routes].index("2 players 60 s #1: player 2"))      # a bad pick keeps the route
        first = self.p.routes[self.p.route]["idx"][0]
        self.assertIn("skipped", self.p.skip_route_chest())
        self.assertEqual(self.p.route_i, 1)
        self.assertTrue(self.p.dug[first])
        self.assertIn("off", self.p.select_route("none"))
        self.assertIsNone(self.p.route)
        self.assertEqual(self.p.skip_route_chest(), "no route is on")

    def test_the_start_up_picker_uses_the_command_line_and_never_waits_for_a_keyboard_in_the_tests(self):
        import live_app
        cfg = dict(live.DEFAULTS)
        live_app.pick_route(self.p, cfg, ["--route", "Solo 75"], False)
        self.assertEqual(self.p.routes[self.p.route]["name"], "Solo 75 s, quiet")
        self.p.select_route(0)
        live_app.pick_route(self.p, cfg, [], False)                                  # not asked, no flag: leaves things alone
        self.assertIsNone(self.p.route)

    def test_f12_cycles_through_the_routes_and_back_to_off(self):
        n = len(self.p.routes)
        seen = [self.p.cycle_route() for _ in range(n + 1)]
        self.assertIn("route", seen[0])
        self.assertEqual(seen[-1], "route off: every chest is marked")
        self.assertIsNone(self.p.route)

    def test_route_mode_marks_only_the_route_numbered_with_one_next(self):
        self.p.cycle_route()
        r = self.p.routes[0]
        start = tuple(self.p.world.chests[r["idx"][0]] + [200.0, 0.0])      # 17 m east of the first chest
        ms = self.p.project(1600, 900, start, 270.0)
        self.assertTrue(ms and all(m["step"] > 0 for m in ms if m["show"]))                # nothing but route chests
        self.assertEqual(sum(m["next"] for m in ms), 1)
        self.assertEqual(sorted(m["step"] for m in ms if m["show"]), [1, 2, 3])        # the next three only
        nxt = next(m for m in ms if m["next"])
        self.assertEqual(nxt["step"], 1)

    def test_walking_to_the_chests_advances_the_route(self):
        self.p.cycle_route()
        r = self.p.routes[0]
        for i in r["idx"]:
            self.p.project(1600, 900, tuple(self.p.world.chests[i]), 0.0)
        self.assertEqual(self.p.route_i, len(r["idx"]))
        self.assertIn("done", self.p.route_text())
        ms = self.p.project(1600, 900, tuple(self.p.world.chests[r["idx"][-1]]), 0.0)
        self.assertFalse(any(m["next"] for m in ms))
        self.p.reset_dug()
        self.assertEqual(self.p.route_i, 0)

    def test_far_next_chest_is_still_listed_for_the_edge_arrow(self):
        self.p.cycle_route()
        r = self.p.routes[0]
        far = tuple(self.p.world.chests[r["idx"][0]] + [4000.0, 0.0])         # 350 m away
        ms = self.p.project(1600, 900, far, 90.0)
        self.assertTrue(any(m["next"] and m["show"] for m in ms))

    def test_badges_are_drawn(self):
        def render(step, nxt):
            hud, cv = Hud(1600, 900), np.zeros((900, 1600, 4), np.uint8)
            m = dict(id=1, x=800.0, y=500.0, top=(800.0, 380.0), ring=None, dist=12.0, dz=0.0, rel=0.0, front=True, dug=False, show=True, sure=True, kind="chest_sure", step=step, next=nxt)
            hud.render(cv, dict(markers=[m]), "", hex_bgr("#3dff8a"), "", t=0.1)
            return cv
        plain, one, hot = render(0, False), render(3, False), render(3, True)
        self.assertGreater(int(np.abs(plain.astype(int) - one).sum()), 0)
        self.assertGreater(int(np.abs(one.astype(int) - hot).sum()), 0)


if __name__ == "__main__":
    unittest.main()
