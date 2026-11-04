import json, os, tempfile, unittest
import numpy as np
from chestlab import live


class LearnSpawn(unittest.TestCase):
    def setUp(self):
        self.saved = (live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS)

    def tearDown(self):
        live.USER, live.CHESTS, live.LEARNED, live.HEIGHTS = self.saved

    def test_new_spawn_learned_and_merged(self):
        d = tempfile.mkdtemp()
        live.USER = d; live.CHESTS = os.path.join(d, "c.json"); live.LEARNED = os.path.join(d, "l.json"); live.HEIGHTS = os.path.join(d, "h.json")
        p = live.Pipeline(reader=object())
        n0 = len(p.world.chests)
        far = np.array([5.0, 5.0])   # corner of the map, nowhere near a chest
        p.recent.extend([far + 0.5] * 6)
        self.assertIn("NEW", p.dig())
        self.assertEqual(len(p.world.chests), n0 + 1)
        self.assertTrue(p.dug[-1])
        p.dug[:] = False
        p.recent.extend([far + 3.0] * 15)
        self.assertIn("re-recorded", p.dig())          # same spawn, no duplicate
        self.assertEqual(len(p.world.chests), n0 + 1)
        q = live.Pipeline(reader=object())              # survives restart
        self.assertEqual(len(q.world.chests), n0 + 1)
        self.assertEqual(len(q.dug), n0 + 1)
        self.assertEqual(len(json.load(open(live.CHESTS))), n0)


if __name__ == "__main__":
    unittest.main()
