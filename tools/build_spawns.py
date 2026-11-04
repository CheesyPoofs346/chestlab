"""assets/spawns.json: every buried-chest spawn on the island, in map pixels.

Sources: assets/gg_buried_island.json (19 spawn points, fortnite.gg, game coordinates) and assets/chests.json (the 11 chest
icons read off a screenshot of the in-game map, i.e. the spawns that existed in that match). The two agree exactly after
a similarity transform (RANSAC over point pairs), which is also how the other 8 spawns get their map position.
    python tools/build_spawns.py"""
import itertools, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "assets")


def fit(gg, det, tol=3.0):
    """Similarity (scale, rotation, translation; no mirroring) mapping gg points onto as many det points as possible."""
    best = (0, None)
    for i, j in itertools.permutations(range(len(det)), 2):
        for k, l in itertools.permutations(range(len(gg)), 2):
            z = complex(*(det[j] - det[i])) / complex(*(gg[l] - gg[k]))
            R = abs(z) * np.array([[np.cos(np.angle(z)), -np.sin(np.angle(z))], [np.sin(np.angle(z)), np.cos(np.angle(z))]])
            t = det[i] - R @ gg[k]
            d = np.linalg.norm((gg @ R.T + t)[None] - det[:, None], axis=2)
            n = int((d.min(axis=1) < tol).sum())
            if n > best[0]:
                best = (n, (R, t))
    return best


if __name__ == "__main__":
    gg = np.array(json.load(open(os.path.join(A, "gg_buried_island.json"))), float)
    det = json.load(open(os.path.join(A, "chests.json")))
    xy = np.array([[c["map_x"], c["map_y"]] for c in det], float)
    n, (R, t) = fit(gg, xy)
    assert n == len(xy), "the spawn list does not explain every chest icon on the map (%d of %d)" % (n, len(xy))
    P = gg @ R.T + t
    owner = [int(np.argmin(np.linalg.norm(P - c, axis=1))) for c in xy]                 # which spawn each map icon is
    order = owner + [i for i in range(len(P)) if i not in owner]                          # map icons first, in their old numbering
    out = [dict(id=k + 1, map_x=round(float(P[i, 0]), 1), map_y=round(float(P[i, 1]), 1), on_map=k < len(owner)) for k, i in enumerate(order)]
    with open(os.path.join(A, "spawns.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("scale %.2f px per game unit; %d spawns, %d of them were on the map screenshot" % (np.sqrt(abs(np.linalg.det(R))), len(out), len(owner)))
