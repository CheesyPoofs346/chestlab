"""assets/gg/full_z4.png (the whole map, 16x16 fortnite.gg tiles at zoom 4) and assets/world_chests.json (every chest on it).
    python tools/build_world.py [season]      season = the map id on fortnite.gg (43 when this was written)
Chest coordinates are fortnite.gg map units (lat, lng); the file holds them as zoom-4 pixels (x = lng*16, y = -lat*16)."""
import concurrent.futures as cf
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GG = os.path.join(ROOT, "assets", "gg")
HDR = {"User-Agent": "Mozilla/5.0", "Referer": "https://fortnite.gg/"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=60) as r:
        return r.read()


def main(season="43", z=4):
    os.makedirs(GG, exist_ok=True)
    n, T = 2 ** z, 256
    full = np.zeros((n * T, n * T, 3), np.uint8)

    def tile(xy):
        x, y = xy
        img = cv2.imdecode(np.frombuffer(get("https://fortnite.gg/maps/%s/%d/%d/%d.webp" % (season, z, x, y)), np.uint8), cv2.IMREAD_COLOR)
        return x, y, img

    with cf.ThreadPoolExecutor(16) as ex:
        for x, y, img in ex.map(tile, [(x, y) for y in range(n) for x in range(n)]):
            if img is not None:
                full[y * T:y * T + img.shape[0], x * T:x * T + img.shape[1]] = img[:T, :T]
    cv2.imwrite(os.path.join(GG, "full_z%d.png" % z), full)
    js = get("https://fortnite.gg/data/spawns.js").decode("utf-8")           # (the page adds ?v=<build>; any value works)
    S = json.loads(js[js.index("{"):js.rindex("}") + 1])

    def coords(key, icon=None):
        return [c for e in S[key] if icon is None or e.get("icon") == icon for c in e["coords"]]

    kinds = dict(buried=coords("buried_chests"), chest_sure=coords("chests", 1327), rare=coords("rare_chests"),
                 sprite=coords("sprite_chests"), chest=coords("chests", 2))
    out = {k: [[round(c[1] * 16, 1), round(-c[0] * 16, 1)] for c in v] for k, v in kinds.items()}
    # the 4 "rare" chests next to every vault ARE the vault's chests: behind a door that needs a key or a bomb (about 50/50 to have one)
    vaults = np.array([[c[1] * 16, -c[0] * 16] for e in S["key_vaults"] for c in e["coords"]])
    near = lambda p: len(vaults) and np.hypot(*(vaults - p).T).min() < 12.0 / 8.75 * 16
    out["vault"] = [p for p in out["rare"] if near(p)]
    out["rare"] = [p for p in out["rare"] if not near(p)]
    drop_piles(out)
    with open(os.path.join(ROOT, "assets", "world_chests.json"), "w") as f:
        json.dump(out, f)
    print({k: len(v) for k, v in out.items()})


# fortnite.gg lists ~15 "always" chests inside 2 m of each other at the east end of the Rubber Resort bridge (the duck): not real,
# the user confirmed it. (map units x, y; radius in metres). Buried spawns are a separate list and stay.
WRONG_PILES = [((105.3, 97.1), 20.0)]


def drop_piles(out):
    for (ux, uy), r_m in WRONG_PILES:
        c = np.array([ux, uy]) * 16
        for k in out:
            if k != "buried":
                out[k] = [p for p in out[k] if np.hypot(*(np.array(p) - c)) * 8.75 / 16 > r_m]
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["43"]))
