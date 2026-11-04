# ChestLab

Fortnite chest radar for Windows. It reads your minimap and the compass number off the screen, works out where on the whole season map you are and
which way you face, and draws a radar overlay with every nearby chest at its true bearing and distance. Optionally it also draws the chests on the
game's full map (M).

It only looks at the screen (screen capture, no game files, no memory access, no injection) and draws its own transparent window on top.

## Run it

Download `ChestLab.exe` from the latest release and run it **while Fortnite is in Borderless / Windowed Fullscreen** (exclusive fullscreen can't be
captured). A small window asks which route to follow (or none), then the radar appears. Settings and the log are saved in a `user` folder next to the exe.

From source: `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`, then `run.bat`.

| Key | Does |
| --- | --- |
| F1 | move the radar: right side (default), left, each corner |
| Alt+M | chests on the game's full map: on / off |
| M | the game's full map (chests drawn on it while it is open) |
| Shift | while it is only *looking* for the map (M pressed by accident): back to the radar |
| F12 | next route / off |
| F2 | skip a chest that is not there |
| F5 | new match |
| F6 | hide / show the overlay |
| F10 | quit |

`--advanced` adds calibration, 3D markers and spawn-learning keys (see the top of `live_app.py`).

## Needs

* Windows 10/11, 16:9 screen (resolutions scale; ultrawide is untested), default or near-default HUD scale
* Windows' English OCR language pack (it reads the compass number with the built-in OCR)
* Fortnite in Borderless / Windowed Fullscreen

Accuracy: about 1 to 4 m depending on where you are; close to a chest the bearing can be several degrees off. Storms over the minimap can hide the map for a
moment (the radar then coasts, dimmed). Only tested on one PC.

## Build the exe

`build_exe.bat` (PyInstaller, one file) -> `dist\ChestLab.exe`. `ChestLab.exe --selftest` loads everything and exits without opening a window.
Tests: `python -m unittest discover -s tests`; `python bench.py`.

## Data

`assets/` holds map and chest data taken from fortnite.gg and screenshots of the game; those belong to their owners. Not affiliated with Epic Games.
