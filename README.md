# ChestLab

Fortnite chest radar for Windows. It reads your minimap and the compass number off the screen, works out where on the whole season map you are and
which way you face, and draws a radar overlay with every nearby chest at its true bearing and distance. Optionally it also draws the chests on the
game's full map (M).

It only looks at the screen (screen capture, no game files, no memory access, no injection) and draws its own transparent window on top.

## Run it

Download `ChestLab.exe` from the latest release and run it **while Fortnite is in Borderless / Windowed Fullscreen** (exclusive fullscreen can't be
captured). A small window asks which route to follow (or none), then the radar appears. Settings and the log are saved in a `user` folder next to the exe.

From source: `python -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`, then `run.bat`.

## Settings

There are no key binds to learn. The same settings are on the **Settings** button of the start-up window and in the **ChestLab settings window**,
which starts minimised in the taskbar (it never pops up over the game): radar position (right side by default, left, any corner), radar range and
orientation, chests on the game's full map on/off, overlay refresh rate (60 / 90 / 120 / 144). The window also has the route, skip a chest, new match,
hide the overlay and quit. Changes apply at once and are saved in `user/config.json`.

The only keys it watches are the game's own: **M** and **Esc** for the full map, and **Shift** to go back to the radar when M was pressed by accident
(while only *looking* for the map; once the map is found Shift stays the game's Reset).


## Needs

* Windows 10/11, 16:9 screen (resolutions scale; ultrawide is untested), default or near-default HUD scale
* Windows' English OCR language pack (it reads the compass number with the built-in OCR)
* Fortnite in Borderless / Windowed Fullscreen

Accuracy: about 1 to 4 m depending on where you are; close to a chest the bearing can be several degrees off. Storms over the minimap can hide the map for a
moment (the radar then coasts, dimmed)

## Build the exe

`build_exe.bat` (PyInstaller, one file) -> `dist\ChestLab.exe`. `ChestLab.exe --selftest` loads everything and exits without opening a window.
Tests: `python -m unittest discover -s tests`; `python bench.py`. The tests that need real game screenshots (`tests/data/`, not included) are skipped when it is missing.

## Data

`assets/` holds map and chest data taken from fortnite.gg; those belong to their owners. Not affiliated with Epic Games.
