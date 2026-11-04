"""Where things live. BASE: the read-only files that ship with the program (assets, routes). APP: the folder the program may write to (user/: config, log).
From source both are the project folder; in the built .exe BASE is its bundle and APP is the folder the .exe sits in."""
import os
import sys

FROZEN = bool(getattr(sys, "frozen", False))
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = getattr(sys, "_MEIPASS", None) or _SRC
APP = os.path.dirname(os.path.abspath(sys.executable)) if FROZEN else _SRC
