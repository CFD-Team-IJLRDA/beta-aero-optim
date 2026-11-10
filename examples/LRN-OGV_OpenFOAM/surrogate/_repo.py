"""
Imported first by every script of this folder: puts the repository root on sys.path (so that aero_optim, including
aero_optim.surrogate, imports without installing the repo in the PyTorch environment) and names the folders.
SRC: this folder (the scripts). WORK: the surrogate work folder with the data, models and runs (aero_optim/surrogate/paths.py).
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SRC, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from aero_optim.surrogate.paths import RUNS, WORK  # noqa: E402,F401
