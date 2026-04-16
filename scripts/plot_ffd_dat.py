"""Plot all .dat profiles in a FFD folder.

Each .dat file is expected to have two header lines and two columns: x and y.
This script looks for files in:

  ~/GPROptimization/beta-aero-optim/examples/test-Cascade/beta_opt_feasible/output_mf/XXX/FFD

where XXX is a user-provided string (subdirectory name). You can also pass a full
path to the FFD folder.

Usage examples:

  python scripts/plot_ffd_dat.py --xxx mycase
  python scripts/plot_ffd_dat.py --ffd-dir /home/user/.../output_mf/mycase/FFD --save plot.png

Optional arguments: --save to save the figure, --dpi to change resolution, --no-show to skip interactive show.
"""

from __future__ import annotations
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def load_dat_files(folder: str) -> Dict[str, np.ndarray]:
    """Load all .dat files in folder.

    Assumes each file has two header lines and two columns (x, y).
    Returns a dict mapping filename -> ndarray shape (N,2).
    """
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "*.dat")))
    if not files:
        raise FileNotFoundError(f"No .dat files found in {folder}")

    data = {}
    for f in files:
        try:
            arr = np.loadtxt(f, skiprows=2)
        except Exception as e:
            # try a more permissive reading (strip empty lines)
            with open(f, 'r', encoding='utf-8') as fh:
                lines = [L for L in fh.readlines()[2:] if L.strip()]
            try:
                arr = np.loadtxt(lines)
            except Exception:
                raise RuntimeError(f"Could not read {f}: {e}")

        if arr.ndim == 1 and arr.size == 2:
            arr = arr.reshape((1, 2))
        if arr.shape[1] < 2:
            raise RuntimeError(f"File {f} does not contain at least two columns")

        data[os.path.basename(f)] = arr[:, :2]
    return data


def plot_profiles(data: Dict[str, np.ndarray], title: str | None = None, save: str | None = None, dpi: int = 150, show: bool = True) -> None:
    """Plot multiple x-y profiles on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, arr in data.items():
        x = arr[:, 0]
        y = arr[:, 1]
        ax.plot(x, y, label=name)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    ax.grid(True)
    # only show legend if not too many files
    if len(data) <= 20:
        ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot to {save}")
    if show:
        plt.show()
    plt.close(fig)


def default_ffd_folder(xxx: str) -> str:
    base = os.path.expanduser('~/GPROptimization/beta-aero-optim/examples/test-Cascade/beta_opt_feasible/output_mf')
    return os.path.join(base, xxx, 'FFD')


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot all .dat FFD profiles in a folder')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--xxx', type=str, help='subfolder name XXX under output_mf (will use default base path)')
    group.add_argument('--ffd-dir', type=str, help='full path to the FFD folder')
    parser.add_argument('--save', type=str, default=None, help='optional output filename to save the figure (png, pdf, ... )')
    parser.add_argument('--dpi', type=int, default=150, help='dpi for saved figure')
    parser.add_argument('--no-show', action='store_true', help='do not call plt.show() (useful in headless environments)')

    args = parser.parse_args()

    if args.ffd_dir:
        folder = args.ffd_dir
    else:
        folder = default_ffd_folder(args.xxx)

    data = load_dat_files(folder)
    title = f"FFD profiles in {os.path.basename(os.path.abspath(folder))}"
    plot_profiles(data, title=title, save=args.save, dpi=args.dpi, show=not args.no_show)


if __name__ == '__main__':
    main()
