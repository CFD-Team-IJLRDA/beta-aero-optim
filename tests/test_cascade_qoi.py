import math
import os

import numpy as np
import pandas as pd
import pytest

from aero_optim.simulator.cascade_qoi import (
    mixedout_state, plane_qois, qoi_history, summarize, total_pressure
)

# roughly the ADP inflow state at MP1
RHO, UX, UY, P = 0.1839, 145.5, 137.1, 14424.


def uniform_plane(n: int = 11, rho=RHO, ux=UX, uy=UY, p=P) -> pd.DataFrame:
    return pd.DataFrame({"rho": np.full(n, rho), "Ux": np.full(n, ux), "Uy": np.full(n, uy), "p": np.full(n, p)})


def test_mixedout_of_uniform_flow_is_the_local_state():
    df = uniform_plane()
    p_bar, p0_bar = mixedout_state(df.rho, df.rho * df.Ux, df.rho * df.Uy, df.p)
    assert p_bar == pytest.approx(P, rel=1e-10)
    assert p0_bar == pytest.approx(total_pressure(RHO, UX**2 + UY**2, P), rel=1e-10)


def test_identical_planes_give_zero_loss_and_flow_angle():
    qoi = plane_qois(uniform_plane(), uniform_plane())
    assert qoi["MixedoutLossCoef"] == pytest.approx(0., abs=1e-12)
    assert qoi["MPLossCoef"] == pytest.approx(0., abs=1e-12)
    assert qoi["OutflowAngle"] == pytest.approx(math.degrees(math.atan(UY / UX)))
    assert qoi["InflowAngle"] == pytest.approx(math.degrees(math.atan(UY / UX)))


def test_total_pressure_drop_gives_positive_loss():
    # same static state downstream but lower velocity -> lower total pressure
    qoi = plane_qois(uniform_plane(), uniform_plane(ux=0.9 * UX, uy=0.9 * UY))
    assert 0 < qoi["MixedoutLossCoef"] < 1
    assert qoi["MPLossCoef"] == pytest.approx(qoi["MixedoutLossCoef"], rel=1e-6)


def write_sample(sets_dir: str, it: int, ux2: float):
    tdir = os.path.join(sets_dir, str(it))
    os.makedirs(tdir)
    for name, ux in (("MP1", UX), ("MP2", ux2)):
        df = pd.DataFrame({"x": 0., "y": np.linspace(0, 1, 5), "z": 0.005, "T": 273., "p": P,
                           "rho": RHO, "U_0": ux, "U_1": UY, "U_2": 0.})
        df.to_csv(os.path.join(tdir, f"{name}_T_p_rho_U.csv"), index=False)


def test_history_and_summary_from_openfoam_sets(tmp_path):
    sets_dir = str(tmp_path / "MP")
    write_sample(sets_dir, 100, UX)
    write_sample(sets_dir, 101, 0.9 * UX)
    history = qoi_history(sets_dir)
    assert list(history.index) == [100, 101]
    assert history.loc[100, "MixedoutLossCoef"] == pytest.approx(0., abs=1e-12)
    assert history.loc[101, "MixedoutLossCoef"] > 0
    summary = summarize(history)
    assert summary["n_samples"] == 2
    assert summary["MixedoutLossCoef_var"] == pytest.approx(summary["MixedoutLossCoef_std"] ** 2)
    assert summary["MixedoutLossCoef_std"] > 0
