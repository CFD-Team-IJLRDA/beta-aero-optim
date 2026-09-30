import os

import numpy as np
import pandas as pd

from aero_optim.geom import le_radius_violation
from aero_optim.main import optim_foam
from aero_optim.optim import cascade_foam
from aero_optim.shape.bladegen_pod import BladeGenPOD

OPS = ["ADP", "OP1", "OP2"]
CHORD = 0.07


def blade(k_th=1., k_cam=0., n=160):
    """Elliptic blade of chord CHORD, thickness scaled by k_th, camber k_cam."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 0.5 * CHORD * (1 - np.cos(t))
    y = 0.004 * k_th * np.sin(t) + k_cam * 0.004 * np.sin(np.pi * x / CHORD)
    return np.column_stack((x, y))


def stadium(R, L=1., n=400):
    """Straight-sided shape of half-thickness R with semicircular ends of radius R."""
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = []
    for a in s:
        if np.cos(a) >= 0:
            pts.append([L + R * np.cos(a), R * np.sin(a)])
        else:
            pts.append([R * np.cos(a), R * np.sin(a)])
    return np.array(pts)


def test_le_radius_violation():
    R = 0.05
    shape = stadium(R)
    assert le_radius_violation(shape, 0.8 * R) < 0
    assert le_radius_violation(shape, 1.2 * R) > 0
    wedge = np.array([[0., 0.], [1., 0.05], [1., -0.05]])
    assert le_radius_violation(np.array([wedge[0] + t * (wedge[1] - wedge[0]) for t in np.linspace(0, 1, 50)]
                                        + [wedge[1] + t * (wedge[2] - wedge[1]) for t in np.linspace(0, 1, 50)[1:]]
                                        + [wedge[2] + t * (wedge[0] - wedge[2]) for t in np.linspace(0, 1, 50)[1:-1]]),
                                0.01) > 0


def fake_qois(coeffs, case_dir):
    """Writes qois.csv and qoi histories of a blade with analytic losses; c1 < -0.5 * max makes OP1 drift."""
    c1, c2 = coeffs
    rows = {}
    for k, op in enumerate(OPS):
        w = 0.04 + 0.01 * k + 30 * (c1 - 1e-3 * k)**2 + 50 * c2**2
        it = np.arange(4000, 6001)
        hist = w * np.ones(len(it))
        if op == "OP1" and c1 < -2e-3:
            hist = w * (1 + 0.05 * (it - 5000) / 1000)  # 5 % drift -> unusable
        os.makedirs(os.path.join(case_dir, op), exist_ok=True)
        pd.DataFrame({"MixedoutLossCoef": hist, "MPLossCoef": hist, "OutflowAngle": 1. + 100 * c2,
                      "InflowAngle": 43.}, index=pd.Index(it, name="iteration")).to_csv(
            os.path.join(case_dir, op, "qoi_history.csv"))
        rows[op] = {"MixedoutLossCoef": hist.mean(), "OutflowAngle": 1. + 100 * c2, "status": "ok"}
    pd.DataFrame.from_dict(rows, orient="index").to_csv(os.path.join(case_dir, "qois.csv"))


def test_optimisation_loop(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    base = blade()
    k = rng.uniform([0.6, -1.], [1.4, 1.], size=(60, 2))
    D = np.array([(blade(*kk) - base).ravel() for kk in k])
    pod_dir = tmp_path / "doe" / "pod"
    os.makedirs(pod_dir)
    pod = BladeGenPOD.fit(base, D, n_modes=2)
    pod.save(str(pod_dir / "basis.npz"))
    np.savez(pod_dir / "dataset.npz", D=D, failed=np.zeros(len(D), bool))
    np.savetxt(tmp_path / "baseline.dat", base, header="baseline\nx y")

    # a finished DoE of 10 training blades plus the baseline
    doe = tmp_path / "doe"
    coeffs = np.array([pod.project(d) for d in D[:10]])
    for i, c in enumerate(coeffs):
        fake_qois(c, doe / "OPENFOAM" / f"openfoam_g0_c{i}")
    bsl = {"blade": "baseline", "L_ADP": 0.04, "L_OP": 0.05, **{f"{op}_angle": 1. for op in OPS}}
    pd.DataFrame([bsl] + [{"blade": f"{i:04d}", "c1": c[0], "c2": c[1]} for i, c in enumerate(coeffs)]).to_csv(
        doe / "doe_results.csv", index=False)

    calls = []

    def fake_run_blades(config, sim, blades, outdir, budget):
        for name, gid, cid, profile in blades:
            calls.append(name)
            fake_qois(pod.project(profile - base), sim.get_sim_outdir(gid, cid))
        return {(gid, cid): "ok" for _, gid, cid, _ in blades}

    monkeypatch.setattr(cascade_foam, "run_blades", fake_run_blades)
    config = {
        "study": {"file": str(tmp_path / "baseline.dat"), "outdir": str(doe)},
        "simulator": {"ref_input": "unused", "operating_points": {op: {} for op in OPS}},
        "optim": {"outdir": str(tmp_path / "opt"), "pod_dir": str(pod_dir), "init_doe": str(doe),
                  "pop_size": 6, "n_gen": 2, "seed": 3, "budget": 6},
    }
    _, problem = optim_foam.run(config)

    h = pd.read_csv(tmp_path / "opt" / "history.csv", dtype={"blade": str})
    assert (h.source == "doe").sum() == 10
    assert set(h[h.source == "optim"].gen) == {1, 2}
    # only blades passing the geometric constraints were simulated
    optim_rows = h[h.source == "optim"]
    simulated = optim_rows[optim_rows.run_quality != "not simulated"]
    assert sorted(calls) == sorted(simulated.blade)
    geom_ok = (optim_rows[[c for c in cascade_foam.CONSTRAINTS[:6]]] <= 0).all(axis=1)
    assert (geom_ok == (optim_rows.run_quality != "not simulated")).all()
    # drifting OP1 runs are infeasible, whatever their loss
    drifting = h[h.c1 < -2e-3]
    assert not drifting.feasible.any()
    front = pd.read_csv(tmp_path / "opt" / "pareto.csv")
    assert len(front) > 0 and front.feasible.all()
    assert os.path.isfile(tmp_path / "opt" / "Figs" / "loss_plane_g002.png")
