import numpy as np

from aero_optim.geom import self_intersects
from aero_optim.shape.bladegen_pod import BladeGenPOD, arclength_resample


def ellipse(n=100, a=1., b=0.1):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack((a * np.cos(t), b * np.sin(t)))


def rotate(p, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return p @ np.array([[c, s], [-s, c]])


def test_self_intersection():
    prof = ellipse()
    assert not self_intersects(prof)
    crossed = prof.copy()
    crossed[:50, 1] *= -1  # upper surface flipped below the lower one
    assert self_intersects(crossed)


def test_arclength_resample_identity_and_rotation():
    base = ellipse(120)
    assert np.allclose(arclength_resample(base, base), base)
    # a finer, rotated and reversed copy: resampling must land on it and keep base's structure
    fine = rotate(ellipse(700), 3.)[::-1]
    out = arclength_resample(fine, base)
    assert out.shape == base.shape
    assert np.allclose(out, rotate(base, 3.), atol=2e-3)
    assert not self_intersects(out)


def test_pod_reconstructs_training_blades_and_orders_modes():
    rng = np.random.default_rng(0)
    base = ellipse()
    shapes = rng.normal(size=(2, base.size))
    D = (rng.normal(size=(50, 2)) * [3e-3, 1e-3]) @ shapes
    pod = BladeGenPOD.fit(base, D, n_modes=2)
    assert pod.energy[0] >= pod.energy[1]
    assert np.allclose(pod.modes.T @ pod.modes, np.eye(2))
    for d in D[:5]:
        assert np.allclose(pod.reconstruct(pod.project(d)), base + d.reshape(-1, 2), atol=1e-12)
    assert np.all(pod.bounds[:, 0] < pod.bounds[:, 1])
