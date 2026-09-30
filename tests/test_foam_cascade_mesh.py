import re

import numpy as np
import pytest

from aero_optim.mesh.foam_cascade_mesh import DEFAULT_CASCADE_TEMPLATE, write_cascade_geo

POINT = re.compile(r"^Point\((\d+)\) = \{([^,]+),([^,]+),")


def template_points(path):
    return {int(m.group(1)): (float(m.group(2)), float(m.group(3)))
            for line in open(path) if (m := POINT.match(line))}


def test_blade_points_are_substituted_and_domain_kept(tmp_path):
    original = template_points(DEFAULT_CASCADE_TEMPLATE)
    new_blade = [(x + 1e-3, 2 * y) for x, y in (original[i] for i in range(1, 323))]
    out = tmp_path / "blade.geo"
    write_cascade_geo(DEFAULT_CASCADE_TEMPLATE, new_blade, str(out))
    written = template_points(str(out))
    assert np.allclose([written[i] for i in range(1, 323)], new_blade)
    assert all(written[i] == original[i] for i in original if i > 322)
    body = lambda p: [ln for ln in open(p) if not ln.startswith("Point(")]  # noqa: E731
    assert body(out) == body(DEFAULT_CASCADE_TEMPLATE)


@pytest.mark.parametrize("n", [321, 330])
def test_wrong_point_count_is_rejected(tmp_path, n):
    with pytest.raises(RuntimeError):
        write_cascade_geo(DEFAULT_CASCADE_TEMPLATE, [(0., 0.)] * n, str(tmp_path / "bad.geo"))
