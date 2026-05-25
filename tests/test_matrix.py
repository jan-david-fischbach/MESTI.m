from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mesti import mesti, mesti_build_fdfd_matrix, mesti_matrix_solver


def test_build_fdfd_matrix_periodic_tm_shape_and_pattern() -> None:
    eps = np.ones((1, 3), dtype=np.complex128)
    A, is_sym, _, _ = mesti_build_fdfd_matrix(eps, k0dx=0.5, xBC="periodic", yBC="periodic")

    dense = A.toarray()
    expected_lap = np.array([
        [2.0, -1.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -1.0, 2.0],
    ])
    expected = expected_lap - (0.5**2) * np.eye(3)

    assert A.shape == (3, 3)
    assert is_sym is True
    np.testing.assert_allclose(dense.real, expected, atol=1e-12)
    np.testing.assert_allclose(dense.imag, 0.0, atol=1e-12)


def test_build_fdfd_matrix_bloch_boundary_breaks_symmetry() -> None:
    eps = np.ones((2, 2), dtype=np.complex128)
    A, is_sym, _, _ = mesti_build_fdfd_matrix(eps, k0dx=0.3, xBC=0.7, yBC="PEC")
    assert is_sym is False
    assert not np.allclose(A.toarray(), A.toarray().T)


def test_mesti_matrix_solver_matches_direct_projection() -> None:
    A = sp.csc_matrix(np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.complex128))
    B = np.array([[1.0, 2.0], [0.0, -1.0]], dtype=np.complex128)
    C = np.array([[2.0, -1.0]], dtype=np.complex128)

    S, info = mesti_matrix_solver(A, B, C, opts={"solver": "SCIPY"})
    X_ref = spla.spsolve(A, B)
    S_ref = C @ X_ref

    np.testing.assert_allclose(S, S_ref, atol=1e-12)
    assert info["solver_used"] == "SCIPY"


def test_mesti_end_to_end_with_baseline_subtraction() -> None:
    syst = {
        "epsilon": np.ones((2, 2), dtype=np.complex128),
        "wavelength": 1.55,
        "dx": 0.1,
        "xBC": "PEC",
        "yBC": "PEC",
    }
    B = np.eye(4, dtype=np.complex128)
    C = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.complex128)

    S_no_baseline, _ = mesti(syst, B, C, D=None, opts={"solver": "SCIPY"})
    S_with_baseline, _ = mesti(syst, B, C, D=np.ones_like(S_no_baseline), opts={"solver": "SCIPY"})

    np.testing.assert_allclose(S_with_baseline, S_no_baseline - 1.0, atol=1e-12)
