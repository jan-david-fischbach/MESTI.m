from __future__ import annotations

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

BCName = Literal["periodic", "PEC", "PMC", "PECPMC", "PMCPEC"]


def _is_bloch_bc(bc: Any) -> bool:
    return isinstance(bc, (int, float, complex, np.number))


def _normalize_bc(bc: Any) -> Any:
    if _is_bloch_bc(bc):
        return complex(bc)
    if not isinstance(bc, str):
        raise TypeError("Boundary condition must be a string or scalar Bloch phase.")
    key = bc.strip().lower()
    if key not in {"periodic", "pec", "pmc", "pecpmc", "pmcpec"}:
        raise ValueError(f"Unsupported boundary condition: {bc}")
    return key


def _second_diff_matrix(n: int, bc: Any) -> sp.csr_matrix:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=np.complex128)

    bc = _normalize_bc(bc)
    main = -2.0 * np.ones(n, dtype=np.complex128)
    off = np.ones(max(n - 1, 0), dtype=np.complex128)
    d2 = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="lil")

    if _is_bloch_bc(bc):
        phase = np.exp(1j * bc)
        if n == 1:
            d2[0, 0] = -2.0 + phase + np.conj(phase)
        else:
            d2[0, n - 1] = np.conj(phase)
            d2[n - 1, 0] = phase
    elif bc == "periodic":
        if n == 1:
            d2[0, 0] = 0.0
        else:
            d2[0, n - 1] = 1.0
            d2[n - 1, 0] = 1.0
    elif bc == "pec":
        pass
    elif bc == "pmc":
        if n == 1:
            d2[0, 0] = 0.0
        else:
            d2[0, 0] = -1.0
            d2[n - 1, n - 1] = -1.0
    elif bc == "pecpmc":
        if n == 1:
            d2[0, 0] = -1.0
        else:
            d2[n - 1, n - 1] = -1.0
    elif bc == "pmcpec":
        if n == 1:
            d2[0, 0] = -1.0
        else:
            d2[0, 0] = -1.0

    return d2.tocsr()


def _build_tm_matrix(
    epsilon: np.ndarray,
    k0dx: complex,
    xbc: Any,
    ybc: Any,
) -> sp.csr_matrix:
    if epsilon.ndim != 2:
        raise ValueError("epsilon must be a 2D array")

    ny, nx = epsilon.shape
    dxx = _second_diff_matrix(nx, xbc)
    dyy = _second_diff_matrix(ny, ybc)

    lap = sp.kron(sp.eye(nx, format="csr"), dyy, format="csr") + sp.kron(dxx, sp.eye(ny, format="csr"), format="csr")
    eps_diag = sp.diags(np.asarray(epsilon, dtype=np.complex128).reshape(-1, order="F"), format="csr")
    return (-lap - (k0dx**2) * eps_diag).tocsr()


def mesti_build_fdfd_matrix(
    eps_or_inv_eps: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray],
    k0dx: complex,
    xBC: Any,
    yBC: Any,
    xPML: Any | None = None,
    yPML: Any | None = None,
    use_UPML: bool = True,
) -> tuple[sp.csr_matrix, bool, Any, Any]:
    """Build the 2D FDFD matrix for TM polarization.

    The TE path and PML profile shaping are intentionally unsupported in this
    initial port and raise NotImplementedError.
    """
    if xPML not in (None, [], {}) or yPML not in (None, [], {}):
        raise NotImplementedError("PML is not supported in this initial Python port.")
    if not use_UPML:
        raise NotImplementedError("SC-PML is not supported in this initial Python port.")

    if isinstance(eps_or_inv_eps, (list, tuple)):
        raise NotImplementedError("TE polarization is not yet implemented in the Python port.")

    epsilon = np.asarray(eps_or_inv_eps)
    A = _build_tm_matrix(epsilon, complex(k0dx), xBC, yBC)
    is_symmetric = not (_is_bloch_bc(xBC) or _is_bloch_bc(yBC))
    return A, is_symmetric, [], []


def _solve_with_mumps(A: sp.spmatrix, B: np.ndarray) -> np.ndarray:
    # Try the pymumps API first.
    try:
        from pymumps import Context  # type: ignore

        ctx = Context()
        try:
            ctx.set_centralized_sparse(A.tocoo())
            ctx.run(job=4)
            out = np.empty_like(B, dtype=np.complex128)
            for i in range(B.shape[1]):
                rhs = np.array(B[:, i], dtype=np.complex128, copy=True)
                ctx.set_rhs(rhs)
                ctx.run(job=3)
                out[:, i] = rhs
            return out
        finally:
            ctx.destroy()
    except Exception:
        pass

    # Try the python-mumps API available on conda-forge.
    try:
        from mumps import Context  # type: ignore

        ctx = Context()
        try:
            ctx.set_matrix(A.tocsc())
            ctx.factor()
            out = np.empty_like(B, dtype=np.complex128)
            for i in range(B.shape[1]):
                rhs = np.array(B[:, i], dtype=np.complex128, copy=True)
                out[:, i] = np.asarray(ctx.solve(rhs), dtype=np.complex128)
            return out
        finally:
            destroy = getattr(ctx, "destroy", None)
            if callable(destroy):
                destroy()
    except Exception as mumps_error:
        # Fallback to scipy solver when MUMPS is unavailable/misconfigured.
        raise RuntimeError("MUMPS solve failed") from mumps_error


def mesti_matrix_solver(
    A: sp.spmatrix,
    B: np.ndarray,
    C: np.ndarray | str | None = None,
    opts: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute inv(A)@B or C@inv(A)@B with optional MUMPS backend."""
    opts = {} if opts is None else dict(opts)
    solver = str(opts.get("solver", "MUMPS")).upper()

    A_csc = A.tocsc()
    B_arr = np.asarray(B, dtype=np.complex128)

    if B_arr.ndim == 1:
        B_arr = B_arr[:, None]

    info: dict[str, Any] = {"opts": {"solver": solver}, "nnz": {"A": int(A.nnz), "B": int(np.count_nonzero(B_arr))}}

    if solver == "MUMPS":
        try:
            X = _solve_with_mumps(A_csc, B_arr)
            info["solver_used"] = "MUMPS"
        except Exception as mumps_error:
            raise RuntimeError("MUMPS solver requested, but MUMPS is unavailable or failed.") from mumps_error
    elif solver in {"MATLAB", "SCIPY"}:
        lu = spla.splu(A_csc)
        X = lu.solve(B_arr)
        info["solver_used"] = "SCIPY"
    else:
        raise ValueError(f"Unsupported solver option: {solver}")

    if C is None or (isinstance(C, (list, tuple, np.ndarray)) and np.size(C) == 0):
        return np.asarray(X), info

    if isinstance(C, str):
        if C != "transpose(B)":
            raise ValueError("C string must be 'transpose(B)'.")
        C_mat = B_arr.T
    else:
        C_mat = np.asarray(C, dtype=np.complex128)

    S = C_mat @ np.asarray(X)
    info["nnz"]["S"] = int(np.count_nonzero(S))
    return S, info
