from __future__ import annotations

from typing import Any

import numpy as np

from .matrix import mesti_build_fdfd_matrix, mesti_matrix_solver


def _get(syst: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in syst:
        return syst[key]
    if default is not None:
        return default
    raise KeyError(f"Missing required system field: {key}")


def mesti(
    syst: dict[str, Any],
    B: np.ndarray,
    C: np.ndarray | None = None,
    D: np.ndarray | None = None,
    opts: dict[str, Any] | None = None,
):
    """Python port of mesti() for TM polarization and no-PML systems."""
    opts = {} if opts is None else dict(opts)

    epsilon = syst.get("epsilon")
    inv_epsilon = syst.get("inv_epsilon")
    if epsilon is None and inv_epsilon is None:
        raise ValueError("Either syst['epsilon'] (TM) or syst['inv_epsilon'] (TE) must be provided.")

    if inv_epsilon is not None:
        raise NotImplementedError("TE polarization is not implemented in this initial Python port.")

    wavelength = complex(_get(syst, "wavelength"))
    dx = float(_get(syst, "dx"))
    k0dx = (2.0 * np.pi * dx) / wavelength

    xBC = syst.get("xBC", "periodic")
    yBC = syst.get("yBC", "periodic")

    A, is_symmetric_A, _, _ = mesti_build_fdfd_matrix(
        epsilon,
        k0dx,
        xBC,
        yBC,
        xPML=syst.get("xPML", None),
        yPML=syst.get("yPML", None),
        use_UPML=syst.get("PML_type", "UPML").upper() != "SC-PML",
    )

    solver_opts = dict(opts)
    solver_opts.setdefault("is_symmetric_A", is_symmetric_A)

    S, info = mesti_matrix_solver(A, B, C, solver_opts)
    if D is not None and np.size(D) != 0:
        S = S - np.asarray(D)

    info["A_shape"] = A.shape
    return S, info


def mesti2s(
    syst: dict[str, Any],
    in_: Any,
    out: Any | None = None,
    opts: dict[str, Any] | None = None,
):
    """Placeholder for mesti2s() in this initial Python port."""
    raise NotImplementedError(
        "mesti2s() is not yet implemented in the Python port; use mesti() with explicit B/C matrices."
    )
