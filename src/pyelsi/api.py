from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ._errors import BackendUnavailableError, InputValidationError
from ._runtime import detect_runtime
from ._typing import BackendOpts, UseGpu
from ._core import build_info
from ._core import _elsi_dm_real_coo, _elsi_dm_real_csc, _elsi_dm_real_dense
from ._core import _elsi_ev_real_dense, _elsi_ev_real_coo
from ._core import _elsi_dm_complex_dense, _elsi_ev_complex_dense


def _is_csr(x: Any) -> bool:
    try:
        import scipy.sparse  # type: ignore

        return scipy.sparse.isspmatrix_csr(x)
    except Exception:
        return False


def _as_fortran_f64(a: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        raise InputValidationError(f"{name} must be a numpy.ndarray")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise InputValidationError(f"{name} must be a square 2D array; got shape={a.shape}")
    if a.dtype != np.float64:
        a = a.astype(np.float64, copy=False)
    if not np.isfortran(a):
        a = np.asfortranarray(a)
    return a


def _as_fortran_c128(a: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        raise InputValidationError(f"{name} must be a numpy.ndarray")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise InputValidationError(f"{name} must be a square 2D array; got shape={a.shape}")
    if a.dtype != np.complex128:
        a = a.astype(np.complex128, copy=False)
    if not np.isfortran(a):
        a = np.asfortranarray(a)
    return a


def eigh(
    H,
    S=None,
    *,
    k: int | None = None,
    solver: str = "auto",
    backend_opts: BackendOpts = None,
    n_threads: int | None = None,
    use_gpu: UseGpu = "auto",
    mpi_comm: Any = None,
    tol: float | None = None,
    max_iter: int | None = None,
    return_eigenvectors: bool = True,
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Solve a symmetric/Hermitian eigenproblem.

    Parameters
    ----------
    H:
        Hamiltonian matrix.  Dense ``numpy.ndarray`` for ELPA/ChASE/OMM;
        ``scipy.sparse.csr_matrix`` for SIPS (SLEPc-SIP).
    S:
        Overlap matrix (same type as H) for generalized problems.
    k:
        Number of eigenpairs (not yet supported; full/partial spectrum via
        ``backend_opts['n_state']``).
    solver:
        ELSI solver name — ``"elpa"`` (default), ``"chase"``, ``"sips"``,
        ``"omm"``, ``"ntpoly"``, ``"magma"``, ``"dlaf"``, etc.
    backend_opts:
        Solver-specific keyword settings forwarded to ELSI.  Recognized keys
        include ``n_electron``, ``n_state``, ``chase_tol``,
        ``chase_filter_deg``, ``chase_extra_space``, ``sips_n_slice``,
        ``sips_ev_min``, ``sips_ev_max``, ``force_single_proc``, etc.
    n_threads:
        Thread count hint (respects OMP_NUM_THREADS by default).
    use_gpu:
        ``"auto"``/``True``/``False``  (GPU requires CUDA-enabled build).
    mpi_comm:
        Optional MPI communicator (reserved; serial BLACS used internally).
    tol, max_iter:
        Reserved for iterative solvers.
    return_eigenvectors:
        If ``False``, return only eigenvalues.

    Returns
    -------
    w : ndarray, shape (n_state,)
        Eigenvalues in ascending order.
    v : ndarray, shape (n_basis, n_state)  [only when return_eigenvectors=True]
        Eigenvectors as columns.
    """

    if k is not None:
        raise NotImplementedError("Partial spectrum (k!=None) is not implemented in v0.")

    if solver == "auto":
        solver = "elpa"

    if use_gpu not in ("auto", True, False):
        raise InputValidationError("use_gpu must be 'auto', True, or False")

    rt = detect_runtime(n_threads=n_threads)
    if use_gpu is True and not rt.has_cuda:
        raise BackendUnavailableError("This pyELSI build does not include CUDA support.")

    solver_map = {
        "elpa": 1,
        "omm": 2,
        "pexsi": 3,
        "eigenexa": 4,
        "sips": 5,
        "ntpoly": 6,
        "magma": 7,
        "chase": 9,
        "dlaf": 10,
    }

    if solver not in solver_map:
        raise BackendUnavailableError(
            f"solver={solver!r} is not recognized. Supported: {sorted(solver_map.keys())} (or 'auto')."
        )

    if backend_opts is None:
        backend_opts = {}
    opts = dict(backend_opts)

    n_basis = int(H.shape[0])
    n_electron = float(opts.get("n_electron", n_basis // 2))
    n_state = int(opts.get("n_state", n_basis // 2))

    # ------------------------------------------------------------------ SIPS
    # SLEPc-SIP operates on sparse matrices in COO format.
    if solver == "sips":
        if not build_info().get("has_sips", False):
            raise BackendUnavailableError(
                "solver='sips' is not available: SLEPc-SIPs was not enabled "
                "in this ELSI build (PYELSI_ENABLE_SIPS=OFF).  "
                "Rebuild with -DPYELSI_ENABLE_SIPS=ON after installing SLEPc/PETSc."
            )
        if not _is_csr(H):
            raise InputValidationError("solver='sips' requires H as scipy.sparse.csr_matrix")
        if S is not None and not _is_csr(S):
            raise InputValidationError("solver='sips' requires S as scipy.sparse.csr_matrix (or omit S)")

        import scipy.sparse  # type: ignore

        if H.shape[0] != H.shape[1]:
            raise InputValidationError("H must be square")

        # Estimate eigenvalue interval via Gershgorin circle theorem.
        if "sips_ev_min" not in opts or "sips_ev_max" not in opts:
            diag = np.asarray(H.diagonal(), dtype=np.float64)
            abs_row_sum = np.asarray(np.abs(H).sum(axis=1), dtype=np.float64).ravel()
            off_diag_sum = abs_row_sum - np.abs(diag)
            lambda_min = float((diag - off_diag_sum).min())
            lambda_max = float((diag + off_diag_sum).max())
            margin = max(0.1 * (lambda_max - lambda_min), 0.5)
            opts.setdefault("sips_ev_min", lambda_min - margin)
            opts.setdefault("sips_ev_max", lambda_max + margin)

        opts.setdefault("sips_n_elpa", 3)
        opts.setdefault("sips_n_slice", max(2 * n_state, 20))

        H_coo = H.tocoo()
        H_coo.sum_duplicates()
        ham_val  = np.asarray(H_coo.data, dtype=np.float64)
        row_ind_1 = np.asarray(H_coo.row,  dtype=np.int32) + 1
        col_ind_1 = np.asarray(H_coo.col,  dtype=np.int32) + 1

        ovlp_val = None
        if S is not None:
            S_coo = S.tocoo()
            S_coo.sum_duplicates()
            ovlp_val = np.asarray(S_coo.data, dtype=np.float64)

        w, v_full = _elsi_ev_real_coo(
            ham_val, row_ind_1, col_ind_1,
            n_basis, ovlp_val,
            solver_map["sips"],
            n_electron, n_state,
            opts,
            bool(return_eigenvectors),
        )

        if return_eigenvectors:
            # v_full is (n_basis, n_basis) Fortran-order; only first n_state cols valid
            return w, np.asfortranarray(v_full[:, :n_state])
        return w

    # ----------------------------------------------------------------- ChASE
    # Apply default Chebyshev-filter parameters when not overridden.
    if solver == "chase":
        if not build_info().get("has_chase", False):
            raise BackendUnavailableError(
                "solver='chase' is not available: ChASE was not enabled "
                "in this ELSI build (PYELSI_ENABLE_CHASE=OFF).  "
                "Rebuild with -DPYELSI_ENABLE_CHASE=ON."
            )
        opts.setdefault("chase_tol", 1e-10)
        opts.setdefault("chase_filter_deg", 25)
        opts.setdefault("chase_extra_space", 0.25)  # fraction of n_state (0–0.5)

    # ----------------------------------------------------------------- Dense
    is_complex = np.iscomplexobj(H) or (S is not None and np.iscomplexobj(S))
    if is_complex:
        Hc = _as_fortran_c128(H, "H")
        Sc = _as_fortran_c128(S, "S") if S is not None else None
    else:
        Hc = _as_fortran_f64(H, "H")
        Sc = _as_fortran_f64(S, "S") if S is not None else None

    if is_complex:
        w, v = _elsi_ev_complex_dense(
            Hc, Sc, solver_map[solver],
            float(n_electron), int(n_state),
            opts,
            bool(return_eigenvectors),
        )
    else:
        w, v = _elsi_ev_real_dense(
            Hc, Sc, solver_map[solver],
            float(n_electron), int(n_state),
            opts,
            bool(return_eigenvectors),
        )
    return (w, v) if return_eigenvectors else w


def density_matrix(
    H,
    S=None,
    *,
    n_electrons: int | None = None,
    temperature: float = 0.0,
    mu: float | None = None,
    solver: str = "auto",
    backend_opts: BackendOpts = None,
    n_threads: int | None = None,
    use_gpu: UseGpu = "auto",
    mpi_comm: Any = None,
    tol: float | None = None,
    return_energy: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Compute a (zero-temperature) density matrix using an ELSI backend.

    Two computation routes are available:

    **Native ELSI DM solvers** — ELSI internally computes the density matrix
    without explicitly constructing eigenvectors:

    =========  ===================  ============================
    solver     Input format         Notes
    =========  ===================  ============================
    ``elpa``   Dense real/complex   Default; two-stage diag.
    ``omm``    Dense real/complex   Orbital minimization method
    ``pexsi``  Sparse CSR (real)    Pole expansion; linear scale
    ``ntpoly`` Sparse CSR (real)    Polynomial expansion
    =========  ===================  ============================

    **Eigenvector-based DM solvers** — pyELSI calls :func:`eigh` then builds
    ``D = V_occ @ V_occ†`` from the *n_electrons* lowest eigenvectors.
    The result is always a dense array:

    =========  ===================  ============================
    solver     Input format         Notes
    =========  ===================  ============================
    ``chase``  Dense real           ChASE Chebyshev eigensolver
    ``sips``   Sparse CSR (real)    SLEPc-SIP (requires SLEPc)
    ``magma``  Dense real/complex   GPU-accelerated via MAGMA
    ``dlaf``   Dense real/complex   DLA-Future
    =========  ===================  ============================
    """

    if temperature != 0.0:
        raise NotImplementedError("Finite temperature is not implemented in v0.")
    if mu is not None:
        raise NotImplementedError("Chemical potential input is not implemented in v0.")
    if n_electrons is None:
        raise InputValidationError("n_electrons is required in v0.")

    if solver == "auto":
        solver = "elpa"

    # Solvers handled by the native ELSI DM API (c_elsi_dm_real / c_elsi_dm_real_sparse).
    _native_dm_map = {
        "elpa": 1,
        "omm": 2,
        "pexsi": 3,
        "eigenexa": 4,
        "ntpoly": 6,
    }

    # Solvers implemented by computing eigenpairs (eigh) then D = V_occ @ V_occ†.
    _eigh_dm_sparse = {"sips"}          # require sparse CSR input
    _eigh_dm_dense  = {"chase", "magma", "dlaf"}  # require dense input
    _eigh_dm_solvers = _eigh_dm_sparse | _eigh_dm_dense

    _all_solvers = set(_native_dm_map) | _eigh_dm_solvers
    if solver not in _all_solvers:
        raise BackendUnavailableError(
            f"solver={solver!r} is not recognized for density matrix. "
            f"Supported: {sorted(_all_solvers)} (or 'auto')."
        )

    # Retain solver_map alias for the native path used below.
    solver_map = _native_dm_map

    opts = dict(backend_opts or {})
    opts["n_electron"] = float(n_electrons)

    # ------------------------------------------------------------------
    # Eigenvector-based DM path: call eigh() then form D = V_occ @ V_occ†
    # ------------------------------------------------------------------
    if solver in _eigh_dm_solvers:
        if solver in _eigh_dm_sparse and not _is_csr(H):
            raise InputValidationError(
                f"solver={solver!r} requires H as scipy.sparse.csr_matrix"
            )
        if solver in _eigh_dm_dense and _is_csr(H):
            raise InputValidationError(
                f"solver={solver!r} requires a dense numpy.ndarray H"
            )

        n_basis = int(H.shape[0])
        n_occ   = int(n_electrons)
        # We need exactly n_occ eigenpairs; allow the caller to request more.
        n_state = int(opts.get("n_state", n_occ))
        if n_state < n_occ:
            raise InputValidationError(
                f"n_state ({n_state}) must be >= n_electrons ({n_occ}) "
                "to compute the density matrix from eigenvectors."
            )
        opts["n_state"] = n_state

        w, v = eigh(
            H, S=S,
            solver=solver,
            backend_opts=opts,
            return_eigenvectors=True,
        )

        v_occ = v[:, :n_occ]
        if np.iscomplexobj(v_occ):
            D: np.ndarray = v_occ @ v_occ.conj().T
        else:
            D = v_occ @ v_occ.T

        energy = float(np.sum(w[:n_occ]))
        return (D, energy) if return_energy else D

    # ------------------------------------------------------------------
    # Native ELSI DM path (ELPA, OMM, PEXSI, EigenExa, NTPoly)
    # ------------------------------------------------------------------
    if solver in ("pexsi", "ntpoly"):
        if not _is_csr(H):
            raise InputValidationError(f"solver={solver!r} requires H as scipy.sparse.csr_matrix")
        if S is not None and not _is_csr(S):
            raise InputValidationError(f"solver={solver!r} requires S as scipy.sparse.csr_matrix (or omit S)")

        import scipy.sparse  # type: ignore

        n_basis = int(H.shape[0])
        if H.shape[0] != H.shape[1]:
            raise InputValidationError("H must be square")

        # ------------------------------------------------------------------ PEXSI
        if solver == "pexsi":
            opts.setdefault("pexsi_np_per_pole", 1)
            opts.setdefault("pexsi_n_mu", 1)
            opts.setdefault("pexsi_n_pole", 40)

            H_coo = H.tocoo()
            H_coo.sum_duplicates()
            ham_val = np.asarray(H_coo.data, dtype=np.float64)
            row_ind_1 = np.asarray(H_coo.row, dtype=np.int32) + 1
            col_ind_1 = np.asarray(H_coo.col, dtype=np.int32) + 1
            row_out = H_coo.row
            col_out = H_coo.col

            ovlp_val = None
            if S is not None:
                S_coo = S.tocoo()
                S_coo.sum_duplicates()
                if not (S_coo.row.shape == H_coo.row.shape and S_coo.col.shape == H_coo.col.shape):
                    raise InputValidationError("For PEXSI, sparse S must share the coordinate list of H.")
                ovlp_val = np.asarray(S_coo.data, dtype=np.float64)

        # ----------------------------------------------------------------- NTPoly
        else:
            # NTPoly requires spectrum bounds to build the polynomial expansion.
            # Estimate them via the Gershgorin circle theorem if not provided.
            if "spectrum_width" not in opts:
                diag = np.asarray(H.diagonal(), dtype=np.float64)
                abs_row_sum = np.asarray(np.abs(H).sum(axis=1), dtype=np.float64).ravel()
                off_diag_sum = abs_row_sum - np.abs(diag)
                lambda_min = float((diag - off_diag_sum).min())
                lambda_max = float((diag + off_diag_sum).max())
                margin = max(0.1 * (lambda_max - lambda_min), 0.5)
                opts["spectrum_width"] = (lambda_max - lambda_min) + 2.0 * margin
            opts.setdefault("energy_gap", 0.1)
            opts.setdefault("ntpoly_method", 2)   # TRS4 — stable polynomial method
            opts.setdefault("ntpoly_tol", 1e-8)
            opts.setdefault("ntpoly_filter", 1e-8)
            opts.setdefault("ntpoly_max_iter", 100)

            H_coo = H.tocoo()
            H_coo.sum_duplicates()
            h_row = np.asarray(H_coo.row, dtype=np.int32)
            h_col = np.asarray(H_coo.col, dtype=np.int32)
            h_data = np.asarray(H_coo.data, dtype=np.float64)

            if S is not None:
                S_coo = S.tocoo()
                S_coo.sum_duplicates()
                if not (S_coo.row.shape == H_coo.row.shape and S_coo.col.shape == H_coo.col.shape):
                    raise InputValidationError("For NTPoly, sparse S must share the coordinate list of H.")
                ovlp_val = np.asarray(S_coo.data, dtype=np.float64)
                row_out = h_row
                col_out = h_col
                ham_val = h_data
                row_ind_1 = h_row + 1
                col_ind_1 = h_col + 1
            else:
                # CRITICAL: NTPoly calls InverseSquareRoot(ovlp) unconditionally on
                # the first solve even when unit_ovlp=1.  Passing a zero array causes
                # a segfault.  We must supply an actual sparse identity: 1.0 on each
                # diagonal entry, 0.0 off-diagonal, with all n diagonal positions
                # present in the pattern (extend H's pattern if any are missing).
                is_diag = h_row == h_col
                diag_present = np.zeros(n_basis, dtype=bool)
                diag_present[h_row[is_diag]] = True
                missing = np.where(~diag_present)[0].astype(np.int32)

                if len(missing) > 0:
                    row_merged = np.concatenate([h_row, missing])
                    col_merged = np.concatenate([h_col, missing])
                    ham_val = np.concatenate([h_data, np.zeros(len(missing), dtype=np.float64)])
                else:
                    row_merged = h_row
                    col_merged = h_col
                    ham_val = h_data

                # Identity: 1.0 where row == col, 0.0 elsewhere
                ovlp_val = np.where(row_merged == col_merged, 1.0, 0.0).astype(np.float64)
                row_ind_1 = row_merged + 1
                col_ind_1 = col_merged + 1
                row_out = row_merged
                col_out = col_merged

        dm_val, energy = _elsi_dm_real_coo(
            ham_val,
            row_ind_1,
            col_ind_1,
            n_basis,
            ovlp_val,
            solver_map[solver],
            float(n_electrons),
            int(opts.get("n_state", min(int(n_electrons), n_basis))),
            opts,
        )
        D_sparse = scipy.sparse.coo_matrix((dm_val, (row_out, col_out)), shape=(n_basis, n_basis)).tocsr()
        return (D_sparse, float(energy)) if return_energy else D_sparse

    is_complex = np.iscomplexobj(H) or (S is not None and np.iscomplexobj(S))
    if is_complex:
        Hf = _as_fortran_c128(H, "H")
        Sf = _as_fortran_c128(S, "S") if S is not None else None
    else:
        Hf = _as_fortran_f64(H, "H")
        Sf = _as_fortran_f64(S, "S") if S is not None else None

    if is_complex:
        D, energy = _elsi_dm_complex_dense(
            Hf,
            Sf,
            solver_map[solver],
            float(n_electrons),
            int(opts.get("n_state", min(int(n_electrons), int(Hf.shape[0])))),
            opts,
        )
    else:
        D, energy = _elsi_dm_real_dense(
            Hf,
            Sf,
            solver_map[solver],
            float(n_electrons),
            int(opts.get("n_state", min(int(n_electrons), int(Hf.shape[0])))),
            opts,
        )

    return (D, float(energy)) if return_energy else D

