"""SLEPc-SIP (SIPS) sparse eigensolver and density-matrix tests.

SIPS solves sparse eigenproblems using SLEPc's shift-and-invert spectral
transformation.  It is an optional solver that requires SLEPc/PETSc to be
installed and the package to be built with ``-DPYELSI_ENABLE_SIPS=ON``.

Skip-condition: if pyELSI was not compiled with SIPS support, the whole module
is skipped.  All other tests must pass regardless of whether SIPS is enabled.

Build with SIPS:
    pip install -e . --no-build-isolation -CPYELSI_ENABLE_SIPS=ON

Then run:
    python -m pytest tests/test_sips_eigh.py -v -s
"""
import numpy as np
import pytest
import scipy.linalg
import scipy.sparse

import pyelsi
from pyelsi._core import build_info

# ---------------------------------------------------------------------------
# Skip the whole module if SIPS was not compiled in.
# ---------------------------------------------------------------------------
_info = build_info()
if not _info.get("has_sips", False):
    pytest.skip(
        "SLEPc-SIP (SIPS) was not compiled into this pyELSI build "
        "(rebuild with -DPYELSI_ENABLE_SIPS=ON and SLEPc/PETSc installed).",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sparse_sym(rng, n, density=0.1):
    """Random sparse symmetric matrix."""
    A = scipy.sparse.random(n, n, density=density, format="csr", random_state=rng)
    A = (A + A.T) / 2
    # Ensure diagonal dominance for positive-definiteness of shifted problem
    A = A + scipy.sparse.eye(n) * (n * 0.5)
    return A.tocsr()


def _sparse_spd(rng, n, density=0.1):
    """Random sparse SPD matrix via A^T A + diagonal shift."""
    A = scipy.sparse.random(n, n, density=density, format="csr", random_state=rng)
    B = A.T @ A + scipy.sparse.eye(n) * 1.0
    return B.tocsr()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sips_eigh_standard():
    """SIPS eigenvalues match SciPy for a sparse standard eigenproblem."""
    rng = np.random.default_rng(0)
    n = 100
    n_state = n // 2
    H_sp = _sparse_sym(rng, n, density=0.15)
    H_dense = H_sp.toarray()

    w_ref = scipy.linalg.eigh(H_dense, eigvals_only=True)

    w, v = pyelsi.eigh(
        H_sp, solver="sips",
        backend_opts={"n_state": n_state, "n_electron": float(n_state)},
    )

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[SIPS vs SciPy] standard n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
        f"  mean|Δw| = {np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-8, atol=1e-8)

    # Eigenvectors should be orthonormal
    vv = v[:, :n_state]
    orth_err = np.max(np.abs(vv.T @ vv - np.eye(n_state)))
    print(f"  max|VᵀV - I| = {orth_err:.3e}")
    assert orth_err < 1e-7, f"Eigenvectors not orthonormal: {orth_err:.3e}"


def test_sips_eigh_generalized():
    """SIPS eigenvalues match SciPy for a sparse generalized eigenproblem."""
    rng = np.random.default_rng(5)
    n = 80
    n_state = n // 2
    H_sp = _sparse_sym(rng, n, density=0.15)
    S_sp = _sparse_spd(rng, n, density=0.10)
    H_dense = H_sp.toarray()
    S_dense = S_sp.toarray()

    w_ref = scipy.linalg.eigh(H_dense, S_dense, eigvals_only=True)

    w, v = pyelsi.eigh(
        H_sp, S=S_sp, solver="sips",
        backend_opts={"n_state": n_state, "n_electron": float(n_state)},
    )

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[SIPS vs SciPy] generalized n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
        f"  mean|Δw| = {np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-7, atol=1e-7)


def test_sips_eigh_eigenvalues_only():
    """SIPS returns eigenvalues correctly when return_eigenvectors=False."""
    rng = np.random.default_rng(77)
    n = 60
    n_state = n // 2
    H_sp = _sparse_sym(rng, n, density=0.2)
    H_dense = H_sp.toarray()

    w_ref = scipy.linalg.eigh(H_dense, eigvals_only=True)

    result = pyelsi.eigh(
        H_sp, solver="sips",
        backend_opts={"n_state": n_state, "n_electron": float(n_state)},
        return_eigenvectors=False,
    )
    w = result[0] if isinstance(result, tuple) else result

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[SIPS vs SciPy] eigenvalues-only n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
    )
    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-8, atol=1e-8)


def test_sips_explicit_interval():
    """SIPS respects explicit eigenvalue interval from backend_opts."""
    rng = np.random.default_rng(11)
    n = 50
    n_state = n // 2
    H_sp = _sparse_sym(rng, n, density=0.2)
    H_dense = H_sp.toarray()

    w_ref = scipy.linalg.eigh(H_dense, eigvals_only=True)

    # Provide explicit interval (Gershgorin-based estimate + margin)
    diag = np.diag(H_dense)
    off = np.abs(H_dense).sum(axis=1) - np.abs(diag)
    ev_min = float((diag - off).min()) - 1.0
    ev_max = float((diag + off).max()) + 1.0

    w, _ = pyelsi.eigh(
        H_sp, solver="sips",
        backend_opts={
            "n_state": n_state,
            "n_electron": float(n_state),
            "sips_ev_min": ev_min,
            "sips_ev_max": ev_max,
            "sips_n_slice": 2 * n_state,
        },
    )

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[SIPS explicit interval] n={n} n_state={n_state}",
        f"  interval=[{ev_min:.2f}, {ev_max:.2f}]",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
    )
    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# density_matrix via eigh (D = V_occ @ V_occ.T)
# ---------------------------------------------------------------------------

def _ref_dm_dense(H_dense, n_electrons):
    """SciPy reference density matrix: D = V_occ @ V_occ.T."""
    _, v = scipy.linalg.eigh(H_dense)
    return v[:, :n_electrons] @ v[:, :n_electrons].T


def test_sips_density_matrix_standard():
    """SIPS density_matrix matches SciPy reference (standard problem)."""
    rng = np.random.default_rng(0)
    n, ne = 100, 50
    H_sp = _sparse_sym(rng, n, density=0.15)
    H_dense = H_sp.toarray()

    D_ref = _ref_dm_dense(H_dense, ne)
    D = pyelsi.density_matrix(H_sp, n_electrons=ne, solver="sips")

    tr_D   = float(np.trace(D))
    frob   = float(np.linalg.norm(D - D_ref, ord="fro"))
    max_ab = float(np.max(np.abs(D - D_ref)))
    print(
        f"\n[SIPS DM vs SciPy] standard n={n} ne={ne}",
        f"\n  Tr(D)={tr_D:.4f}  (expected {ne})",
        f"  ||D-Dref||_F={frob:.3e}",
        f"  max|D-Dref|={max_ab:.3e}",
    )
    np.testing.assert_allclose(tr_D, float(ne), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(D, D_ref, rtol=1e-7, atol=1e-7)


def test_sips_density_matrix_returns_energy():
    """SIPS density_matrix with return_energy=True returns band energy."""
    rng = np.random.default_rng(3)
    n, ne = 80, 40
    H_sp = _sparse_sym(rng, n, density=0.15)
    H_dense = H_sp.toarray()

    w_ref = scipy.linalg.eigh(H_dense, eigvals_only=True)
    energy_ref = float(np.sum(w_ref[:ne]))

    D, energy = pyelsi.density_matrix(H_sp, n_electrons=ne, solver="sips", return_energy=True)

    print(
        f"\n[SIPS DM energy] n={n} ne={ne}",
        f"\n  band_energy={energy:.6f}  ref={energy_ref:.6f}",
        f"  |Δ|={abs(energy - energy_ref):.3e}",
    )
    np.testing.assert_allclose(float(np.trace(D)), float(ne), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(energy, energy_ref, rtol=1e-7, atol=1e-7)
