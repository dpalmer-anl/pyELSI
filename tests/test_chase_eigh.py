"""ChASE (Chebyshev Accelerated Subspace Iteration) eigensolver tests.

ChASE is compiled into pyELSI when built with PYELSI_ENABLE_CHASE=ON (the
default).  It is a dense, distributed eigensolver designed for partial-spectrum
problems.  All results — both eigh and density_matrix — are compared against
scipy.linalg.eigh / scipy reference.

Run with:
    python -m pytest tests/test_chase_eigh.py -v -s
"""
import numpy as np
import pytest
import scipy.linalg

import pyelsi
from pyelsi._core import build_info


# ---------------------------------------------------------------------------
# Skip the whole module if ChASE was not compiled in.
# ---------------------------------------------------------------------------
_info = build_info()
if not _info.get("has_chase", False):
    pytest.skip(
        "ChASE was not compiled into this pyELSI build "
        "(rebuild with -DPYELSI_ENABLE_CHASE=ON).",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(rng, n):
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2.0


def _spd(rng, n):
    b = rng.standard_normal((n, n))
    return b @ b.T + np.eye(n) * 5e-2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chase_eigh_standard():
    """ChASE eigenvalues match SciPy for a real standard eigenproblem."""
    rng = np.random.default_rng(7)
    n = 60
    n_state = n // 2
    # Diagonal shift ensures well-separated eigenvalues (better ChASE convergence).
    H = _sym(rng, n) + n * np.eye(n)

    w_ref = scipy.linalg.eigh(H, eigvals_only=True)

    w, v = pyelsi.eigh(
        H, solver="chase",
        backend_opts={"n_state": n_state, "n_electron": float(n_state)},
    )

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[ChASE vs SciPy] standard n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
        f"  mean|Δw| = {np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-8, atol=1e-8)
    # Eigenvectors should be orthonormal (columns of v)
    vv = v[:, :n_state]
    orth_err = np.max(np.abs(vv.T @ vv - np.eye(n_state)))
    print(f"  max|VᵀV - I| = {orth_err:.3e}")
    assert orth_err < 1e-7, f"Eigenvectors not orthonormal: {orth_err:.3e}"


def test_chase_eigh_generalized():
    """ChASE eigenvalues match SciPy for a real generalized eigenproblem."""
    rng = np.random.default_rng(7)
    n = 60
    n_state = n // 2
    H = _sym(rng, n)
    S = _spd(rng, n)

    w_ref = scipy.linalg.eigh(H, S, eigvals_only=True)

    w, v = pyelsi.eigh(
        H, S=S, solver="chase",
        backend_opts={"n_state": n_state, "n_electron": float(n_state)},
    )

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[ChASE vs SciPy] generalized n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
        f"  mean|Δw| = {np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-7, atol=1e-7)


def test_chase_eigh_eigenvalues_only():
    """ChASE returns eigenvalues correctly when return_eigenvectors=False."""
    rng = np.random.default_rng(99)
    n = 50
    n_state = n // 2
    H = _sym(rng, n) + n * np.eye(n)

    w_ref = scipy.linalg.eigh(H, eigvals_only=True)

    result = pyelsi.eigh(
        H, solver="chase",
        backend_opts={"n_state": n_state, "n_electron": float(n_state)},
        return_eigenvectors=False,
    )
    w = result[0] if isinstance(result, tuple) else result

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[ChASE vs SciPy] eigenvalues-only n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
    )
    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-8, atol=1e-8)


def test_chase_custom_options():
    """ChASE respects custom filter-degree and tolerance options."""
    rng = np.random.default_rng(123)
    n = 40
    n_state = n // 2
    H = _sym(rng, n) + n * np.eye(n)

    w_ref = scipy.linalg.eigh(H, eigvals_only=True)

    # Explicitly pass all tunable ChASE knobs; use conservative values so the
    # test remains robust across different matrix instances.
    w, _ = pyelsi.eigh(
        H, solver="chase",
        backend_opts={
            "n_state": n_state,
            "n_electron": float(n_state),
            "chase_tol": 1e-10,
            "chase_filter_deg": 30,    # higher degree = more accurate filtering
            "chase_extra_space": 0.30,  # fraction of n_state (0–0.5)
        },
    )

    dw = w[:n_state] - w_ref[:n_state]
    print(
        f"\n[ChASE custom opts] n={n} n_state={n_state}",
        f"\n  max|Δw| = {np.max(np.abs(dw)):.3e}",
    )
    np.testing.assert_allclose(w[:n_state], w_ref[:n_state], rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# density_matrix via eigh (D = V_occ @ V_occ.T)
# ---------------------------------------------------------------------------

def _ref_dm(H, n_electrons):
    """SciPy reference density matrix: D = V_occ @ V_occ.T."""
    _, v = scipy.linalg.eigh(H)
    return v[:, :n_electrons] @ v[:, :n_electrons].T


def test_chase_density_matrix_standard():
    """ChASE density_matrix matches SciPy reference for standard problem."""
    rng = np.random.default_rng(42)
    n, ne = 60, 30
    H = _sym(rng, n) + n * np.eye(n)

    D_ref = _ref_dm(H, ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver="chase")

    tr_D   = float(np.trace(D))
    frob   = float(np.linalg.norm(D - D_ref, ord="fro"))
    max_ab = float(np.max(np.abs(D - D_ref)))
    print(
        f"\n[ChASE DM vs SciPy] standard n={n} ne={ne}",
        f"\n  Tr(D)={tr_D:.4f}  (expected {ne})",
        f"  ||D-Dref||_F={frob:.3e}",
        f"  max|D-Dref|={max_ab:.3e}",
    )
    np.testing.assert_allclose(tr_D, float(ne), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(D, D_ref, rtol=1e-7, atol=1e-7)


def test_chase_density_matrix_returns_energy():
    """ChASE density_matrix with return_energy=True returns band energy."""
    rng = np.random.default_rng(5)
    n, ne = 50, 25
    H = _sym(rng, n) + n * np.eye(n)

    w_ref = scipy.linalg.eigh(H, eigvals_only=True)
    energy_ref = float(np.sum(w_ref[:ne]))

    D, energy = pyelsi.density_matrix(H, n_electrons=ne, solver="chase", return_energy=True)

    print(
        f"\n[ChASE DM energy] n={n} ne={ne}",
        f"\n  band_energy={energy:.6f}  ref={energy_ref:.6f}",
        f"  |Δ|={abs(energy - energy_ref):.3e}",
    )
    np.testing.assert_allclose(float(np.trace(D)), float(ne), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(energy, energy_ref, rtol=1e-7, atol=1e-7)
