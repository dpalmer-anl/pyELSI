"""Density-matrix tests for dense problems (standard and generalized).

The reference is computed with scipy.linalg.eigh: D_ref = V_occ @ V_occ^T
(standard) or V_occ @ V_occ^T (S-orthonormal columns for generalized).
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.linalg  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(rng, n):
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2.0


def _spd(rng, n):
    b = rng.standard_normal((n, n))
    return b @ b.T + np.eye(n) * 1e-2


def _ref_density_matrix(H, n_electrons, S=None):
    """SciPy reference density matrix."""
    if S is None:
        w, v = scipy.linalg.eigh(H)
    else:
        w, v = scipy.linalg.eigh(H, S)
    occ = v[:, :n_electrons]
    return occ @ occ.T


# ---------------------------------------------------------------------------
# Standard (S = I) density matrix — default ELPA solver
# ---------------------------------------------------------------------------

def test_density_matrix_elpa_standard_matches_scipy():
    """ELPA DM (standard, S=I) matches the SciPy eigenvector reference."""
    import pyelsi

    rng = np.random.default_rng(2)
    n, ne = 50, 25
    H = _sym(rng, n)

    D_ref = _ref_density_matrix(H, ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver="elpa")

    diff = D - D_ref
    frob = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref, ord="fro"))

    print(
        "\n[pyELSI vs SciPy] density_matrix solver=elpa (standard)",
        f"n={n} ne={ne}",
        f"||D-Dref||_F={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|={max_abs:.3e}",
    )

    np.testing.assert_allclose(D, D_ref, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Generalized (S ≠ I) density matrix — ELPA solver
# ---------------------------------------------------------------------------

def test_density_matrix_elpa_generalized_matches_scipy():
    """ELPA DM (generalized, S given) matches the SciPy eigenvector reference."""
    import pyelsi

    rng = np.random.default_rng(5)
    n, ne = 30, 15
    H = _sym(rng, n)
    S = _spd(rng, n)

    D_ref = _ref_density_matrix(H, ne, S=S)
    D = pyelsi.density_matrix(H, S=S, n_electrons=ne, solver="elpa")

    diff = D - D_ref
    frob = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref, ord="fro"))

    print(
        "\n[pyELSI vs SciPy] density_matrix solver=elpa (generalized)",
        f"n={n} ne={ne}",
        f"||D-Dref||_F={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|={max_abs:.3e}",
    )

    np.testing.assert_allclose(D, D_ref, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# OMM density matrix
# ---------------------------------------------------------------------------

def test_density_matrix_omm_standard_matches_scipy():
    """OMM DM (standard, S=I) matches the SciPy eigenvector reference."""
    import pyelsi

    rng = np.random.default_rng(6)
    n, ne = 40, 20
    H = _sym(rng, n)

    D_ref = _ref_density_matrix(H, ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver="omm")

    diff = D - D_ref
    frob = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref, ord="fro"))

    print(
        "\n[pyELSI vs SciPy] density_matrix solver=omm (standard)",
        f"n={n} ne={ne}",
        f"||D-Dref||_F={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|={max_abs:.3e}",
    )

    np.testing.assert_allclose(D, D_ref, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# return_energy flag
# ---------------------------------------------------------------------------

def test_density_matrix_return_energy():
    """density_matrix(return_energy=True) returns (D, float) with a finite energy."""
    import pyelsi

    rng = np.random.default_rng(7)
    n, ne = 20, 10
    H = _sym(rng, n)

    result = pyelsi.density_matrix(H, n_electrons=ne, solver="elpa", return_energy=True)
    assert isinstance(result, tuple) and len(result) == 2, "Expected (D, energy) tuple"
    D, energy = result
    assert D.shape == (n, n), f"D shape mismatch: {D.shape}"
    assert np.isfinite(energy), f"Energy is not finite: {energy}"

    print(
        "\n[pyELSI] density_matrix solver=elpa return_energy=True",
        f"n={n} ne={ne}",
        f"energy={energy:.6f}",
    )
