"""Complex Hermitian eigenvalue and density-matrix tests.

All results are compared against scipy.linalg.eigh.
ELPA is the primary complex eigensolver; OMM is also tested for DM.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.linalg  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _herm(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (a + a.conj().T) / 2.0


def _ref_dm_complex(H, n_electrons):
    """SciPy complex density matrix reference: D = V_occ @ V_occ^†."""
    w, v = scipy.linalg.eigh(H)
    occ = v[:, :n_electrons]
    return occ @ occ.conj().T


# ---------------------------------------------------------------------------
# Complex eigh (ELPA)
# ---------------------------------------------------------------------------

def test_eigh_complex_elpa_matches_scipy():
    """ELPA complex eigh eigenvalues and eigenvectors match SciPy."""
    import pyelsi

    rng = np.random.default_rng(10)
    n = 32
    H = _herm(rng, n).astype(np.complex128)

    w_ref, v_ref = scipy.linalg.eigh(H)
    # Request all n eigenpairs so eigenvector comparison covers the full spectrum.
    w, v = pyelsi.eigh(H, solver="elpa", backend_opts={"n_state": n, "n_electron": float(n)})

    dw = w[:n] - w_ref
    print(
        "\n[pyELSI vs SciPy] complex eigh solver=elpa",
        f"n={n}",
        f"max|dw|={np.max(np.abs(dw)):.3e}",
        f"mean|dw|={np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w[:n], w_ref, rtol=1e-9, atol=1e-9)
    proj = v[:, :n].conj().T @ v_ref
    np.testing.assert_allclose(np.abs(np.diag(proj)), np.ones(n), rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# Complex density matrix (ELPA)
# ---------------------------------------------------------------------------

def test_density_matrix_complex_elpa_matches_scipy():
    """ELPA complex DM matches SciPy reference."""
    import pyelsi

    rng = np.random.default_rng(11)
    n, ne = 40, 20
    H = _herm(rng, n).astype(np.complex128)

    D_ref = _ref_dm_complex(H, ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver="elpa")

    diff = D - D_ref
    frob = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref, ord="fro"))

    print(
        "\n[pyELSI vs SciPy] complex density_matrix solver=elpa",
        f"n={n} ne={ne}",
        f"||D-Dref||_F={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|={max_abs:.3e}",
    )

    np.testing.assert_allclose(D, D_ref, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Complex density matrix (OMM)
# ---------------------------------------------------------------------------

def test_density_matrix_complex_omm_matches_scipy():
    """OMM complex DM matches SciPy reference."""
    import pyelsi

    rng = np.random.default_rng(12)
    n, ne = 36, 18
    H = _herm(rng, n).astype(np.complex128)

    D_ref = _ref_dm_complex(H, ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver="omm")

    diff = D - D_ref
    frob = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref, ord="fro"))

    print(
        "\n[pyELSI vs SciPy] complex density_matrix solver=omm",
        f"n={n} ne={ne}",
        f"||D-Dref||_F={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|={max_abs:.3e}",
    )

    np.testing.assert_allclose(D, D_ref, rtol=1e-8, atol=1e-8)
