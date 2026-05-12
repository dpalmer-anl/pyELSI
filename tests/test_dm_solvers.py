"""Density-matrix tests for all supported ELSI solvers.

Dense solvers (ELPA, OMM) are compared against a SciPy eigh reference.
Sparse solvers (PEXSI, NTPoly) are tested on a banded Hamiltonian; results
are compared on the sparsity pattern supplied to ELSI.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.linalg  # noqa: E402
import scipy.sparse  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(rng, n):
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2.0


def _banded_csr(rng, n):
    """Symmetric banded CSR matrix stored with both upper and lower diagonals."""
    main = rng.standard_normal(n)
    off1 = 0.1 * rng.standard_normal(n - 1)
    off2 = 0.05 * rng.standard_normal(n - 2)
    return scipy.sparse.diags(
        diagonals=[main, off1, off1, off2, off2],
        offsets=[0, -1, 1, -2, 2],
        shape=(n, n),
        format="csr",
    )


def _ref_density_matrix(H, n_electrons):
    """SciPy reference: D = V_occ @ V_occ^T."""
    w, v = scipy.linalg.eigh(H)
    occ = v[:, :n_electrons]
    return occ @ occ.T


# ---------------------------------------------------------------------------
# Dense DM solvers: ELPA and OMM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "solver,rtol,atol",
    [
        ("elpa", 1e-10, 1e-10),
        ("omm",  1e-10, 1e-10),
    ],
)
def test_density_matrix_dense_matches_scipy(solver, rtol, atol):
    """Dense DM via ELPA / OMM must reproduce the SciPy reference."""
    import pyelsi

    rng = np.random.default_rng(3)
    n, ne = 30, 15
    H = _sym(rng, n)

    D_ref = _ref_density_matrix(H, ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver=solver)

    diff = D - D_ref
    frob = float(np.linalg.norm(diff, ord="fro"))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref, ord="fro"))

    print(
        f"\n[pyELSI vs SciPy] density_matrix solver={solver}",
        f"n={n} ne={ne}",
        f"||D-Dref||_F={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|={max_abs:.3e}",
        f"rtol={rtol:g} atol={atol:g}",
    )

    np.testing.assert_allclose(D, D_ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Sparse DM: PEXSI
# ---------------------------------------------------------------------------

def test_sparse_density_matrix_pexsi_matches_scipy_on_pattern():
    """PEXSI sparse DM on banded CSR must match SciPy on the sparsity pattern."""
    import pyelsi

    rng = np.random.default_rng(4)
    n, ne = 40, 20
    H = _banded_csr(rng, n)

    D_ref = _ref_density_matrix(H.toarray(), ne)
    D = pyelsi.density_matrix(H, n_electrons=ne, solver="pexsi")
    assert scipy.sparse.issparse(D), "PEXSI DM must return a sparse matrix"

    D_dense = D.toarray()
    mask = H.toarray() != 0.0
    diff = D_dense[mask] - D_ref[mask]
    frob = float(np.linalg.norm(diff))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref))

    print(
        "\n[pyELSI vs SciPy] density_matrix solver=pexsi (sparse, on pattern)",
        f"n={n} ne={ne}",
        f"nnz={H.nnz}",
        f"||D-Dref||_F(pattern)={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|(pattern)={max_abs:.3e}",
    )

    # PEXSI returns values only on the supplied sparsity pattern; compare there.
    np.testing.assert_allclose(D_dense[mask], D_ref[mask], rtol=2e-4, atol=2e-4)


# ---------------------------------------------------------------------------
# Sparse DM: NTPoly
# ---------------------------------------------------------------------------

def test_sparse_density_matrix_ntpoly_matches_scipy_on_pattern():
    """NTPoly sparse DM on banded CSR must match SciPy on the sparsity pattern.

    NTPoly is a polynomial-expansion DM solver.  Its accuracy is controlled by
    ``ntpoly_tol`` and it only returns values on the supplied sparsity pattern,
    so we compare there with a generous tolerance.
    """
    import pyelsi

    rng = np.random.default_rng(4)
    n, ne = 40, 20
    H = _banded_csr(rng, n)

    D_ref = _ref_density_matrix(H.toarray(), ne)
    D = pyelsi.density_matrix(
        H,
        n_electrons=ne,
        solver="ntpoly",
        backend_opts={
            "ntpoly_tol": 1e-8,
            "ntpoly_filter": 1e-8,
            "ntpoly_max_iter": 200,
        },
    )
    assert scipy.sparse.issparse(D), "NTPoly DM must return a sparse matrix"

    D_dense = D.toarray()
    mask = H.toarray() != 0.0
    diff = D_dense[mask] - D_ref[mask]
    frob = float(np.linalg.norm(diff))
    max_abs = float(np.max(np.abs(diff)))
    rel_frob = frob / float(np.linalg.norm(D_ref))

    print(
        "\n[pyELSI vs SciPy] density_matrix solver=ntpoly (sparse, on pattern)",
        f"n={n} ne={ne}",
        f"nnz={H.nnz}",
        f"||D-Dref||_F(pattern)={frob:.3e}",
        f"rel_F={rel_frob:.3e}",
        f"max|D-Dref|(pattern)={max_abs:.3e}",
    )

    np.testing.assert_allclose(D_dense[mask], D_ref[mask], rtol=1e-4, atol=1e-4)
