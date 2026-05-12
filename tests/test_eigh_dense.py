"""Eigenvalue tests for dense symmetric / Hermitian problems.

ELPA is the primary eigensolver backend in ELSI.  OMM and PEXSI are
density-matrix solvers; ELSI routes their ``ev`` calls through ELPA
internally, so we test ELPA explicitly and verify OMM via its DM path.
All results are compared against scipy.linalg.eigh.
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


def _herm(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (a + a.conj().T) / 2.0


# ---------------------------------------------------------------------------
# Real ELPA eigh — standard and generalized
# ---------------------------------------------------------------------------

def test_eigh_elpa_standard():
    """ELPA eigenvalues and eigenvectors match SciPy for a standard problem."""
    import pyelsi

    rng = np.random.default_rng(0)
    n = 40
    H = _sym(rng, n)

    w_ref, v_ref = scipy.linalg.eigh(H)
    w, v = pyelsi.eigh(H, solver="elpa")

    dw = w - w_ref
    print(
        "\n[pyELSI vs SciPy] eigh solver=elpa (standard)",
        f"n={n}",
        f"max|dw|={np.max(np.abs(dw)):.3e}",
        f"mean|dw|={np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-10)
    proj = v.T @ v_ref
    np.testing.assert_allclose(np.abs(np.diag(proj)), np.ones(n), rtol=1e-8, atol=1e-8)


def test_eigh_elpa_generalized():
    """ELPA eigenvalues and eigenvectors match SciPy for a generalized problem."""
    import pyelsi

    rng = np.random.default_rng(1)
    n = 30
    H = _sym(rng, n)
    S = _spd(rng, n)

    w_ref, v_ref = scipy.linalg.eigh(H, S)
    w, v = pyelsi.eigh(H, S=S, solver="elpa")

    dw = w - w_ref
    print(
        "\n[pyELSI vs SciPy] eigh solver=elpa (generalized)",
        f"n={n}",
        f"max|dw|={np.max(np.abs(dw)):.3e}",
        f"mean|dw|={np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w, w_ref, rtol=1e-9, atol=1e-9)
    proj = v.T @ (S @ v_ref)
    np.testing.assert_allclose(np.abs(np.diag(proj)), np.ones(n), rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# Complex ELPA eigh
# ---------------------------------------------------------------------------

def test_eigh_elpa_complex_standard():
    """ELPA complex eigh eigenvalues and eigenvectors match SciPy."""
    import pyelsi

    rng = np.random.default_rng(10)
    n = 32
    H = _herm(rng, n).astype(np.complex128)

    w_ref, v_ref = scipy.linalg.eigh(H)
    w, v = pyelsi.eigh(H, solver="elpa")

    dw = w - w_ref
    print(
        "\n[pyELSI vs SciPy] eigh solver=elpa (complex standard)",
        f"n={n}",
        f"max|dw|={np.max(np.abs(dw)):.3e}",
        f"mean|dw|={np.mean(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w, w_ref, rtol=1e-9, atol=1e-9)
    proj = v.conj().T @ v_ref
    np.testing.assert_allclose(np.abs(np.diag(proj)), np.ones(n), rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# Eigenvalues only (no eigenvectors)
# ---------------------------------------------------------------------------

def test_eigh_elpa_eigenvalues_only():
    """Requesting return_eigenvectors=False returns only eigenvalues."""
    import pyelsi

    rng = np.random.default_rng(20)
    n = 25
    H = _sym(rng, n)

    w_ref = scipy.linalg.eigh(H, eigvals_only=True)
    result = pyelsi.eigh(H, solver="elpa", return_eigenvectors=False)

    # When return_eigenvectors=False the API returns (w, None); unpack accordingly.
    w = result[0] if isinstance(result, tuple) else result

    dw = w - w_ref
    print(
        "\n[pyELSI vs SciPy] eigh solver=elpa (eigenvalues only)",
        f"n={n}",
        f"max|dw|={np.max(np.abs(dw)):.3e}",
    )

    np.testing.assert_allclose(w, w_ref, rtol=1e-10, atol=1e-10)
