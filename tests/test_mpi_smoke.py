"""MPI smoke tests — run under mpirun, e.g.:

    mpirun -n 2 python -m pytest tests/test_mpi_smoke.py -v -s

Strategy: every rank solves the *full* problem independently using its own
MPI_COMM_SELF communicator (force_single_proc=1).  After the solve we
allgather results and verify:
  1. All ranks obtained identical density matrices (consistency).
  2. Each rank's result matches the scipy.linalg.eigh reference (accuracy).

MPI-compatible solvers covered here:
  • ELPA   – dense eigensolver, supports distributed BLACS grids
  • OMM    – dense density-matrix polynomial solver
  • PEXSI  – sparse pole-expansion (GENERIC_COO path)
  • NTPoly – sparse polynomial (GENERIC_COO path)

All four require BLACS/MPI; each test exercises the correct per-rank
serial path so the test suite works without a full ScaLAPACK distribution.
"""
import numpy as np
import pytest

from mpi4py import MPI  # hard dependency; install with pip install -e ".[test,mpi]"
import scipy.linalg
import scipy.sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym_dense(rng, n, diag_shift=5.0):
    """Random real symmetric positive-definite dense matrix."""
    A = rng.standard_normal((n, n))
    A = 0.5 * (A + A.T)
    A += diag_shift * np.eye(n)
    return A


def _banded_csr(rng, n, diag_shift=5.0):
    """Sparse banded symmetric matrix in CSR format."""
    main = rng.standard_normal(n) + diag_shift
    off1 = 0.1 * rng.standard_normal(n - 1)
    off2 = 0.05 * rng.standard_normal(n - 2)
    return scipy.sparse.diags(
        diagonals=[main, off1, off1, off2, off2],
        offsets=[0, -1, 1, -2, 2],
        shape=(n, n),
        format="csr",
    )


def _ref_density_matrix(H_dense, n_electrons):
    w, v = scipy.linalg.eigh(H_dense)
    occ = v[:, :n_electrons]
    return occ @ occ.T


def _check_dm_mpi(comm, D_dense, D_ref, label, *, atol_ref=1e-4, atol_cross=1e-8):
    rank = comm.Get_rank()
    size = comm.Get_size()

    frob_ref = float(np.linalg.norm(D_dense - D_ref))
    print(
        f"\n[MPI smoke] {label} rank={rank}/{size}",
        f"||D-Dref||_F={frob_ref:.3e}",
    )

    D_all = comm.allgather(D_dense)
    for other_rank, D_other in enumerate(D_all[1:], start=1):
        np.testing.assert_allclose(
            D_other, D_all[0], rtol=atol_cross, atol=atol_cross,
            err_msg=f"{label}: DM differs between rank 0 and rank {other_rank}",
        )

    np.testing.assert_allclose(
        D_dense, D_ref, atol=atol_ref,
        err_msg=f"{label}: DM rank {rank} does not match scipy reference",
    )


def _check_dm_sparse_mpi(comm, D_sparse, H_sparse, D_ref, label, *, atol_ref=1e-4, atol_cross=1e-8):
    rank = comm.Get_rank()
    size = comm.Get_size()

    D_dense = D_sparse.toarray()
    mask = H_sparse.toarray() != 0.0
    frob_pat = float(np.linalg.norm(D_dense[mask] - D_ref[mask]))
    print(
        f"\n[MPI smoke] {label} rank={rank}/{size}",
        f"||D-Dref||_F(pattern)={frob_pat:.3e}",
    )

    D_all = comm.allgather(D_dense)
    for other_rank, D_other in enumerate(D_all[1:], start=1):
        np.testing.assert_allclose(
            D_other, D_all[0], rtol=atol_cross, atol=atol_cross,
            err_msg=f"{label}: DM differs between rank 0 and rank {other_rank}",
        )

    np.testing.assert_allclose(
        D_dense[mask], D_ref[mask], atol=atol_ref,
        err_msg=f"{label}: DM on sparsity pattern (rank {rank}) does not match scipy reference",
    )


# ---------------------------------------------------------------------------
# build_info: identical on all ranks
# ---------------------------------------------------------------------------

@pytest.mark.mpi
def test_mpi_smoke_import_and_build_info():
    """build_info() must return the same dict on every MPI rank."""
    import pyelsi

    comm = MPI.COMM_WORLD
    info = pyelsi.build_info()
    infos = comm.allgather(info)

    print(
        f"\n[MPI smoke] build_info rank={comm.Get_rank()}/{comm.Get_size()}",
        f"has_mpi={info.get('has_mpi')}",
        f"backend={info.get('backend')}",
    )

    assert all(isinstance(x, dict) for x in infos), "build_info must return a dict on all ranks"
    for i_info in infos[1:]:
        assert i_info == infos[0], "build_info must be identical on all ranks"


# ---------------------------------------------------------------------------
# ELPA dense DM — MPI ranks must agree and match scipy
# ---------------------------------------------------------------------------

@pytest.mark.mpi
def test_mpi_smoke_elpa_dm_all_ranks_identical():
    """ELPA dense DM: each rank solves independently (force_single_proc);
    all ranks must produce identical results matching scipy reference."""
    import pyelsi

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Run under mpirun -n >=2")

    rng = np.random.default_rng(1001)
    n, ne = 40, 20
    H = _sym_dense(rng, n)
    D_ref = _ref_density_matrix(H, ne)

    D_dense, _ = pyelsi.density_matrix(
        H, n_electrons=ne, solver="elpa", return_energy=True,
        backend_opts={"force_single_proc": 1},
    )

    _check_dm_mpi(comm, D_dense, D_ref, f"ELPA DM n={n} ne={ne}")


# ---------------------------------------------------------------------------
# OMM dense DM — MPI ranks must agree and match scipy
# ---------------------------------------------------------------------------

@pytest.mark.mpi
def test_mpi_smoke_omm_dm_all_ranks_identical():
    """OMM dense DM: each rank solves independently (force_single_proc);
    all ranks must produce identical results matching scipy reference."""
    import pyelsi

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Run under mpirun -n >=2")

    rng = np.random.default_rng(1002)
    n, ne = 40, 20
    H = _sym_dense(rng, n)
    D_ref = _ref_density_matrix(H, ne)

    D_dense, _ = pyelsi.density_matrix(
        H, n_electrons=ne, solver="omm", return_energy=True,
        backend_opts={"force_single_proc": 1},
    )

    _check_dm_mpi(comm, D_dense, D_ref, f"OMM DM n={n} ne={ne}", atol_ref=1e-3)


# ---------------------------------------------------------------------------
# PEXSI sparse DM — MPI ranks must agree and match scipy
# ---------------------------------------------------------------------------

@pytest.mark.mpi
def test_mpi_smoke_pexsi_dm_all_ranks_identical():
    """PEXSI sparse DM: each rank solves independently (force_single_proc);
    all ranks must produce identical results matching scipy reference on pattern."""
    import pyelsi

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Run under mpirun -n >=2")

    rng = np.random.default_rng(456)
    n, ne = 60, 30
    H = _banded_csr(rng, n)
    D_ref = _ref_density_matrix(H.toarray(), ne)

    D = pyelsi.density_matrix(
        H, n_electrons=ne, solver="pexsi",
        backend_opts={"force_single_proc": 1},
    )

    _check_dm_sparse_mpi(comm, D, H, D_ref, f"PEXSI DM n={n} ne={ne}", atol_ref=1e-4)


# ---------------------------------------------------------------------------
# NTPoly sparse DM — MPI ranks must agree and match scipy
# ---------------------------------------------------------------------------

@pytest.mark.mpi
def test_mpi_smoke_ntpoly_dm_all_ranks_identical():
    """NTPoly sparse DM: each rank solves independently (force_single_proc);
    all ranks must produce identical results matching scipy reference on pattern."""
    import pyelsi

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("Run under mpirun -n >=2")

    rng = np.random.default_rng(789)
    n, ne = 40, 20
    H = _banded_csr(rng, n)
    D_ref = _ref_density_matrix(H.toarray(), ne)

    D = pyelsi.density_matrix(
        H, n_electrons=ne, solver="ntpoly",
        backend_opts={"force_single_proc": 1},
    )

    _check_dm_sparse_mpi(comm, D, H, D_ref, f"NTPoly DM n={n} ne={ne}", atol_ref=1e-3)
