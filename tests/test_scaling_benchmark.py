"""
Wall-clock scaling benchmark for pyELSI solvers.

Times every available solver over 15 log-spaced matrix sizes N = 100 … 10,000.
Generates one plot per MPI-rank count in outputs/scaling_nproc_{N}.png.

Solvers benchmarked (density matrix only)
-----------------------------------------
Dense DM    : ELPA, OMM, ChASE (via eigh)
Sparse DM   : PEXSI, NTPoly, SLEPc-SIP (skipped automatically if not compiled in)

How to run
----------
Serial (1 process):
    PYELSI_RUN_SCALING_BENCH=1 pytest tests/test_scaling_benchmark.py -v -s

2 MPI ranks:
    PYELSI_RUN_SCALING_BENCH=1 mpirun -n 2 python -m pytest tests/test_scaling_benchmark.py -v -s

4 MPI ranks:
    PYELSI_RUN_SCALING_BENCH=1 mpirun -n 4 python -m pytest tests/test_scaling_benchmark.py -v -s

Under mpirun each rank solves independently (force_single_proc=1) so the
per-rank timing is directly comparable across different rank counts.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

import pyelsi  # noqa: E402 — hard dependency; install with pip install -e ".[test]"
import scipy.sparse  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# MPI helpers (graceful if mpi4py is absent — serial fallback)
# ---------------------------------------------------------------------------

def _mpi_world():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


def _rank_size():
    comm = _mpi_world()
    return (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)


def _barrier():
    comm = _mpi_world()
    if comm is not None:
        comm.Barrier()


# ---------------------------------------------------------------------------
# Matrix factories
# ---------------------------------------------------------------------------

def _sym_dense(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random symmetric matrix with diagonal shift n to ensure positive definiteness.

    Random normal matrices have eigenvalues O(√n), so a fixed small shift fails
    for large n.  Shifting by n guarantees all eigenvalues are positive and
    well-separated, which is required for ChASE convergence.
    """
    a = rng.standard_normal((n, n))
    return 0.5 * (a + a.T) + float(n) * np.eye(n)


def _banded_csr(rng: np.random.Generator, n: int, diag_shift: float = 5.0):
    main = rng.standard_normal(n) + diag_shift
    off1 = 0.1 * rng.standard_normal(n - 1)
    off2 = 0.05 * rng.standard_normal(n - 2)
    return scipy.sparse.diags(
        [main, off1, off1, off2, off2],
        [0, -1, 1, -2, 2],
        shape=(n, n),
        format="csr",
    )


def _ne(n: int) -> int:
    return max(1, min(n - 1, n // 2))


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

SIZES: list[int] = sorted(set(
    int(round(x))
    for x in np.logspace(np.log10(100), np.log10(10_000), 15)
))


# ---------------------------------------------------------------------------
# Per-solver timers (return None on any exception / unavailable solver)
# ---------------------------------------------------------------------------

def _time_dense_dm(solver: str, n: int, n_procs: int, rng: np.random.Generator) -> float | None:
    """Time pyelsi.density_matrix() for a dense symmetric H."""
    H = _sym_dense(rng, n)
    ne = _ne(n)
    opts: dict = {}
    if n_procs > 1:
        opts["force_single_proc"] = 1
    _barrier()
    t0 = time.perf_counter()
    try:
        pyelsi.density_matrix(H, n_electrons=ne, solver=solver, backend_opts=opts)
    except Exception:
        return None
    return time.perf_counter() - t0


def _time_sparse_dm(solver: str, n: int, n_procs: int, rng: np.random.Generator) -> float | None:
    """Time pyelsi.density_matrix() for a sparse CSR H (PEXSI, NTPoly, SIPS)."""
    # Guard: SIPS calls MPI_ABORT (not catchable) when not compiled in.
    if solver == "sips" and not pyelsi.build_info().get("has_sips", False):
        return None
    H = _banded_csr(rng, n)
    ne = _ne(n)
    opts: dict = {}
    if n_procs > 1:
        opts["force_single_proc"] = 1
    _barrier()
    t0 = time.perf_counter()
    try:
        pyelsi.density_matrix(H, n_electrons=ne, solver=solver, backend_opts=opts)
    except Exception:
        return None
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Solver list
# Entries whose timer returns None for every N are silently omitted from the
# plot, so solvers not compiled in (e.g. SIPS without SLEPc) are safe to list.
# ---------------------------------------------------------------------------

SOLVERS = [
    # Dense DM solvers
    ("ELPA DM",   lambda n, k, r: _time_dense_dm("elpa",  n, k, r)),
    ("OMM DM",    lambda n, k, r: _time_dense_dm("omm",   n, k, r)),
    # Sparse DM solvers (native ELSI path)
    ("PEXSI DM",  lambda n, k, r: _time_sparse_dm("pexsi",  n, k, r)),
    ("NTPoly DM", lambda n, k, r: _time_sparse_dm("ntpoly", n, k, r)),
    # SIPS DM via eigh (skipped automatically if SIPS not compiled in)
    ("SIPS DM",   lambda n, k, r: _time_sparse_dm("sips",   n, k, r)),
]
# Note: ChASE is intentionally excluded from the density matrix benchmark.
# ChASE is designed for extremal eigenpairs (≤ ~20% of the spectrum).  For a
# density matrix we need the lowest n//2 ≈ 50% of eigenpairs, which places the
# spectral gap right at the middle of the spectrum where the Chebyshev filter
# has no leverage.  Use ELPA for dense DM calculations instead.


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.scaling_benchmark
def test_scaling_benchmark():
    """Benchmark all available solvers for N=100…10,000; save plot to outputs/."""
    if os.environ.get("PYELSI_RUN_SCALING_BENCH") != "1":
        pytest.skip("Set PYELSI_RUN_SCALING_BENCH=1 to run this benchmark.")

    rank, n_procs = _rank_size()

    rng = np.random.default_rng(42)
    series: dict[str, list[tuple[int, float]]] = {label: [] for label, _ in SOLVERS}

    for n in SIZES:
        ne = _ne(n)
        if rank == 0:
            print(f"\n  N={n:4d}  n_electrons={ne}", flush=True)
        for label, timer in SOLVERS:
            dt = timer(n, n_procs, rng)
            if dt is not None:
                series[label].append((n, dt))
                if rank == 0:
                    print(f"    {label:<14s}: {dt:.4f} s", flush=True)

    # Only rank 0 writes the plot
    if rank != 0:
        return

    assert any(pts for pts in series.values()), (
        "No solver produced any timing data — check your build."
    )

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]
    # Group dense (solid) vs sparse (dashed) solvers by line style
    sparse_solvers = {"PEXSI DM", "NTPoly DM", "SIPS eigh", "SIPS DM"}

    for idx, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ls = "--" if label in sparse_solvers else "-"
        ax.loglog(
            xs, ys,
            marker=markers[idx % len(markers)],
            linestyle=ls,
            color=colors[idx % len(colors)],
            label=label,
            linewidth=1.8,
            markersize=6,
        )

    # Reference complexity lines anchored to the first ELPA DM point
    anchor_series = series.get("ELPA DM") or []
    if anchor_series:
        n0, t0 = anchor_series[0]
        ref_n = np.array([SIZES[0], SIZES[-1]], dtype=float)
        ax.loglog(ref_n, t0 * (ref_n / n0) ** 3, "--", color="gray",
                  alpha=0.45, linewidth=1.2, label="O(N³) ref")
        ax.loglog(ref_n, t0 * (ref_n / n0),       ":",  color="gray",
                  alpha=0.45, linewidth=1.2, label="O(N) ref")

    rank_label = f"{n_procs} MPI rank{'s' if n_procs > 1 else ''}"
    ax.set_xlabel("Hamiltonian dimension N", fontsize=12)
    ax.set_ylabel("Wall time (s) — rank 0", fontsize=12)
    ax.set_title(
        f"pyELSI solver scaling  |  {rank_label}  |  N = 100 – 10,000",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scaling_nproc_{n_procs}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[scaling] Plot saved → {out_path}", flush=True)
    assert out_path.is_file(), f"Expected output at {out_path}"
