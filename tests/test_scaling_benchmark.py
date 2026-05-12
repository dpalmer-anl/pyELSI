"""
Wall-clock scaling benchmark for pyELSI solvers.

Times each solver over 15 log-spaced matrix sizes from N=10 to N=1000.
Generates one plot per MPI-rank count in outputs/scaling_nproc_{N}.png.

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

def _sym_dense(rng: np.random.Generator, n: int, diag_shift: float = 5.0) -> np.ndarray:
    a = rng.standard_normal((n, n))
    a = 0.5 * (a + a.T) + diag_shift * np.eye(n)
    return a


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
# Per-solver timers (return None on any exception)
# ---------------------------------------------------------------------------

def _time_eigh(n: int, n_procs: int, rng: np.random.Generator) -> float | None:
    import pyelsi  # already guaranteed by importorskip; kept for clarity
    H = _sym_dense(rng, n)
    opts = {"force_single_proc": 1} if n_procs > 1 else {}
    _barrier()
    t0 = time.perf_counter()
    try:
        pyelsi.eigh(H, solver="elpa", backend_opts=opts)
    except Exception:
        return None
    return time.perf_counter() - t0


def _time_dense_dm(solver: str, n: int, n_procs: int, rng: np.random.Generator) -> float | None:
    import pyelsi
    H = _sym_dense(rng, n)
    ne = _ne(n)
    opts = {"force_single_proc": 1} if n_procs > 1 else {}
    _barrier()
    t0 = time.perf_counter()
    try:
        pyelsi.density_matrix(H, n_electrons=ne, solver=solver, backend_opts=opts)
    except Exception:
        return None
    return time.perf_counter() - t0


def _time_sparse_dm(solver: str, n: int, n_procs: int, rng: np.random.Generator) -> float | None:
    import pyelsi
    H = _banded_csr(rng, n)
    ne = _ne(n)
    opts = {"force_single_proc": 1} if n_procs > 1 else {}
    _barrier()
    t0 = time.perf_counter()
    try:
        pyelsi.density_matrix(H, n_electrons=ne, solver=solver, backend_opts=opts)
    except Exception:
        return None
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.scaling_benchmark
def test_scaling_benchmark():
    """Benchmark all solvers for N=100…10,000; save scaling plot to outputs/."""
    if os.environ.get("PYELSI_RUN_SCALING_BENCH") != "1":
        pytest.skip("Set PYELSI_RUN_SCALING_BENCH=1 to run this benchmark.")

    rank, n_procs = _rank_size()

    # (label, callable(n, n_procs, rng) → float|None)
    SOLVERS = [
        ("ELPA eigh",   lambda n, k, r: _time_eigh(n, k, r)),
        ("ELPA DM",     lambda n, k, r: _time_dense_dm("elpa",   n, k, r)),
        ("OMM DM",      lambda n, k, r: _time_dense_dm("omm",    n, k, r)),
        ("PEXSI DM",    lambda n, k, r: _time_sparse_dm("pexsi",  n, k, r)),
        ("NTPoly DM",   lambda n, k, r: _time_sparse_dm("ntpoly", n, k, r)),
    ]

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
                    print(f"    {label:<12s}: {dt:.4f} s", flush=True)

    # Only rank 0 writes the plot
    if rank != 0:
        return

    assert any(pts for pts in series.values()), (
        "No solver produced any timing data — check your build."
    )

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v"]

    for idx, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.loglog(
            xs, ys,
            marker=markers[idx % len(markers)],
            linestyle="-",
            color=colors[idx % len(colors)],
            label=label,
            linewidth=1.8,
            markersize=6,
        )

    # Reference complexity lines anchored to the first ELPA DM or eigh point
    anchor_series = series.get("ELPA DM") or series.get("ELPA eigh") or []
    if anchor_series:
        n0, t0 = anchor_series[0]
        ref_n = np.array([SIZES[0], SIZES[-1]], dtype=float)
        ax.loglog(ref_n, t0 * (ref_n / n0) ** 3, "--", color="gray",
                  alpha=0.55, linewidth=1.2, label="O(N³) ref")
        ax.loglog(ref_n, t0 * (ref_n / n0),       ":",  color="gray",
                  alpha=0.55, linewidth=1.2, label="O(N) ref")

    rank_label = f"{n_procs} MPI rank{'s' if n_procs > 1 else ''}"
    ax.set_xlabel("Hamiltonian dimension N", fontsize=12)
    ax.set_ylabel("Wall time (s) — rank 0", fontsize=12)
    ax.set_title(
        f"pyELSI solver scaling  |  {rank_label}  |  N = 100 – 10,000",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scaling_nproc_{n_procs}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[scaling] Plot saved → {out_path}", flush=True)
    assert out_path.is_file(), f"Expected output at {out_path}"
