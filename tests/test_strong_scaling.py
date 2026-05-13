"""
Strong-scaling (fixed-N) benchmark for pyELSI density-matrix solvers.

Fixes the Hamiltonian at N=4000 and measures wall time with 1, 2, and 4 MPI
processes.  Run once per process count; after all three counts have been run the
script automatically writes a combined plot showing time vs. n_procs for each
solver, together with ideal (T₁ / P) reference lines.

How to run
----------
Serial (1 rank):
    PYELSI_RUN_SCALING_BENCH=1 python -m pytest tests/test_strong_scaling.py -v -s

2 ranks:
    PYELSI_RUN_SCALING_BENCH=1 mpirun -n 2 python -m pytest tests/test_strong_scaling.py -v -s

4 ranks (combined plot written here once 1- and 2-rank results exist):
    PYELSI_RUN_SCALING_BENCH=1 mpirun -n 4 python -m pytest tests/test_strong_scaling.py -v -s

Timings are cached in outputs/strong_scaling_{P}proc.json so you can re-run a
single process count without re-timing the others.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pyelsi


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIXED_N = 4_000
TARGET_NPROCS = [1, 2, 4]


# ---------------------------------------------------------------------------
# MPI helpers
# ---------------------------------------------------------------------------

def _mpi_rank_size() -> tuple[int, int]:
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except ImportError:
        return 0, 1


def _mpi_barrier() -> None:
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Matrix factories  (fixed seed → all ranks build the same matrix)
# ---------------------------------------------------------------------------

def _sym_dense(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((n, n))
    return 0.5 * (a + a.T) + float(n) * np.eye(n)


def _banded_csr(n: int):
    rng = np.random.default_rng(42)
    main = rng.standard_normal(n) + 5.0
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
# Solver list  (label, matrix kind, pyelsi solver name)
# ---------------------------------------------------------------------------

SOLVERS = [
    ("ELPA DM",   "dense",  "elpa"),
    ("OMM DM",    "dense",  "omm"),
    ("PEXSI DM",  "sparse", "pexsi"),
    ("NTPoly DM", "sparse", "ntpoly"),
]


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def _time_solver(kind: str, solver: str, n: int) -> float | None:
    """
    Time one pyelsi.density_matrix() call.  All ranks participate in the
    collective MPI computation; barriers before/after ensure the measured wall
    time is the true parallel duration (not just rank-0 latency).
    Returns elapsed seconds on rank 0, or None on error / unavailable solver.
    """
    ne = _ne(n)
    if kind == "dense":
        H = _sym_dense(n)
    else:
        if solver == "sips" and not pyelsi.build_info().get("has_sips", False):
            return None
        H = _banded_csr(n)

    try:
        _mpi_barrier()                                  # sync all ranks before start
        t0 = time.perf_counter()
        pyelsi.density_matrix(H, n_electrons=float(ne), solver=solver)
        _mpi_barrier()                                  # sync all ranks after finish
        return time.perf_counter() - t0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_combined_plot(
    all_timings: dict[int, dict[str, float | None]],
    out_path: Path,
    n: int,
) -> None:
    """
    Strong-scaling plot:  x = n_procs,  y = wall time (s).
    Solid/dashed lines = measured times.
    Faint dotted lines of the same colour = ideal T₁/P reference.
    """
    nprocs_sorted = sorted(all_timings.keys())
    colors  = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted: list[tuple[str, int]] = []   # (label, color_idx) for legend

    for idx, (label, kind, _) in enumerate(SOLVERS):
        xs, ys = [], []
        for p in nprocs_sorted:
            t = all_timings[p].get(label)
            if t is not None:
                xs.append(p)
                ys.append(t)
        if not xs:
            continue

        c = colors[idx % len(colors)]
        ls = "--" if kind == "sparse" else "-"

        # Measured times
        ax.loglog(xs, ys,
                  marker=markers[idx % len(markers)],
                  linestyle=ls,
                  color=c,
                  linewidth=1.8,
                  markersize=8,
                  label=label)

        # Ideal reference: T₁/P  (dotted, same colour, no label — legend entry added below)
        t1 = all_timings.get(1, {}).get(label)
        if t1 is not None:
            ref_p = np.array(nprocs_sorted, dtype=float)
            ax.loglog(ref_p, t1 / ref_p,
                      linestyle=":",
                      color=c,
                      linewidth=1.0,
                      alpha=0.45)

        plotted.append((label, idx))

    # Manual x-axis ticks
    ax.set_xticks(nprocs_sorted)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel("Number of MPI processes", fontsize=12)
    ax.set_ylabel(f"Wall time (s)  |  N = {n:,}", fontsize=12)
    ax.set_title(
        f"pyELSI strong scaling  |  density matrix  |  N = {n:,}",
        fontsize=12,
    )
    ax.grid(True, which="both", alpha=0.3)

    # Append a single legend entry for the ideal-scaling reference
    handles, labels = ax.get_legend_handles_labels()
    ideal_handle = Line2D([0], [0], linestyle=":", color="gray",
                          linewidth=1.0, alpha=0.6)
    ax.legend(handles + [ideal_handle],
              labels   + ["ideal (T₁/P)"],
              loc="upper right", fontsize=9, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.scaling_benchmark
def test_strong_scaling():
    """
    Fixed-N strong-scaling benchmark.  Run with mpirun -n 1/2/4 to populate
    timing data; combined plot is written once ≥ 2 process counts are present.
    """
    if os.environ.get("PYELSI_RUN_SCALING_BENCH") != "1":
        pytest.skip("Set PYELSI_RUN_SCALING_BENCH=1 to run this benchmark.")

    rank, n_procs = _mpi_rank_size()

    if rank == 0:
        print(f"\n[strong scaling]  N = {FIXED_N:,}   n_procs = {n_procs}", flush=True)

    timings: dict[str, float | None] = {}

    for label, kind, solver in SOLVERS:
        _mpi_barrier()
        if rank == 0:
            print(f"  {label} ...", end="  ", flush=True)
        dt = _time_solver(kind, solver, FIXED_N)
        timings[label] = dt
        if rank == 0:
            msg = f"{dt:.4f} s" if dt is not None else "FAILED / unavailable"
            print(msg, flush=True)

    # Only rank 0 handles I/O from here on
    if rank != 0:
        return

    assert any(t is not None for t in timings.values()), (
        f"All solvers failed for N={FIXED_N} with {n_procs} ranks — check your build."
    )

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist timings for this process count
    json_path = out_dir / f"strong_scaling_{n_procs}proc.json"
    with open(json_path, "w") as fh:
        json.dump(
            {"n_procs": n_procs, "n": FIXED_N, "timings": timings},
            fh, indent=2,
        )
    print(f"  Timings saved → {json_path}", flush=True)

    # Accumulate results from all available process counts
    all_timings: dict[int, dict[str, float | None]] = {}
    for p in TARGET_NPROCS:
        path = out_dir / f"strong_scaling_{p}proc.json"
        if path.exists():
            with open(path) as fh:
                data = json.load(fh)
            all_timings[p] = data["timings"]

    if len(all_timings) < 2:
        print(
            f"  Run with other process counts to generate the combined plot "
            f"(have: {sorted(all_timings.keys())}, need: {TARGET_NPROCS}).",
            flush=True,
        )
        return

    plot_path = out_dir / "strong_scaling.png"
    _make_combined_plot(all_timings, plot_path, FIXED_N)
    print(f"\n[strong scaling]  Combined plot saved → {plot_path}", flush=True)
    assert plot_path.is_file(), f"Expected plot at {plot_path}"
