from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


@dataclass(frozen=True)
class RuntimeInfo:
    n_threads: int
    has_mpi: bool
    has_cuda: bool
    mpi_rank: int | None = None
    mpi_size: int | None = None


def detect_runtime(*, n_threads: int | None = None) -> RuntimeInfo:
    omp_threads = _env_int("OMP_NUM_THREADS")
    n = n_threads or omp_threads or (os.cpu_count() or 1)

    # These are compile-time constants reported by the extension.
    from ._core import build_info

    info = build_info()
    has_mpi = bool(info.get("has_mpi", False))
    has_cuda = bool(info.get("has_cuda", False))

    mpi_rank = None
    mpi_size = None
    if has_mpi:
        try:
            from mpi4py import MPI  # type: ignore

            comm = MPI.COMM_WORLD
            mpi_rank = int(comm.Get_rank())
            mpi_size = int(comm.Get_size())
        except Exception:
            # MPI build without mpi4py in runtime env: leave as unknown.
            pass

    return RuntimeInfo(n_threads=int(n), has_mpi=has_mpi, has_cuda=has_cuda, mpi_rank=mpi_rank, mpi_size=mpi_size)

