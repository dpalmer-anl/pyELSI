"""Array-framework abstraction for transparent NumPy / CuPy / PyTorch support.

ELSI's dense C API accepts **host (CPU) pointers only** — there is no
zero-copy device-pointer entry point.  To support "construct on the GPU,
diagonalize on the same GPU" workflows we therefore:

1. Detect the framework and device of the user's input array.
2. Copy it to a host NumPy array for the ELSI call.  ELPA's internal GPU
   path then streams the matrix back onto the originating GPU for the actual
   diagonalization (this is how ELSI/ELPA GPU support is designed to work).
3. Move the results back to the same framework and device as the input, so
   from the caller's perspective the data "stays" on the GPU.

The framework dependencies (``cupy``/``torch``) are imported lazily and only
when an array of that type is actually passed, so importing pyelsi never
requires them.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def _module_root(x: Any) -> str:
    return (type(x).__module__ or "").split(".", 1)[0]


def is_cupy(x: Any) -> bool:
    return _module_root(x) == "cupy"


def is_torch(x: Any) -> bool:
    # torch.Tensor reports module "torch"; this avoids importing torch.
    return _module_root(x) == "torch"


def is_gpu_array(x: Any) -> bool:
    """True if ``x`` is an array that lives in GPU memory."""
    if is_cupy(x):
        return True
    if is_torch(x):
        try:
            return bool(x.is_cuda)
        except Exception:
            return False
    return False


@dataclass(frozen=True)
class ArrayBackend:
    """Captures the framework/device of an input array so results can be restored.

    ``kind`` is one of ``"numpy"``, ``"cupy"``, ``"torch"``.  ``is_gpu`` is True
    when the original array lived in GPU memory.  ``device`` is the
    framework-specific device handle used to restore results and to bind the
    diagonalization to the correct GPU.
    """

    kind: str
    is_gpu: bool
    device: Any = None

    def device_context(self):
        """Context manager that makes this array's GPU the current device.

        Entering it before the ELSI call ensures ELPA initializes CUDA on the
        same GPU that holds the user's data, so the diagonalization runs on the
        expected device.  For CPU inputs this is a no-op.
        """
        if not self.is_gpu:
            return contextlib.nullcontext()
        if self.kind == "cupy":
            # x.device is already a cupy.cuda.Device, which is a context manager.
            return self.device if self.device is not None else contextlib.nullcontext()
        if self.kind == "torch":
            import torch  # type: ignore

            return torch.cuda.device(self.device)
        return contextlib.nullcontext()


def detect_backend(x: Any) -> ArrayBackend:
    """Inspect ``x`` and return how to convert/restore it."""
    if is_cupy(x):
        return ArrayBackend("cupy", True, getattr(x, "device", None))
    if is_torch(x):
        gpu = is_gpu_array(x)
        return ArrayBackend("torch", gpu, getattr(x, "device", None))
    return ArrayBackend("numpy", False, None)


def to_host_numpy(x: Any) -> np.ndarray:
    """Return a host NumPy view/copy of ``x`` regardless of source framework."""
    if is_cupy(x):
        return x.get()
    if is_torch(x):
        return x.detach().cpu().numpy()
    return x


def restore(arr: np.ndarray, backend: ArrayBackend) -> Any:
    """Move a host NumPy result back to ``backend``'s framework and device.

    NumPy inputs return the array unchanged.  ``None`` (e.g. omitted
    eigenvectors) is passed through.
    """
    if arr is None or backend.kind == "numpy":
        return arr
    if backend.kind == "cupy":
        import cupy  # type: ignore

        ctx = backend.device if backend.device is not None else contextlib.nullcontext()
        with ctx:
            return cupy.asarray(arr)
    if backend.kind == "torch":
        import torch  # type: ignore

        # from_numpy requires a contiguous buffer; eigenvectors are Fortran-order.
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if backend.device is not None:
            t = t.to(backend.device)
        return t
    return arr
