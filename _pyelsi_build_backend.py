"""
Thin build-backend wrapper around scikit_build_core.build that:

  1. Automatically removes the CMake build cache (_skbuild) before each build
     so that stale caches from failed or changed builds never cause failures.

  2. Auto-enables CUDA when the environment signals GPU availability — i.e.
     when the user has loaded a CUDA module (sets CUDA_HOME / CUDA_PATH) or
     has nvcc on PATH.  This means `pip install ".[gpu,mpi]"` Just Works after
     `module load cuda` with no extra flags required.

Configured in pyproject.toml:
    [build-system]
    build-backend = "_pyelsi_build_backend"
    backend-path  = ["."]
"""

import os
import pathlib
import shutil

import scikit_build_core.build as _skb

_BUILD_DIR = pathlib.Path(__file__).parent / "_skbuild"


def _clean_build_dir() -> None:
    if _BUILD_DIR.exists():
        shutil.rmtree(_BUILD_DIR)


def _apply_cuda_auto_detect() -> None:
    """
    Set PYELSI_ENABLE_CUDA=1 when CUDA tools are present in the environment.

    PEP 517 does not give a build backend access to which pip extras were
    selected, so we can't detect `.[gpu]` directly.  Instead we rely on the
    fact that a GPU build only makes sense when CUDA is actually available:
    the user must have run `module load cuda` (or equivalent), which sets
    CUDA_HOME / CUDA_PATH and puts nvcc on PATH.  We treat any of those
    signals as intent to build with GPU support.

    The explicit env var PYELSI_ENABLE_CUDA=0 can suppress this detection.
    """
    explicit = os.environ.get("PYELSI_ENABLE_CUDA", "")
    if explicit:
        return  # honour whatever the user set explicitly

    cuda_available = (
        bool(os.environ.get("CUDA_HOME"))
        or bool(os.environ.get("CUDA_PATH"))
        or bool(os.environ.get("CUDA_ROOT"))
        or bool(shutil.which("nvcc"))
    )
    if cuda_available:
        os.environ["PYELSI_ENABLE_CUDA"] = "1"


# ── PEP 517 hooks ────────────────────────────────────────────────────────────

def get_requires_for_build_wheel(config_settings=None):
    return _skb.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _skb.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _skb.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _apply_cuda_auto_detect()
    _clean_build_dir()
    return _skb.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _apply_cuda_auto_detect()
    _clean_build_dir()
    return _skb.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _apply_cuda_auto_detect()
    # prepare_metadata_for_build_wheel already cleaned if it was called; if pip
    # skipped that step and called build_wheel directly, clean here instead.
    if _BUILD_DIR.exists():
        _clean_build_dir()
    return _skb.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _apply_cuda_auto_detect()
    if _BUILD_DIR.exists():
        _clean_build_dir()
    return _skb.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    return _skb.build_sdist(sdist_directory, config_settings)
