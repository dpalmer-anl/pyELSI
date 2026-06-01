"""
Thin build-backend wrapper around scikit_build_core.build that:

  1. Automatically removes the CMake build cache (_skbuild) before each build
     so that stale caches from failed or changed builds never cause failures.

  2. Auto-enables CUDA when the environment signals GPU availability — i.e.
     when the user has loaded a CUDA module (sets CUDA_HOME / CUDA_PATH / CUDA_ROOT)
     or has nvcc on PATH.  This means `pip install ".[gpu,mpi]"` Just Works after
     `module load cuda` with no extra flags required.

     The detection is done by injecting `cmake.define.PYELSI_ENABLE_CUDA = "ON"`
     directly into the config_settings dict passed to scikit-build-core.  This
     avoids the [[tool.scikit-build.overrides]] mechanism, which replaces rather
     than merges the cmake.define table and would silently drop other defines
     (e.g. PYELSI_FETCH_ELSI) that default to OFF in CMakeLists.txt.

     Override behaviour:
       PYELSI_ENABLE_CUDA=0  →  always CPU-only, suppresses auto-detect
       PYELSI_ENABLE_CUDA=1  →  always CUDA, bypasses auto-detect
       (unset) + CUDA in env  →  CUDA enabled automatically
       (unset) + no CUDA      →  CPU-only

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_build_dir() -> None:
    if _BUILD_DIR.exists():
        shutil.rmtree(_BUILD_DIR)


def _cuda_is_available() -> bool:
    """
    Return True when a CUDA build should be performed.

    Checks (in order):
      • PYELSI_ENABLE_CUDA=0 → always False
      • PYELSI_ENABLE_CUDA=1 → always True
      • CUDA_HOME / CUDA_PATH / CUDA_ROOT set (by 'module load cuda')
      • nvcc found on PATH
    """
    explicit = os.environ.get("PYELSI_ENABLE_CUDA", "")
    if explicit == "0":
        return False
    if explicit == "1":
        return True
    return (
        bool(os.environ.get("CUDA_HOME"))
        or bool(os.environ.get("CUDA_PATH"))
        or bool(os.environ.get("CUDA_ROOT"))
        or bool(shutil.which("nvcc"))
    )


def _augment_config_settings(config_settings):
    """
    Inject cmake.define.PYELSI_ENABLE_CUDA into config_settings when CUDA is
    available.  Using config_settings (the PEP 517 API) guarantees that ALL
    base cmake.define entries from pyproject.toml are preserved alongside the
    injected flag — unlike scikit-build-core overrides, which replace the dict.
    """
    if not _cuda_is_available():
        return config_settings
    result = dict(config_settings) if config_settings else {}
    result["cmake.define.PYELSI_ENABLE_CUDA"] = "ON"
    return result


# ── PEP 517 hooks ─────────────────────────────────────────────────────────────

def get_requires_for_build_wheel(config_settings=None):
    return _skb.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _skb.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _skb.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _clean_build_dir()
    return _skb.prepare_metadata_for_build_wheel(
        metadata_directory, _augment_config_settings(config_settings)
    )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _clean_build_dir()
    return _skb.prepare_metadata_for_build_editable(
        metadata_directory, _augment_config_settings(config_settings)
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    # prepare_metadata_for_build_wheel already cleaned if it was called; if pip
    # skipped that step and called build_wheel directly, clean here instead.
    if _BUILD_DIR.exists():
        _clean_build_dir()
    return _skb.build_wheel(
        wheel_directory, _augment_config_settings(config_settings), metadata_directory
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    if _BUILD_DIR.exists():
        _clean_build_dir()
    return _skb.build_editable(
        wheel_directory, _augment_config_settings(config_settings), metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    return _skb.build_sdist(sdist_directory, config_settings)
