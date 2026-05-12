# Troubleshooting

## Build fails due to missing compilers / toolchains

`pyELSI` builds a compiled extension. You need a working C/C++ toolchain and CMake.

On many Linux systems this means installing packages similar to:
- `gcc`, `g++`, `gfortran`
- `cmake`
- `ninja-build` (optional)

## MPI / CUDA builds

MPI and CUDA are opt-in source builds via `CMAKE_ARGS`:

```bash
CMAKE_ARGS="-DPYELSI_ENABLE_MPI=ON" pip install -v .
CMAKE_ARGS="-DPYELSI_ENABLE_CUDA=ON" pip install -v .
```

If you enable these flags but don’t have the corresponding toolchain (MPI compiler wrappers, CUDA toolkit), the build will fail with a configuration error.

## Build stops while compiling PEXSI (`ppexsi.cpp`, `-Wstringop-overflow`)

On GCC 14 (including conda-forge `compilers`), bundled PEXSI can trigger **false-positive** `-Wstringop-overflow` diagnostics inside inlined libstdc++ code. `pyELSI`’s CMake adds `-Wno-stringop-overflow` (and related flags) for C++ when using GCC.

If you still see failures, try clearing the build tree and reinstalling:

```bash
rm -rf _skbuild
pip install -v -e .
```

You can also pass extra flags explicitly:

```bash
CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-Wno-stringop-overflow" pip install -v -e .
```

## SciPy comparison tests skip

The test suite uses SciPy; install test extras:

```bash
pip install -e ".[test]"
pytest -q
```

## Standard eigenproblem (no overlap matrix)

ELSI’s C API always binds the overlap pointer to a Fortran `(n,n)` array. When you omit `S`, `pyelsi` allocates an identity overlap internally and enables ELSI’s unit-overlap mode (`unit_ovlp`); you must not pass a null overlap buffer through the compiled layer.

