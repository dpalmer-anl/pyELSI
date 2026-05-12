# Installation

## CPU install (default)

```bash
pip install pyELSI
```

This builds a local wheel using a CMake/pybind11 extension.

## Source build options (clusters)

### MPI build

```bash
CMAKE_ARGS="-DPYELSI_ENABLE_MPI=ON" pip install -v pyELSI
```

### CUDA build

```bash
CMAKE_ARGS="-DPYELSI_ENABLE_CUDA=ON" pip install -v pyELSI
```

## Vendoring ELSI

If you want to vendor the ELSI source directly (instead of fetching it during build), clone it into:

`third_party/elsi/elsi-interface/`

and install with:

```bash
CMAKE_ARGS="-DPYELSI_FETCH_ELSI=OFF" pip install -v .
```

