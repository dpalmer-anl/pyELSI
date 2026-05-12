# API overview

The primary user-facing functions are:

- `pyelsi.eigh(H, S=None, **kwargs)` for eigenvalues/eigenvectors
- `pyelsi.density_matrix(H, S=None, n_electrons=..., **kwargs)` for density matrices

All solver settings are intended to be passed via keywords; in v0 the backend is a dense LAPACK reference implementation while the ELSI-backed backend integration is being wired into the build.

