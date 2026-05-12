# `pyelsi.density_matrix`

```python
pyelsi.density_matrix(
    H,
    S=None,
    *,
    n_electrons,
    solver="auto",
    n_threads=None,
    use_gpu="auto",
    backend_opts=None,
)
```

## Example

```python
import numpy as np
import pyelsi

rng = np.random.default_rng(0)
A = rng.standard_normal((80, 80))
H = (A + A.T) / 2

D = pyelsi.density_matrix(H, n_electrons=40)
```

