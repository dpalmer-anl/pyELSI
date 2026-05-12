# `pyelsi.eigh`

```python
pyelsi.eigh(
    H,
    S=None,
    *,
    solver="auto",
    n_threads=None,
    use_gpu="auto",
    backend_opts=None,
    return_eigenvectors=True,
)
```

## Examples

Standard eigenproblem:

```python
import numpy as np
import pyelsi

rng = np.random.default_rng(0)
A = rng.standard_normal((50, 50))
H = (A + A.T) / 2

w, v = pyelsi.eigh(H)
```

Generalized eigenproblem:

```python
import numpy as np
import pyelsi

rng = np.random.default_rng(0)
A = rng.standard_normal((50, 50))
H = (A + A.T) / 2

B = rng.standard_normal((50, 50))
S = B @ B.T + np.eye(50) * 1e-2

w, v = pyelsi.eigh(H, S=S)
```

