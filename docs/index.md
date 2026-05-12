# pyELSI

`pyELSI` is a Python interface to [ELSI (ELectronic Structure Infrastructure)](https://wordpress.elsi-interchange.org/).

The core goal is to let you pass Hamiltonians (and optional overlap matrices) as NumPy arrays (and, later, CSR sparse matrices) and use ELSI backends to compute:

- eigenvalues / eigenvectors
- density matrices

At a high level:

```python
import numpy as np
import pyelsi

rng = np.random.default_rng(0)
A = rng.standard_normal((100, 100))
H = (A + A.T) / 2

w, v = pyelsi.eigh(H)
```

See **Installation** and the API pages for details.

