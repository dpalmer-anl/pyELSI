from __future__ import annotations

from typing import Any, Literal, Mapping, TypedDict

import numpy as np

ArrayLike = np.ndarray

UseGpu = Literal["auto", True, False]


class BackendOptions(TypedDict, total=False):
    # Placeholder for solver-specific keywords.
    # In v0 we accept arbitrary keys but validate known ones when integrated with ELSI.
    dummy: Any


BackendOpts = Mapping[str, Any] | None

