class PyElsiError(RuntimeError):
    """Base error for pyELSI."""


class BackendUnavailableError(PyElsiError):
    """Requested backend is not available in this build/runtime."""


class InputValidationError(PyElsiError, ValueError):
    """Invalid inputs were passed to the solver."""

