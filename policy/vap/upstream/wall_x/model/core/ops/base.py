"""Base class for operator proxies with lazy backend resolution."""

import importlib
import logging
import os
import threading

logger = logging.getLogger(__name__)


class OpsProxy:
    """Callable proxy that lazily resolves to the best available backend.

    Three-level fallback:
      Level 1: xcompute (fastest, requires xcompute package)
      Level 2: CUDA inline kernel (JIT compiled, requires CUDA)
      Level 3: Pure PyTorch (always available)

    Subclasses define _xcompute_name, _pytorch_fallback, and optionally _cuda_kernel.

    Set _xcompute_name to None to skip xcompute resolution (e.g., when the
    xcompute API is incompatible).
    """

    def __init__(self):
        self._resolved_fn = None
        self._backend = None
        self._resolve_lock = threading.Lock()

    def _resolve(self):
        """Resolve the best available backend. Called once on first use (thread-safe)."""
        if self._resolved_fn is not None:
            return
        with self._resolve_lock:
            # Double-check after acquiring lock
            if self._resolved_fn is not None:
                return

            mode = os.environ.get("WALLX_OPS_BACKEND", "auto")
            if mode not in ("auto", "pytorch"):
                raise ValueError("WALLX_OPS_BACKEND must be auto or pytorch")
            if mode == "pytorch":
                # Explicit deployment diagnostic path; skip BOTH native/JIT
                # probes. Never retry after a failed CUDA launch in-process.
                self._backend = "pytorch"
                self._resolved_fn = self._pytorch_fallback
                logger.info("%s: forced PyTorch backend", self.__class__.__name__)
                return

            # Level 1: try xcompute (skip if _xcompute_name is None)
            fn = self._probe_xcompute()
            if fn is not None:
                self._backend = "xcompute"
                self._resolved_fn = fn
                return

            # Level 2: try CUDA inline kernel (subclass override)
            cuda_fn = self._get_cuda_kernel()
            if cuda_fn is not None:
                self._backend = "cuda_inline"
                self._resolved_fn = cuda_fn
                return

            # Level 3: PyTorch fallback
            self._backend = "pytorch"
            self._resolved_fn = self._pytorch_fallback
            logger.info(f"{self.__class__.__name__}: using PyTorch fallback")

    def _get_cuda_kernel(self):
        """Override in subclass to provide Level 2 CUDA kernel. Returns None if unavailable."""
        return None

    @property
    def _xcompute_name(self):
        """Return xcompute attribute name, or None to skip xcompute resolution.

        Base returns None (skip xcompute). Subclasses override to return the
        attribute name when xcompute integration is desired.
        """
        return None

    def _pytorch_fallback(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} has no PyTorch fallback")

    def __call__(self, *args, **kwargs):
        if self._resolved_fn is None:
            self._resolve()
        return self._resolved_fn(*args, **kwargs)

    @property
    def backend(self) -> str:
        if self._resolved_fn is None:
            self._resolve()
        return self._backend

    # ------------------------------------------------------------------
    # Multi-backend API (for testing / benchmarking)
    # ------------------------------------------------------------------

    def _probe_xcompute(self):
        """Return the xcompute function for this op, or None if unavailable.

        Capability-based: used iff the loaded xcompute build provides it,
        otherwise the PyTorch fallback runs.
        """
        if self._xcompute_name is None:
            return None
        try:
            mod = importlib.import_module("xcompute")
            return getattr(mod, self._xcompute_name)
        except (ImportError, AttributeError):
            return None

    def has_xcompute(self) -> bool:
        """Public capability query: does the loaded xcompute build provide this op?"""
        return self._probe_xcompute() is not None

    def available_backends(self):
        """Return list of available backend names for this operator."""
        backends = ["pytorch"]
        if self._probe_xcompute() is not None:
            backends.append("xcompute")
        if self._get_cuda_kernel() is not None:
            backends.append("cuda_inline")
        return backends

    def _get_backend_fn(self, backend):
        """Return the callable for a specific backend, or None if unavailable."""
        if backend == "pytorch":
            return self._pytorch_fallback
        elif backend == "xcompute":
            return self._probe_xcompute()
        elif backend == "cuda_inline":
            return self._get_cuda_kernel()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def call_with_backend(self, backend, *args, **kwargs):
        """Call this operator using a specific backend.

        This does NOT change the default backend used by ``__call__``.
        The ``backend`` property still reflects the auto-resolved default.

        Args:
            backend: One of "xcompute", "cuda_inline", "pytorch".

        Returns:
            Operator output from the specified backend.

        Raises:
            RuntimeError: If the requested backend is not available.
        """
        fn = self._get_backend_fn(backend)
        if fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: backend '{backend}' not available. "
                f"Available: {self.available_backends()}"
            )
        return fn(*args, **kwargs)

    def __repr__(self):
        backend = self._backend or "unresolved"
        return f"<{self.__class__.__name__} backend={backend}>"
