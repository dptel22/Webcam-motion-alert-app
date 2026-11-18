"""Deep Learning object detection and appearance Re-ID modules."""

import os
import threading

_torch_threads_lock = threading.Lock()
_torch_threads_configured = False


def configure_torch_threads() -> None:
    """Set torch intra-op threads once (half the logical cores, at least 1).

    Imported lazily so CVIP-only runs never import torch at all.
    """
    global _torch_threads_configured
    with _torch_threads_lock:
        if _torch_threads_configured:
            return
        _torch_threads_configured = True
        try:
            import torch

            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:  # pragma: no cover - torch missing or already in parallel use
            pass
