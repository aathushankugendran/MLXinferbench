# src/utils/memory.py

"""
Memory + run status utilities.

Right now this is a thin wrapper to record whether a benchmark run
succeeded or failed, with an optional error message.

Later you can extend this with:
  * peak CPU/RAM usage
  * MLX memory stats (if exposed)
  * OOM detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class RunStatus:
    """
    Simple status record for a single benchmark run.
    """
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "error_message": self.error_message or "",
        }


def record_success() -> RunStatus:
    """
    Convenience helper for a successful run.
    """
    return RunStatus(success=True, error_message=None)


def record_failure(error: Exception | str) -> RunStatus:
    """
    Convenience helper for a failed run.

    Args:
        error: Exception or string describing the failure.
    """
    msg = str(error)
    return RunStatus(success=False, error_message=msg)


def run_with_status(fn: Callable[..., Any], *args, **kwargs) -> tuple[Optional[Any], RunStatus]:
    """
    Run `fn(*args, **kwargs)` and wrap the result in a RunStatus object.

    Returns:
        (result, status)
        - result: function return value, or None if an exception occurred
        - status: RunStatus(success=..., error_message=...)
    """
    try:
        result = fn(*args, **kwargs)
        return result, record_success()
    except Exception as exc:  # noqa: BLE001 for simplicity here
        return None, record_failure(exc)
