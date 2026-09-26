"""
The shared validation result type.

Both :class:`~wildlife_monitor.data.validator.ImageValidator` (FR1) and
:class:`~wildlife_monitor.config.validator.ConfigValidator` (FR9) answer the
same shape of question — is this acceptable, and if not, why — so they return
the same object rather than each inventing a convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """The outcome of validating one thing."""

    valid: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid

    @classmethod
    def ok(cls, **details: Any) -> "ValidationResult":
        return cls(valid=True, details=details)

    @classmethod
    def fail(cls, reason: str, **details: Any) -> "ValidationResult":
        return cls(valid=False, reason=reason, details=details)
