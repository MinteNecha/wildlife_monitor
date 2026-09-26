"""
Configuration validation (Package P4, FR9).

Every settings change is checked against a declared range before it is
applied. The UC8 alternative flow requires that an out-of-range value is
rejected with the valid range shown alongside it, so each rule carries a
human-readable description of what it will accept.

Rules are declared as data rather than as a chain of if-statements, which
means the Settings page can render the valid range for a field without
knowing anything about the rule that produces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from wildlife_monitor.config.settings import DEFAULT_TOP_N, TARGET_SPECIES
from wildlife_monitor.utils.validation import ValidationResult


@dataclass(frozen=True)
class Rule:
    """One setting's constraint, plus how to describe it to a person."""

    check: Callable[[Any], bool]
    describe: str
    bounds: tuple[Any, Any] | None = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _in_range(low: float, high: float) -> Callable[[Any], bool]:
    return lambda value: _is_number(value) and low <= float(value) <= high


def _one_of(options: tuple[str, ...]) -> Callable[[Any], bool]:
    return lambda value: str(value).strip().lower() in {
        option.lower() for option in options}


def _category_list(value: Any) -> bool:
    """A comma-separated list naming at least two distinct categories."""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return len(parts) >= 2 and len(set(parts)) == len(parts)


PIPELINES = ("bioclip_sam", "bioclip_yolo", "bioclip_megadetector")
ARCHITECTURES = ("LSTM", "Transformer")
DEVICES = ("cuda", "cpu")

RULES: dict[str, Rule] = {
    "pipeline": Rule(_one_of(PIPELINES),
                      "one of: " + ", ".join(PIPELINES)),
    "threshold": Rule(_in_range(0.0, 1.0),
                       "a number between 0.0 and 1.0", (0.0, 1.0)),
    "confidence_threshold": Rule(_in_range(0.0, 1.0),
                                  "a number between 0.0 and 1.0", (0.0, 1.0)),
    "top_n": Rule(_in_range(1, 5000),
                   "a whole number between 1 and 5000", (1, 5000)),
    "arch": Rule(_one_of(ARCHITECTURES),
                  "one of: " + ", ".join(ARCHITECTURES)),
    "device": Rule(_one_of(DEVICES), "one of: " + ", ".join(DEVICES)),
    "memory": Rule(_in_range(2, 256),
                    "a whole number of GB between 2 and 256", (2, 256)),
    "max_length": Rule(_in_range(5, 500),
                        "a whole number between 5 and 500", (5, 500)),
    "epochs": Rule(_in_range(1, 5000),
                    "a whole number between 1 and 5000", (1, 5000)),
    "learning_rate": Rule(_in_range(1e-6, 1.0),
                           "a number between 0.000001 and 1.0", (1e-6, 1.0)),
    "species": Rule(_one_of(tuple(TARGET_SPECIES)),
                     "one of the configured target species"),
    "activity": Rule(_category_list,
                      "at least two distinct comma-separated categories"),
    "movement": Rule(_category_list,
                      "at least two distinct comma-separated categories"),
    "social": Rule(_category_list,
                    "at least two distinct comma-separated categories"),
}

DEFAULTS: dict[str, Any] = {
    "pipeline": "bioclip_megadetector",
    "threshold": 0.0,
    "top_n": DEFAULT_TOP_N,
    "arch": "LSTM",
    "device": "cpu",
    "memory": 8,
    "max_length": 40,
}


class ConfigValidator:
    """Validates proposed configuration changes before they are applied."""

    def validate(self, key: str, value: Any) -> ValidationResult:
        """Check one setting, naming the valid range when it fails."""
        rule = RULES.get(key)
        if rule is None:
            return ValidationResult.ok(key=key, unchecked=True)
        if rule.check(value):
            return ValidationResult.ok(key=key, value=value)
        return ValidationResult.fail(
            f"must be {rule.describe}", key=key, value=value,
            expected=rule.describe, bounds=rule.bounds)

    def validate_all(self, pending: dict[str, Any]) -> list[ValidationResult]:
        """Validate a whole settings form, returning only the failures."""
        return [result for result in
                (self.validate(key, value) for key, value in pending.items())
                if not result.valid]

    def valid_range(self, key: str) -> tuple[Any, Any] | None:
        """Numeric bounds for a setting, or None when it is not numeric."""
        rule = RULES.get(key)
        return rule.bounds if rule else None

    def describe(self, key: str) -> str:
        """Human-readable description of what a setting accepts."""
        rule = RULES.get(key)
        return rule.describe if rule else "no constraint"
