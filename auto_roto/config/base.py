"""
Base Configuration
==================

Base class and utilities for configuration dataclasses.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class BaseConfig:
    """
    Base configuration class with common functionality.

    All configuration classes should inherit from this.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: str) -> "BaseConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def validate(self) -> None:
        """
        Validate configuration values.

        Override in subclasses to add specific validation.
        Raises ValueError if validation fails.
        """
        pass

    def update(self, **kwargs) -> "BaseConfig":
        """
        Create a new config with updated values.

        Args:
            **kwargs: Fields to update

        Returns:
            New configuration instance
        """
        current = self.to_dict()
        current.update(kwargs)
        return self.__class__(**current)


def validate_path_exists(path: str, name: str = "path") -> None:
    """
    Validate that a path exists.

    Args:
        path: Path to validate
        name: Name for error message

    Raises:
        ValueError: If path doesn't exist
    """
    if path and not Path(path).exists():
        raise ValueError(f"{name} does not exist: {path}")


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> None:
    """
    Validate that a value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name for error message

    Raises:
        ValueError: If value out of range
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_choice(
    value: Any,
    choices: list,
    name: str = "value"
) -> None:
    """
    Validate that a value is one of the allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed values
        name: Name for error message

    Raises:
        ValueError: If value not in choices
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")
