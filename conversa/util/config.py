"""Configuration management for the conversational teacher."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from conversa.util.io import DEFAULT_SETTINGS_FILE, read_yaml, write_yaml

LEVEL_MAP = {
    "BEGINNER": "A1",
    "INTERMEDIATE": "B1",
    "ADVANCED": "C1",
}

ValidLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]

DEFAULT_CONFIG = {
    "target_language": "es",
    "teacher_language": "en",
    "level": "B1",
}


@dataclass
class Config:
    """Application configuration."""

    target_language: str
    teacher_language: str
    level: ValidLevel

    @classmethod
    def load(cls, settings_path: Path = DEFAULT_SETTINGS_FILE) -> "Config":
        """Load configuration from settings file.

        Creates default settings file if it doesn't exist.

        Args:
            settings_path: Path to settings file.

        Returns:
            Config instance.
        """
        if not settings_path.exists():
            write_yaml(DEFAULT_CONFIG, settings_path)
            print(f"Created settings file: {settings_path}")
            print("Edit the file to customize your settings.")
            print()

        data = read_yaml(settings_path)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary.

        Args:
            data: Dictionary with config values.

        Returns:
            Config instance.
        """
        target_language = data.get("target_language", DEFAULT_CONFIG["target_language"])
        teacher_language = data.get("teacher_language", "")
        level = data.get("level", DEFAULT_CONFIG["level"])

        # Normalize level
        level_upper = str(level).upper()
        if level_upper in LEVEL_MAP:
            level_upper = LEVEL_MAP[level_upper]

        if level_upper not in ("A1", "A2", "B1", "B2", "C1", "C2"):
            print(f"Warning: Invalid level '{level}', using B1")
            level_upper = "B1"

        # Default teacher_language to target_language if empty
        if not teacher_language:
            teacher_language = target_language

        return cls(
            target_language=target_language,
            teacher_language=teacher_language,
            level=level_upper,  # type: ignore
        )

    def print_settings(self) -> None:
        """Print current settings."""
        print(f"Target language: {self.target_language}")
        print(f"Teacher language: {self.teacher_language}")
        print(f"Level: {self.level}")
