"""Tests for Config class."""

from conversa.util.config import DEFAULT_CONFIG, Config
from conversa.util.io import write_yaml


def test_config_load_creates_default_file(tmp_path, capsys):
    """Test that Config.load creates default settings file if not exists."""
    settings_path = tmp_path / "settings.yaml"

    config = Config.load(settings_path)

    assert settings_path.exists()
    assert config.target_language == DEFAULT_CONFIG["target_language"]
    assert config.teacher_language == DEFAULT_CONFIG["teacher_language"]
    assert config.level == DEFAULT_CONFIG["level"]

    captured = capsys.readouterr()
    assert "Created settings file" in captured.out


def test_config_load_reads_existing_file(tmp_path):
    """Test that Config.load reads existing settings file."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml(
        {"target_language": "de", "teacher_language": "en", "level": "A2"},
        settings_path,
    )

    config = Config.load(settings_path)

    assert config.target_language == "de"
    assert config.teacher_language == "en"
    assert config.level == "A2"


def test_config_level_normalization(tmp_path):
    """Test that level is normalized to uppercase."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml({"target_language": "es", "level": "b2"}, settings_path)

    config = Config.load(settings_path)

    assert config.level == "B2"


def test_config_level_beginner_intermediate_advanced(tmp_path):
    """Test that beginner/intermediate/advanced are mapped to CEFR levels."""
    settings_path = tmp_path / "settings.yaml"

    # Test beginner -> A1
    write_yaml({"target_language": "es", "level": "beginner"}, settings_path)
    config = Config.load(settings_path)
    assert config.level == "A1"

    # Test intermediate -> B1
    write_yaml({"target_language": "es", "level": "intermediate"}, settings_path)
    config = Config.load(settings_path)
    assert config.level == "B1"

    # Test advanced -> C1
    write_yaml({"target_language": "es", "level": "advanced"}, settings_path)
    config = Config.load(settings_path)
    assert config.level == "C1"


def test_config_invalid_level_defaults_to_b1(tmp_path, capsys):
    """Test that invalid level defaults to B1 with warning."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml({"target_language": "es", "level": "invalid"}, settings_path)

    config = Config.load(settings_path)

    assert config.level == "B1"
    captured = capsys.readouterr()
    assert "Invalid level" in captured.out


def test_config_teacher_language_defaults_to_target(tmp_path):
    """Test that teacher_language defaults to target_language if empty."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml(
        {"target_language": "fr", "teacher_language": "", "level": "B1"}, settings_path
    )

    config = Config.load(settings_path)

    assert config.target_language == "fr"
    assert config.teacher_language == "fr"


def test_config_teacher_language_defaults_when_missing(tmp_path):
    """Test that teacher_language defaults to target_language if not specified."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml({"target_language": "it", "level": "B1"}, settings_path)

    config = Config.load(settings_path)

    assert config.target_language == "it"
    assert config.teacher_language == "it"


def test_config_invalid_target_language(tmp_path):
    """Test that invalid target_language raises ValueError."""
    import pytest

    settings_path = tmp_path / "settings.yaml"
    write_yaml({"target_language": "invalid", "level": "B1"}, settings_path)

    with pytest.raises(ValueError, match="Unknown target_language"):
        Config.load(settings_path)


def test_config_invalid_teacher_language(tmp_path):
    """Test that invalid teacher_language raises ValueError."""
    import pytest

    settings_path = tmp_path / "settings.yaml"
    write_yaml(
        {"target_language": "es", "teacher_language": "invalid", "level": "B1"},
        settings_path,
    )

    with pytest.raises(ValueError, match="Unknown teacher_language"):
        Config.load(settings_path)


def test_config_language_name_property(tmp_path):
    """Test target_language_name and teacher_language_name properties."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml(
        {"target_language": "es", "teacher_language": "en", "level": "B1"},
        settings_path,
    )

    config = Config.load(settings_path)

    assert config.target_language_name == "Spanish"
    assert config.teacher_language_name == "English"


def test_config_print_settings(tmp_path, capsys):
    """Test print_settings output."""
    settings_path = tmp_path / "settings.yaml"
    write_yaml(
        {"target_language": "ja", "teacher_language": "en", "level": "A1"},
        settings_path,
    )

    config = Config.load(settings_path)
    config.print_settings()

    captured = capsys.readouterr()
    assert "Target language: Japanese (ja)" in captured.out
    assert "Teacher language: English (en)" in captured.out
    assert "Level: A1" in captured.out
