import json
from pathlib import Path
from typing import Any

import appdirs
import yaml

APP_NAME = "conversa"
DEFAULT_DATA_DIR = Path(appdirs.user_data_dir(APP_NAME))
DEFAULT_MISTAKES_FILE = DEFAULT_DATA_DIR / "mistakes.jsonl"
DEFAULT_CONVERSATIONS_FILE = DEFAULT_DATA_DIR / "conversations.jsonl"
DEFAULT_AUDIO_DIR = DEFAULT_DATA_DIR / "audios"
DEFAULT_READING_STATUS = DEFAULT_DATA_DIR / "reading_status.json"
DEFAULT_SETTINGS_FILE = DEFAULT_DATA_DIR / "settings.yaml"


def append_to_jsonl_file(data: dict, fpath: Path) -> None:
    """Append a dictionary as a JSON object to a JSONL file.
    Args:
        data: Dictionary to append as a JSON object.
        fpath: Path to the JSONL file.
    """
    fpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(fpath, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
    except IOError as e:
        print(f"Failed to save data to file {fpath}: {e}")


def read_jsonl_file(fpath: Path) -> list[dict]:
    """Read a JSONL file and return a list of dictionaries.

    Args:
        fpath: Path to the JSONL file.

    Returns:
        List of dictionaries read from the JSONL file.
    """
    data = []
    if not fpath.exists():
        return data
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    parsed = json.loads(line)
                    assert isinstance(parsed, dict)
                    data.append(parsed)
                except Exception as e:
                    print(f"Failed to decode JSON line {i} in file {fpath}: {e}")
    except IOError as e:
        print(f"Failed to read data from file {fpath}: {e}")
    return data


def read_json(fpath: str | Path) -> Any:
    """Read and return data from a JSON file.

    Args:
        fpath: Path to the JSON file.

    Returns:
        Data loaded from the JSON file.
    """
    with open(fpath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, fpath: str | Path) -> None:
    """Write data to a JSON file.

    Args:
        data: Data to write to the JSON file.
        fpath: Path to the JSON file.
    """
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_yaml(fpath: str | Path) -> dict:
    """Read a YAML file.

    Args:
        fpath: Path to the YAML file.

    Returns:
        Dictionary with parsed YAML content, or empty dict if file is empty.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"YAML file not found: {fpath}")

    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict, fpath: str | Path) -> None:
    """Write data to a YAML file.

    Args:
        data: Dictionary to write.
        fpath: Path to the YAML file.
    """
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
