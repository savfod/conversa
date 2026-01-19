import pytest

from conversa.util.io import (
    append_to_jsonl_file,
    read_json,
    read_jsonl_file,
    read_yaml,
    write_json,
    write_yaml,
)


def test_write_and_read_json(tmp_path):
    """Test writing and reading a JSON file."""
    test_data = {"name": "Alice", "age": 30, "hobbies": ["reading", "cycling"]}
    file_path = tmp_path / "test.json"

    write_json(test_data, file_path)

    assert file_path.exists()
    loaded_data = read_json(file_path)
    assert loaded_data == test_data


def test_write_json_creates_parent_directories(tmp_path):
    """Test that write_json creates parent directories if they don't exist."""
    test_data = {"key": "value"}
    file_path = tmp_path / "nested" / "deep" / "test.json"

    write_json(test_data, file_path)

    assert file_path.exists()
    loaded_data = read_json(file_path)
    assert loaded_data == test_data


def test_write_json_with_unicode(tmp_path):
    """Test writing and reading JSON with Unicode characters."""
    test_data = {"message": "Привет мир! 你好世界!", "emoji": "🌍"}
    file_path = tmp_path / "unicode.json"

    write_json(test_data, file_path)
    loaded_data = read_json(file_path)

    assert loaded_data == test_data


def test_write_json_overwrites_existing_file(tmp_path):
    """Test that write_json overwrites existing files."""
    file_path = tmp_path / "overwrite.json"

    write_json({"version": 1}, file_path)
    write_json({"version": 2}, file_path)

    loaded_data = read_json(file_path)
    assert loaded_data == {"version": 2}


def test_append_to_jsonl_file(tmp_path):
    """Test appending multiple entries to a JSONL file."""
    file_path = tmp_path / "test.jsonl"

    data1 = {"id": 1, "name": "Alice"}
    data2 = {"id": 2, "name": "Bob"}
    data3 = {"id": 3, "name": "Charlie"}

    append_to_jsonl_file(data1, file_path)
    append_to_jsonl_file(data2, file_path)
    append_to_jsonl_file(data3, file_path)

    assert file_path.exists()
    loaded_data = read_jsonl_file(file_path)
    assert len(loaded_data) == 3
    assert loaded_data[0] == data1
    assert loaded_data[1] == data2
    assert loaded_data[2] == data3


def test_append_to_jsonl_creates_parent_directories(tmp_path):
    """Test that append_to_jsonl_file creates parent directories."""
    file_path = tmp_path / "nested" / "test.jsonl"
    data = {"key": "value"}

    append_to_jsonl_file(data, file_path)

    assert file_path.exists()
    loaded_data = read_jsonl_file(file_path)
    assert loaded_data == [data]


def test_append_to_jsonl_with_unicode(tmp_path):
    """Test appending to JSONL with Unicode characters."""
    file_path = tmp_path / "unicode.jsonl"

    data1 = {"text": "Hola mundo"}
    data2 = {"text": "🌟🚀✨"}

    append_to_jsonl_file(data1, file_path)
    append_to_jsonl_file(data2, file_path)

    loaded_data = read_jsonl_file(file_path)
    assert loaded_data == [data1, data2]


def test_read_jsonl_file_empty(tmp_path):
    """Test reading an empty JSONL file."""
    file_path = tmp_path / "empty.jsonl"
    file_path.touch()

    loaded_data = read_jsonl_file(file_path)
    assert loaded_data == []


def test_read_jsonl_file_nonexistent(tmp_path):
    """Test reading a non-existent JSONL file returns empty list."""
    file_path = tmp_path / "nonexistent.jsonl"

    loaded_data = read_jsonl_file(file_path)
    assert loaded_data == []


def test_read_jsonl_file_with_invalid_json(tmp_path, capsys):
    """Test reading a JSONL file with invalid JSON line."""
    file_path = tmp_path / "invalid.jsonl"

    # Write valid and invalid lines
    with open(file_path, "w") as f:
        f.write('{"valid": true}\n')
        f.write("invalid json line\n")
        f.write('{"also_valid": true}\n')

    loaded_data = read_jsonl_file(file_path)

    # Should skip the invalid line
    assert len(loaded_data) == 2
    assert loaded_data[0] == {"valid": True}
    assert loaded_data[1] == {"also_valid": True}

    # Check that error was printed
    captured = capsys.readouterr()
    assert "Failed to decode JSON" in captured.out


def test_read_jsonl_file_with_non_dict(tmp_path, capsys):
    """Test reading a JSONL file with non-dictionary JSON."""
    file_path = tmp_path / "non_dict.jsonl"

    # Write a list instead of dict
    with open(file_path, "w") as f:
        f.write('{"valid": true}\n')
        f.write("[1, 2, 3]\n")
        f.write('{"also_valid": true}\n')

    loaded_data = read_jsonl_file(file_path)

    # Should skip the non-dict line
    assert len(loaded_data) == 2
    assert loaded_data[0] == {"valid": True}
    assert loaded_data[1] == {"also_valid": True}

    # Check that error was printed
    captured = capsys.readouterr()
    assert "Failed to decode JSON" in captured.out


def test_append_and_read_jsonl_multiple_sessions(tmp_path):
    """Test appending to JSONL file across multiple sessions."""
    file_path = tmp_path / "multi_session.jsonl"

    # First session
    append_to_jsonl_file({"session": 1, "action": "start"}, file_path)
    append_to_jsonl_file({"session": 1, "action": "end"}, file_path)

    # Read after first session
    data = read_jsonl_file(file_path)
    assert len(data) == 2

    # Second session - append more
    append_to_jsonl_file({"session": 2, "action": "start"}, file_path)
    append_to_jsonl_file({"session": 2, "action": "end"}, file_path)

    # Read all data
    data = read_jsonl_file(file_path)
    assert len(data) == 4
    assert data[0]["session"] == 1
    assert data[2]["session"] == 2


def test_json_with_nested_structures(tmp_path):
    """Test JSON read/write with deeply nested structures."""
    test_data = {
        "users": [
            {
                "name": "Alice",
                "preferences": {
                    "theme": "dark",
                    "notifications": {"email": True, "push": False},
                },
            },
            {
                "name": "Bob",
                "preferences": {
                    "theme": "light",
                    "notifications": {"email": False, "push": True},
                },
            },
        ]
    }
    file_path = tmp_path / "nested.json"

    write_json(test_data, file_path)
    loaded_data = read_json(file_path)

    assert loaded_data == test_data
    assert loaded_data["users"][0]["preferences"]["notifications"]["email"] is True


def test_jsonl_with_complex_objects(tmp_path):
    """Test JSONL with complex nested objects."""
    data1 = {
        "timestamp": "2025-11-24T10:00:00",
        "user": "alice",
        "mistakes": [
            {"word": "their", "correction": "there"},
            {"word": "your", "correction": "you're"},
        ],
    }
    data2 = {"timestamp": "2025-11-24T11:00:00", "user": "bob", "mistakes": []}

    file_path = tmp_path / "complex.jsonl"
    append_to_jsonl_file(data1, file_path)
    append_to_jsonl_file(data2, file_path)

    loaded_data = read_jsonl_file(file_path)
    assert len(loaded_data) == 2
    assert loaded_data[0]["mistakes"][0]["word"] == "their"
    assert loaded_data[1]["mistakes"] == []


def test_write_and_read_yaml(tmp_path):
    """Test writing and reading a YAML file."""
    test_data = {"name": "Alice", "age": 30, "hobbies": ["reading", "cycling"]}
    file_path = tmp_path / "test.yaml"

    write_yaml(test_data, file_path)

    assert file_path.exists()
    loaded_data = read_yaml(file_path)
    assert loaded_data == test_data


def test_write_yaml_creates_parent_directories(tmp_path):
    """Test that write_yaml creates parent directories if they don't exist."""
    test_data = {"key": "value"}
    file_path = tmp_path / "nested" / "deep" / "test.yaml"

    write_yaml(test_data, file_path)

    assert file_path.exists()
    loaded_data = read_yaml(file_path)
    assert loaded_data == test_data


def test_write_yaml_with_unicode(tmp_path):
    """Test writing and reading YAML with Unicode characters."""
    test_data = {"message": "Привет мир! 你好世界!", "emoji": "🌍"}
    file_path = tmp_path / "unicode.yaml"

    write_yaml(test_data, file_path)
    loaded_data = read_yaml(file_path)

    assert loaded_data == test_data


def test_read_yaml_nonexistent(tmp_path):
    """Test reading a non-existent YAML file raises FileNotFoundError."""
    file_path = tmp_path / "nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        read_yaml(file_path)


def test_read_yaml_empty_file(tmp_path):
    """Test reading an empty YAML file returns empty dict."""
    file_path = tmp_path / "empty.yaml"
    file_path.touch()

    loaded_data = read_yaml(file_path)
    assert loaded_data == {}
