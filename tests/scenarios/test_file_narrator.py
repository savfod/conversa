"""Tests for FileChunkLoader and ContentProcessor classes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conversa.scenarios.file_narrator import (
    Chunk,
    ChunkAsyncPreprocessor,
    ContentProcessor,
    FileChunkLoader,
)


def test_load_txt_file(tmp_path: Path):
    """Test loading a simple text file."""
    content = "Hello world. This is a test file."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=100)
    assert loader.text == content
    assert loader.text_length == len(content)


def test_chunk_splitting_basic(tmp_path: Path):
    """Test basic chunk splitting."""
    content = "First sentence. Second sentence. Third sentence."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=20)

    chunks = []
    while loader.has_more_chunks():
        chunk = loader.get_next_chunk()
        assert chunk is not None
        chunks.append(chunk.text)

    # Should have split into multiple chunks
    assert len(chunks) > 1
    # Verify all content is covered
    assert " ".join(chunks).replace("  ", " ") in content or content in " ".join(chunks)


def test_chunk_splitting_at_sentence_boundaries(tmp_path: Path):
    """Test that chunks prefer to split at sentence boundaries."""
    content = "First sentence. Second sentence. Third sentence. Fourth sentence."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=25)

    chunk = loader.get_next_chunk()
    assert chunk is not None

    # First chunk should end at a sentence boundary
    assert chunk.text.endswith(".")


def test_start_position(tmp_path: Path):
    """Test starting from a specific position."""
    content = "First sentence. Second sentence. Third sentence."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    # Start from position 16 (after "First sentence. ")
    loader = FileChunkLoader(
        file_path=str(temp_file), start_position=16, chunk_size=100
    )

    chunk = loader.get_next_chunk()
    assert chunk is not None

    # Should start with "Second sentence"
    assert chunk.text.startswith("Second")
    assert "First" not in chunk.text


def test_file_not_found():
    """Test that FileNotFoundError is raised for non-existent files."""
    with pytest.raises(FileNotFoundError):
        FileChunkLoader(file_path="/nonexistent/file.txt")


def test_has_more_chunks(tmp_path: Path):
    """Test has_more_chunks method."""
    content = "Short text."

    temp_file = tmp_path / "test.txt"
    temp_file.write_text(content)

    loader = FileChunkLoader(file_path=str(temp_file), chunk_size=100)

    assert loader.has_more_chunks() is True
    loader.get_next_chunk()
    assert loader.has_more_chunks() is False


class TestContentProcessorWithMocking:
    """Test ContentProcessor with mocked dependencies."""

    def test_initialization(self):
        """Test ContentProcessor initialization with default parameters."""
        processor = ContentProcessor()

        assert processor.simplify is False
        assert processor.target_language == "English"
        assert processor.simplification_level == "B1"

    def test_initialization_with_custom_params(self):
        """Test ContentProcessor initialization with custom parameters."""
        processor = ContentProcessor(
            simplify=True,
            target_language="Deutsch",
            simplification_level="A2",
        )

        assert processor.simplify is True
        assert processor.target_language == "Deutsch"
        assert processor.simplification_level == "A2"

    @patch("conversa.scenarios.file_narrator.call_llm")
    def test_simplify_text_calls_llm_with_correct_prompts(self, mock_llm):
        """Test that _simplify_text calls LLM with appropriate prompts."""
        mock_llm.return_value = "  Simplified text  "

        processor = ContentProcessor(
            simplify=True,
            target_language="Deutsch",
            simplification_level="A1",
        )

        input_text = "This is a complex sentence with sophisticated vocabulary."
        result = processor._simplify_text(input_text)

        # Verify LLM was called once
        assert mock_llm.call_count == 1

        # Check the call arguments
        call_args = mock_llm.call_args
        assert "query" in call_args.kwargs
        assert "sys_prompt" in call_args.kwargs

        user_prompt = call_args.kwargs["query"]
        sys_prompt = call_args.kwargs["sys_prompt"]

        # Verify prompts contain key information
        assert "A1" in user_prompt
        assert "Deutsch" in user_prompt
        assert input_text in user_prompt
        assert "language teaching assistant" in sys_prompt.lower()

        # Verify result is stripped
        assert result == "Simplified text"

    @patch("conversa.scenarios.file_narrator.text_to_speech")
    def test_text_to_speech_calls_api_with_instructions(self, mock_tts):
        """Test that _text_to_speech calls API with correct instructions."""
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_tts.return_value = mock_audio

        processor = ContentProcessor(
            target_language="Français",
            simplification_level="B2",
        )

        input_text = "Bonjour le monde"
        result = processor._text_to_speech(input_text)

        # Verify TTS was called once
        assert mock_tts.call_count == 1

        # Check the call arguments
        call_args = mock_tts.call_args
        assert call_args.args[0] == input_text
        assert "instructions" in call_args.kwargs

        instructions = call_args.kwargs["instructions"]
        assert "Français" in instructions
        assert "B2" in instructions

        # Verify result matches mock
        np.testing.assert_array_equal(result, mock_audio)

    @patch("conversa.scenarios.file_narrator.text_to_speech")
    def test_prepare_chunk_without_simplification(self, mock_tts, capsys):
        """Test prepare_chunk when simplification is disabled."""
        mock_audio = np.array([0.5, 0.6])
        mock_tts.return_value = mock_audio

        processor = ContentProcessor(simplify=False)

        text = "Test text"
        processed_text, audio = processor.prepare_chunk(text, chunk_index=1)

        # Text should be unchanged
        assert processed_text == text

        # Audio should be from TTS
        np.testing.assert_array_equal(audio, mock_audio)

        # TTS should be called once
        assert mock_tts.call_count == 1

        # Check console output
        captured = capsys.readouterr()
        assert "Chunk 1" in captured.out
        assert text in captured.out

    @patch("conversa.scenarios.file_narrator.text_to_speech")
    @patch("conversa.scenarios.file_narrator.call_llm")
    def test_prepare_chunk_with_simplification(self, mock_llm, mock_tts, capsys):
        """Test prepare_chunk when simplification is enabled."""
        mock_llm.return_value = "Simplified version"
        mock_audio = np.array([0.7, 0.8])
        mock_tts.return_value = mock_audio

        processor = ContentProcessor(
            simplify=True,
            target_language="Español",
            simplification_level="A2",
        )

        text = "Complex original text"
        processed_text, audio = processor.prepare_chunk(text, chunk_index=2)

        # Text should be simplified
        assert processed_text == "Simplified version"

        # Audio should be from TTS
        np.testing.assert_array_equal(audio, mock_audio)

        # Both LLM and TTS should be called once
        assert mock_llm.call_count == 1
        assert mock_tts.call_count == 1

        # Check console output
        captured = capsys.readouterr()
        assert "Chunk 2" in captured.out
        assert "Simplified version" in captured.out
        assert "Translated & Simplified text:" in captured.out

    @patch("conversa.scenarios.file_narrator.text_to_speech")
    @patch("conversa.scenarios.file_narrator.call_llm")
    def test_prepare_chunk_processes_in_correct_order(self, mock_llm, mock_tts):
        """Test that prepare_chunk calls simplification before TTS."""
        mock_llm.return_value = "Simplified"
        mock_tts.return_value = np.array([1.0])

        processor = ContentProcessor(simplify=True)

        processor.prepare_chunk("Original", chunk_index=1)

        # Verify LLM was called before TTS
        assert mock_llm.call_count == 1
        assert mock_tts.call_count == 1

        # TTS should receive the simplified text
        tts_call_args = mock_tts.call_args
        assert tts_call_args.args[0] == "Simplified"

    def test_all_cefr_levels_supported(self):
        """Test that all CEFR levels can be initialized."""
        levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

        for level in levels:
            processor = ContentProcessor(simplification_level=level)
            assert processor.simplification_level == level


class TestContentProcessorIntegration:
    """Integration tests for ContentProcessor with real API calls."""

    @pytest.mark.slow
    def test_prepare_chunk_real_api_no_simplification(self):
        """Integration test: prepare chunk without simplification using real APIs.

        This test makes actual API calls and is marked as slow.
        It verifies the complete pipeline works end-to-end.
        """
        processor = ContentProcessor(
            simplify=False,
            target_language="English",
        )

        text = "Hello world. This is a test."
        processed_text, audio = processor.prepare_chunk(text, chunk_index=1)

        # Text should be unchanged
        assert processed_text == text

        # Audio should be a numpy array with data
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype in [np.float32, np.float64]

        # Audio should have reasonable values (normalized between -1 and 1)
        assert np.all(np.abs(audio) <= 1.5)  # Allow slight headroom

    @pytest.mark.slow
    def test_prepare_chunk_real_api_with_simplification(self):
        """Integration test: prepare chunk with simplification using real APIs.

        This test makes actual API calls and is marked as slow.
        It verifies simplification and TTS work together.
        """
        processor = ContentProcessor(
            simplify=True,
            target_language="English",
            simplification_level="A1",
        )

        text = (
            "The ubiquitous smartphone has revolutionized interpersonal communication."
        )
        processed_text, audio = processor.prepare_chunk(text, chunk_index=1)

        # Text should be different (simplified)
        assert processed_text != text
        assert len(processed_text) > 0

        # Simplified text should be easier to understand
        # (This is a heuristic check - simpler words tend to be shorter on average)
        avg_word_len_original = sum(len(w) for w in text.split()) / len(text.split())
        avg_word_len_simplified = sum(len(w) for w in processed_text.split()) / len(
            processed_text.split()
        )
        assert (
            avg_word_len_simplified <= avg_word_len_original + 1
        )  # Allow some tolerance

        # Audio should be valid
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype in [np.float32, np.float64]


class TestChunkAsyncPreprocessor:
    """Test ChunkAsyncPreprocessor with mocked dependencies."""

    def test_initialization(self):
        """Test ChunkAsyncPreprocessor initialization."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        assert preprocessor._chunk_loader is mock_loader
        assert preprocessor._content_processor is mock_processor
        assert preprocessor._reading_status is mock_status
        assert preprocessor._prepared_chunk is None
        assert preprocessor._preparation_future is None
        assert preprocessor._run_again is False

    def test_iteration_with_single_chunk(self):
        """Test iterating through a single chunk."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        # Mock chunk data
        text_chunk = Chunk(i=1, text="Test text", end_position=9)
        audio_data = np.array([0.1, 0.2, 0.3])

        # Configure mock loader to return one chunk then indicate no more
        mock_loader.has_more_chunks.side_effect = [True, False]
        mock_loader.get_next_chunk.return_value = text_chunk

        # Configure mock processor to return text and audio
        mock_processor.prepare_chunk.return_value = ("Test text", audio_data)

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Collect all chunks
        chunks = []
        for chunk in preprocessor:
            if chunk is not None:
                chunks.append(chunk)

        # Should have received one chunk
        assert len(chunks) == 1
        assert chunks[0].i == 1
        assert chunks[0].text == "Test text"
        np.testing.assert_array_equal(chunks[0].audio, audio_data)
        assert chunks[0].end_position == 9

        # Verify reading status was updated
        mock_status.update_position.assert_called_once_with(9)

    def test_iteration_with_multiple_chunks(self):
        """Test iterating through multiple chunks."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        # Mock multiple chunks
        chunks_data = [
            Chunk(i=1, text="First", end_position=5),
            Chunk(i=2, text="Second", end_position=11),
            Chunk(i=3, text="Third", end_position=16),
        ]

        mock_loader.has_more_chunks.side_effect = [True, True, True, False]
        mock_loader.get_next_chunk.side_effect = chunks_data

        # Mock processor returns
        mock_processor.prepare_chunk.side_effect = [
            ("First", np.array([0.1])),
            ("Second", np.array([0.2])),
            ("Third", np.array([0.3])),
        ]

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Collect all chunks
        result_chunks = []
        for chunk in preprocessor:
            if chunk is not None:
                result_chunks.append(chunk)

        # Should have received three chunks
        assert len(result_chunks) == 3
        assert result_chunks[0].text == "First"
        assert result_chunks[1].text == "Second"
        assert result_chunks[2].text == "Third"

        # Verify reading status was updated for each chunk
        assert mock_status.update_position.call_count == 3
        mock_status.update_position.assert_any_call(5)
        mock_status.update_position.assert_any_call(11)
        mock_status.update_position.assert_any_call(16)

    def test_iteration_with_no_chunks(self):
        """Test iteration when there are no chunks to process."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        # No chunks available
        mock_loader.has_more_chunks.return_value = False

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Should get empty iteration
        chunks = list(preprocessor)
        assert len(chunks) == 0

        # Processor should not be called
        mock_processor.prepare_chunk.assert_not_called()

        # Status should not be updated
        mock_status.update_position.assert_not_called()

    def test_is_preparation_active(self):
        """Test is_preparation_active method."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Initially not active
        assert preprocessor.is_preparation_active() is False

        # Mock a future that's not done
        mock_future = MagicMock()
        mock_future.done.return_value = False
        preprocessor._preparation_future = mock_future

        assert preprocessor.is_preparation_active() is True

        # Mock a future that's done
        mock_future.done.return_value = True
        assert preprocessor.is_preparation_active() is False

    def test_command_run_again(self):
        """Test 'run_again' command keeps the current chunk available."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        # Setup single chunk
        text_chunk = Chunk(i=1, text="Test", end_position=4)
        mock_loader.has_more_chunks.side_effect = [True, True, False]
        mock_loader.get_next_chunk.return_value = text_chunk
        mock_processor.prepare_chunk.return_value = ("Test", np.array([0.1]))

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Get first chunk
        iterator = iter(preprocessor)
        chunk1 = next(iterator)
        assert chunk1 is None
        chunk2 = next(iterator)
        assert chunk2 is not None
        assert chunk2.text == "Test"

        # Issue run_again command
        preprocessor.command("run_again")

        # Next iteration should return the same chunk
        chunk3 = next(iterator)
        assert chunk3 is not None
        assert chunk3.text == "Test"
        assert chunk3 is chunk2  # Should be the same object

        # Reading status should only be updated once (after third next())
        # because run_again prevents finalization on previous call
        # and initial call is just none
        chunk4 = next(iterator)
        assert chunk4 is not None
        assert mock_status.update_position.call_count == 1

    def test_command_none_does_nothing(self):
        """Test that None command does nothing."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Should not raise any error
        preprocessor.command(None)
        assert preprocessor._run_again is False

    def test_command_unknown_raises_error(self):
        """Test that unknown command raises RuntimeError."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        with pytest.raises(RuntimeError, match="Unknown command"):
            preprocessor.command("invalid_command")

    def test_background_processing_error_handling(self):
        """Test that errors in background processing are handled gracefully."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        # Setup chunk
        text_chunk = Chunk(i=1, text="Test", end_position=4)
        mock_loader.has_more_chunks.side_effect = [True, False]
        mock_loader.get_next_chunk.return_value = text_chunk

        # Make processor raise an exception
        mock_processor.prepare_chunk.side_effect = RuntimeError("Processing failed")

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Should handle error and return no chunks
        chunks = list(preprocessor)
        assert chunks == [None]

        # Status should not be updated
        mock_status.update_position.assert_not_called()

    def test_finished_method(self):
        """Test _finished method correctly identifies when iteration is complete."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_status = MagicMock()

        preprocessor = ChunkAsyncPreprocessor(
            chunk_loader=mock_loader,
            content_processor=mock_processor,
            reading_status=mock_status,
        )

        # Initially finished (no chunks, no future, no prepared chunk)
        mock_loader.has_more_chunks.return_value = False
        assert preprocessor._finished() is True

        # Not finished if there's a prepared chunk
        preprocessor._prepared_chunk = Chunk(i=1, text="Test", end_position=4)
        assert preprocessor._finished() is False

        # Not finished if there's a future
        preprocessor._prepared_chunk = None
        mock_future = MagicMock()
        preprocessor._preparation_future = mock_future
        assert preprocessor._finished() is False

        # Not finished if loader has more chunks
        preprocessor._preparation_future = None
        mock_loader.has_more_chunks.return_value = True
        assert preprocessor._finished() is False
