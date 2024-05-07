import pytest
from unittest import TestCase
from unittest.mock import patch
from youtube2text.transcriber import transcribe


@patch("youtube2text.transcriber.YouTube")
@patch("youtube2text.transcriber.WhisperModel")
def test_transcribe_with_settings(mock_whisper_model, mock_youtube):
    """Test transcribe function with settings."""

    with open("tests/media/youtube/test.mp4", "w") as f:
        f.write("test")

    mock_youtube.return_value.streams.filter.return_value.first.return_value.download.return_value = (
        "tests/media/youtube/test.mp4"
    )
    mock_whisper_model.return_value.transcribe.return_value = [
        {"text": "Hello", "start": 0.0, "end": 1.0},
        {"text": "World", "start": 1.0, "end": 2.0},
    ]
    settings = {
        "model_size_or_path": "large-v3",
        "device": "cpu",
        "compute_type": "int8",
    }
    result = transcribe(
        "https://www.youtube.com/watch?v=Jo07YIB3HBU", language="it", settings=settings
    )
    TestCase().assertListEqual(
        result,
        [
            {"text": "Hello", "start": 0.0, "end": 1.0},
            {"text": "World", "start": 1.0, "end": 2.0},
        ],
    )


@patch("youtube2text.transcriber.YouTube")
def test_transcribe_without_settings(mock_youtube):
    """Test transcribe function with settings."""

    with open("tests/media/youtube/test.mp4", "w") as f:
        f.write("test")

    mock_youtube.return_value.streams.filter.return_value.first.return_value.download.return_value = (
        "tests/media/youtube/test.mp4"
    )
    settings = {}
    with pytest.raises(Exception):
        transcribe(
            "https://www.youtube.com/watch?v=Jo07YIB3HBU",
            language="it",
            settings=settings,
        )


@patch("youtube2text.transcriber.YouTube")
def test_transcribe_with_download_exception(mock_youtube):
    """Test transcribe function with settings."""

    # mock youtube stream filter first download as exception
    mock_youtube.return_value.streams.filter.return_value.first.return_value.download.side_effect = Exception(
        "Download failed"
    )
    settings = {
        "model_size_or_path": "large-v3",
        "device": "cpu",
        "compute_type": "int8",
    }
    with pytest.raises(Exception):
        transcribe(
            "https://www.youtube.com/watch?v=Jo07YIB3HBU",
            language="it",
            settings=settings,
        )
