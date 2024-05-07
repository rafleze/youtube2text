"""Transcribe a youtube video."""

from pytube import YouTube
from faster_whisper import WhisperModel
import tempfile
import os


def transcribe(link, language="en", settings={}):
    """Transcribe a youtube video."""
    if not settings:
        raise Exception("No configuration found for Youtube2Text.")
    model_size_or_path = settings.get("model_size_or_path", "large-v3")
    device = settings.get("device", "cpu")
    compute_type = settings.get("compute_type", "int8")
    try:
        yt = YouTube(link)
        filename = yt.streams.filter(only_audio=True).first().download("media/youtube")
    except Exception as e:
        raise e
    else:
        audio = open(filename, "rb")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio.read())
            temp_file_path = temp_file.name
            model = WhisperModel(
                model_size_or_path, device=device, compute_type=compute_type
            )
            result = model.transcribe(temp_file_path, language=language)
            os.remove(temp_file_path)
            os.remove(filename)
            return result


# settings = {
#     "model_size_or_path": "large-v3",
#     "device": "cpu",
#     "compute_type": "int8",
# }
# x, _ = transcribe("https://www.youtube.com/watch?v=Jo07YIB3HBU", language="it", settings=settings)

# for el in x:
#     print(el.text)
