"""Transcribe a youtube video."""
from faster_whisper import WhisperModel
import tempfile
import os
import yt_dlp


def transcribe(link, language="en", settings={}):
    """Transcribe a youtube video."""
    if not settings:
        raise Exception("No configuration found for Youtube2Text.")
    model_size_or_path = settings.get("model_size_or_path", "large-v3")
    device = settings.get("device", "cpu")
    compute_type = settings.get("compute_type", "int8")
    filename = download_audio(link)
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


def download_audio(link, output_folder="media/youtube"):  # pragma: no cover
    try:

        inner_output_folder = os.path.join(os.path.dirname(__file__), output_folder)
        os.makedirs(inner_output_folder, exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{inner_output_folder}/%(title)s.%(ext)s",
            "noprogress": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(link, download=True)
            filename = ydl.prepare_filename(info_dict)

        return filename

    except Exception as e:
        raise e