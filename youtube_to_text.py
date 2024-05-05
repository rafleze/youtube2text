"""Youtube to text plugin."""

from pytube import YouTube
from faster_whisper import WhisperModel
import tempfile
import os
from pydantic import BaseModel, Field
from cat.experimental.form import form, CatForm
from cat.mad_hatter.decorators import plugin


class Settings(BaseModel):
    model_size_or_path: str = Field(
        title="Model size or path",
        description="Size of the model to use (tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large), a path to a converted model directory, or a CTranslate2-converted Whisper model ID from the HF Hub. When a size or a model ID is configured, the converted model is downloaded from the Hugging Face Hub.",
        default="large-v3",
    )
    device: str = Field(
        title="Device",
        description='Device to use for computation ("cpu", "cuda", "auto").',
        default="cpu",
    )
    compute_type: str = Field(
        title="Compute type",
        description="Type to use for computation. See https://opennmt.net/CTranslate2/quantization.html.",
        default="int8",
    )


@plugin
def settings_schema():
    return Settings.schema()


def transcribe(link, language="en", settings={}):
    """Transcribe a youtube video."""
    if not settings:
        return Exception("No configuration found for Youtube2Text.")
    model_size_or_path = settings.get("model_size_or_path", "large-v3")
    device = settings.get("device", "cpu")
    compute_type = settings.get("compute_type", "int8")
    try:
        yt = YouTube(link)
        filename = yt.streams.filter(only_audio=True).first().download("media/youtube")
    except Exception:
        return Exception("Error downloading youtube file.")
    else:
        audio = open(filename, "rb")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio.read())
            temp_file_path = temp_file.name
            model = WhisperModel(model_size_or_path, device, compute_type)
            result = model.transcribe(temp_file_path, language=language)
            os.remove(temp_file_path)
            os.remove(filename)
            return result


class VideoInfo(BaseModel):
    """Information about a youtube video."""

    youtube_link: str
    language: str


@form
class TranscriptionForm(CatForm):
    """Transcription form."""

    description = "Youtube video transcription"
    model_class = VideoInfo
    start_examples = [
        "transcribe a youtube video",
        "transcribe a video",
    ]
    stop_examples = [
        "stop transcribe",
    ]
    ask_confirm = True

    def submit(self, form_data):
        """Submit the form."""
        settings = self.cat.mad_hatter.plugins["youtube2text"].load_settings()
        segments, _ = transcribe(
            form_data["youtube_link"], form_data["language"], settings
        )
        result = "".join([s.text for s in segments])
        prompt = f"Summerize the following text: {result}"
        summary = self.cat.llm(prompt)
        output = f"The transcription is: \n{result}\n\nSummary: {summary}"
        return {"output": output}
