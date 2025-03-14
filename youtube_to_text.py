"""Youtube to text plugin."""

from .transcriber import transcribe
from pydantic import BaseModel, Field
from cat.experimental.form import form, CatForm
from cat.mad_hatter.decorators import plugin


class Settings(BaseModel):
    """Settings for the Youtube2Text plugin."""

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
