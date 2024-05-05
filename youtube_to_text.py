from pytube import YouTube
from faster_whisper import WhisperModel
import tempfile
import os
from pydantic import BaseModel
from cat.experimental.form import form, CatForm


def transcribe(link, language="en"):
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
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            result = model.transcribe(temp_file_path, language=language)
            os.remove(temp_file_path)
            os.remove(filename)
            return result


class VideoInfo(BaseModel):
    youtube_link: str
    language: str


@form
class TranscriptionForm(CatForm):
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
        segments, _ = transcribe(form_data["youtube_link"], form_data["language"])
        result = "".join([s.text for s in segments])
        prompt = f"Summerize the following text: {result}"
        summary = self.cat.llm(prompt)
        output = f"The transcription is: \n{result}\n\nSummary: {summary}"
        return {"output": output}
