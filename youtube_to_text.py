from pytube import YouTube
from faster_whisper import WhisperModel
import tempfile
import os
from pydantic import BaseModel
from cat.experimental.form import form, CatForm

from cat.mad_hatter.decorators import hook
from typing import Iterator
from langchain.schema import Document
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.base import BaseBlobParser

def transcribe(link, language="en"):
    try:
        yt = YouTube(link)
        filename = yt.streams.filter(only_audio=True).first().download("media/youtube")
    except Exception:
        return Exception(
            "Error downloading youtube file."
        )
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
        

# class YoutubeParser(BaseBlobParser):
#     """Parser for audio blobs."""

#     def __init__(self, key: str, lang: str = "en"):
#         self.key = key
#         self.lang = lang

#     def lazy_parse(self, blob: Blob) -> Iterator[Document]:
#         """Lazily parse the blob."""

#         content = transcribe(self.key, self.lang, (blob.path, blob.as_bytes(), blob.mimetype))

#         yield Document(page_content=content, metadata={"source": "whispering_cat", "name": blob.path.rsplit('.', 1)[0]})


# @hook
# def before_cat_sends_message(message, cat):
#     print("SONO DIOCANE QUI")
#     prompt = f'Rephrase the following sentence in a grumpy way: {message["content"]}'
#     message["content"] = cat.llm(prompt)
#     print(message)
#     return message


# @hook
# def before_rabbithole_splits_text(message, cat):
#     print(message)
#     if "https://www.youtube.com" in message[0].metadata["source"]:
#         print(message[0].metadata["source"])
#         cat.send_ws_message(f"I'm starting the transcrption", "chat")
#         segments, _ = transcribe(message[0].metadata["source"], "it")
#         result = "".join([s.text for s in segments])
#         cat.send_ws_message(f"The transcription is: \n{result}", "chat")
#         prompt = f'Summerize the following text: {result}'
#         message["content"] = cat.llm(prompt)
#     return message


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
        print("STO INVIANDO: ", form_data)
        segments, _ = transcribe(form_data["youtube_link"], form_data["language"])
        result = "".join([s.text for s in segments])
        prompt = f'Summerize the following text: {result}'
        summary = self.cat.llm(prompt)
        output = f"The transcription is: \n{result}\n\nSummary: {summary}"
        print("RESULT: ", output)
        return {
            "output": output
        }