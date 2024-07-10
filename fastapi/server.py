from fastapi import FastAPI
import numpy as np
from faster_whisper import WhisperModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = FastAPI()

model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

@app.post("/transcribe")
def transcribe(data: dict):
    audio_data = data["audio_data"]
    segments, _ = model.transcribe(np.array(audio_data))
    text = []
    for segment in segments:
        text.append(segment.text)
    return {"text": " ".join(text)}