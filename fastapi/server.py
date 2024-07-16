from fastapi import FastAPI, File, UploadFile
import numpy as np
from faster_whisper import WhisperModel
import shutil
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = FastAPI()

model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# @app.post("/transcribe")
# def transcribe(data: dict):
#     audio_data = data["audio_data"]
#     segments, _ = model.transcribe(np.array(audio_data))
#     text = []
#     for segment in segments:
#         text.append(segment.text)
#     return {"text": " ".join(text)}

@app.post("/transcribe")
def transcribe(audio_file: UploadFile = File(...)):
    with open(f'{audio_file.filename}', 'wb') as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    segments, _ = model.transcribe(audio_file.filename)
    text = " ".join([seg.text for seg in segments])
    os.remove(audio_file.filename)
    return {"text": " ".join(text)}