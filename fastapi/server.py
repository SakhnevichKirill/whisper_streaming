from fastapi import FastAPI, File, UploadFile
import numpy as np
from faster_whisper import WhisperModel
import shutil
import os
import uvicorn
import tempfile
import torch


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="int8")

@app.post("/transcribe")
def transcribe(audio_file: UploadFile = File(...)):
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        # Копируем содержимое загруженного файла во временный файл
        shutil.copyfileobj(audio_file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # Выполняем транскрибирование
        segments, _ = model.transcribe(temp_file_path)
        text = " ".join([seg.text for seg in segments])
    finally:
        # Удаляем временный файл
        os.unlink(temp_file_path)

    return {"text": text}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9000, log_level="info")