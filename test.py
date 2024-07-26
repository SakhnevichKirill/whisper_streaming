# import soundcard as sc
# import soundfile as sf
# import wave
# from faster_whisper import WhisperModel
# import numpy as np

# model_size = "tiny"
# model = WhisperModel(model_size, device="cpu")

# def transcribe(audio_data):
#     segments, _ = model.transcribe(audio_data)
#     text = []
#     for segment in segments:
#         text.append(segment.text)
#     return " ".join(text)

# OUTPUT_FILE_NAME = "out.wav"    # file name.
# SAMPLE_RATE = 48000              # [Hz]. sampling rate.
# RECORD_SEC = 5                  # [sec]. duration recording audio.
# CHUNK = 8 * RECORD_SEC * SAMPLE_RATE

# while True:
#     with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
#         # record audio with loopback from default speaker.
#         data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
#         text = transcribe(data[:, 0])
#         print(text)
    
from faster_whisper import WhisperModel
import time
import soundcard as sc
import soundfile as sf#to save the audio file
import secrets
import threading
import os
import sys

start = time.time()
model = WhisperModel("small",device="cpu",compute_type="float32")
SAMPLE_RATE = 48000
QUEUE = []
text = ""
LANG = "en"

def do_queue():
    global QUEUE,text
    while True:
        if QUEUE:
            fname = QUEUE.pop(0)
            segments, info = model.transcribe(fname,language=LANG)
            t = " ".join([seg.text for seg in segments])
            text += t.strip()+" "
            text = text.replace("Sous-titres réalisés par la communauté d'Amara.org","")#sowwy amara (for french subtitles)
            os.system('cls' if os.name == 'nt' else 'clear')
            sys.stdout.write(f"[*] Recording, ({round(time.time()-start,2)} seconds):\n {text}\r")
            try:
                os.remove(fname)
            except:
                pass
        else:
            time.sleep(0.1)

sys.stdout.write(f"Initialized in {time.time()-start} seconds\n")
bg_sound = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
with bg_sound.recorder(samplerate=SAMPLE_RATE) as mic:
    threading.Thread(target=do_queue,daemon=True).start()
    for _ in range(100):#record 100 times, you can increase to inf
        data = mic.record(numframes=SAMPLE_RATE*4)#record 4 seconds
        fname = secrets.token_hex(16)+".wav"
        sf.write(fname,data,SAMPLE_RATE)
        QUEUE.append(fname)
        while len(QUEUE) > 50:
            time.sleep(0.1)#wait for the queue to clear
    print("Done recording")