import socket
import time
import numpy as np
import soundcard as sc
import soundfile as sf
import sounddevice as sd
import wave
from faster_whisper import WhisperModel
import secrets
import threading
import os
import sys

from audio_utils import speaker_live, process_audio

model_size = "tiny"
model = WhisperModel(model_size, device="cpu")


def run_client(chunk_length_s=15.0, stream_chunk_s=1.0, host='localhost', port=43007):
    mic = speaker_live()

    print("Start speaking...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.settimeout(5)  # Set a timeout for the socket operations
        for item in mic:
            # raw_audio = item["raw"]
            print("Sending audio chunk to server...")
            s.sendall(item.tobytes())
            try:
                response = b""
                while True:
                    part = s.recv(1024)
                    response += part
                    if b'\n' in part:
                        break
                response_text = response.decode().strip()
                print("Received:", response_text)
            except socket.timeout:
                print("No response from server, possibly a timeout or server issue.")
                continue

def transcribe(audio_data):
    segments, _ = model.transcribe(audio_data)
    text = []
    for segment in segments:
        text.append(segment.text)
    return " ".join(text)


# Example usage
OUTPUT_FILE_NAME = "out.wav"    # file name.
SAMPLE_RATE = 48000              # [Hz]. sampling rate.
RECORD_SEC = 5                  # [sec]. duration recording audio.
CHUNK = 8 * RECORD_SEC * SAMPLE_RATE
LANG = "en"
QUEUE = []
text = ""
# RATE = 16000
# CHUNK = 512
# CHANNELS = 1
# DTYPE = "float32"

# stream = sd.InputStream(
#     device=1,
#     channels=CHANNELS,
#     samplerate=RATE,
#     callback=process_audio,
#     dtype=DTYPE,
#     blocksize=CHUNK,
# )
    

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 43007))

# stream.start()

# while True:
#     with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
#         # record audio with loopback from default speaker.
#         data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
#         sf.write(file=OUTPUT_FILE_NAME, data=data[:, 0], samplerate=SAMPLE_RATE)
#         sock.send(data[:, 0])

#     data = sock.recv(1024)
#     print(data.decode())

def do_queue():
    global QUEUE,text
    while True:
        if QUEUE:
            fname = QUEUE.pop(0)
            f = open(fname, 'rb')
            l = f.read(1024)
            while (l):
                sock.send(l)
                l = f.read(1024)

            os.system('cls' if os.name == 'nt' else 'clear')
            try:
                os.remove(fname)
            except:
                pass
        else:
            time.sleep(0.1)

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

sock.close()
