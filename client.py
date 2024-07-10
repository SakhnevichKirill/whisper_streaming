import socket
import time
import numpy as np
import soundcard as sc
import soundfile as sf
import wave
from faster_whisper import WhisperModel

from audio_utils import speaker_live

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

sock = socket.socket()
sock.connect(('localhost', 43007))

while True:
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
        # record audio with loopback from default speaker.
        data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
        sf.write(file=OUTPUT_FILE_NAME, data=data[:, 0], samplerate=SAMPLE_RATE)
    wf = wave.open("out.wav")
    data = wf.readframes(CHUNK)
    sock.send(data)

    data = sock.recv(1024)
    print(data.decode())

sock.close()
