from faster_whisper import WhisperModel
import socket
import soundfile
import io
import socket
import line_packet
from whisper_online import add_shared_args, set_logging, asr_factory, load_audio_chunk, librosa

import sys
import argparse
import os
import logging
import numpy as np
import uvicorn
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_size = "large-v3"
model = WhisperModel(model_size, device="cpu")

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='localhost')
parser.add_argument("--port", type=int, default=43007)
parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

def transcribe(audio_data):
    segments, _ = model.transcribe(audio_data)
    text = []
    for segment in segments:
        text.append(segment.text)
    return " ".join(text)

SAMPLE_RATE = 48000
RECORD_SEC = 5
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((args.host, args.port))
    s.listen(1)
    logger.info('Listening on'+str((args.host, args.port)))
    conn, addr = s.accept()
    while True:
        # data = conn.recv(8 * SAMPLE_RATE * RECORD_SEC)
        # if not data:
        #     break
        # audio_data = np.frombuffer(data, dtype=np.float64)

        audio_data = conn.recv(1024)
        while (len(audio_data) < 8 * SAMPLE_RATE * RECORD_SEC):
            audio_data += conn.recv(1024)

        text = transcribe(audio_data)
        print(text)
        # conn.send(text.encode())
        logger.info('Connection to client closed')
    conn.close()
print('broken')

