import secrets
import datetime
import platform
import subprocess
import threading
import numpy as np
import requests
import soundcard as sc
import soundfile as sf
import keyboard
import os
from typing import Optional, Tuple, Union


running = True
OUTPUT_FILE_NAME_MIC = "micro_out.wav"
OUTPUT_FILE_NAME_SPK = "dynamic_out.wav"
TEXT_OUTPUT_FILE = "transcriptions.txt"
SAMPLE_RATE = 48000
RECORD_SEC = 5

def check_for_stop():
    global running
    keyboard.wait("1")
    running = False
    print("Остановка записи...")

def speaker_record_stream():
    global running
    while running:
        with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
            data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
            sf.write(OUTPUT_FILE_NAME_SPK, data, SAMPLE_RATE, format='WAV', subtype='PCM_16')

            with open(OUTPUT_FILE_NAME_SPK, "rb") as audio_file:
                files = {"audio_file": (os.path.basename(OUTPUT_FILE_NAME_SPK), audio_file, "audio/wav")}
                response = requests.post("http://localhost:9000/transcribe", files=files)
            
            output_text = response.json().get("text", "")
            with open(TEXT_OUTPUT_FILE, "a") as f:
                f.write(f"Текст с динамиков: {output_text}\n")
            print("speaker")
            print(output_text)

def microphone_record_stream():
    global running
    while running:
        with sc.get_microphone(id=str(sc.default_microphone().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
            data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
            sf.write(OUTPUT_FILE_NAME_MIC, data, SAMPLE_RATE, format='WAV', subtype='PCM_16')

            with open(OUTPUT_FILE_NAME_MIC, "rb") as audio_file:
                files = {"audio_file": (os.path.basename(OUTPUT_FILE_NAME_MIC), audio_file, "audio/wav")}
                response = requests.post("http://localhost:9000/transcribe", files=files)
            
            output_text = response.json().get("text", "")
            with open(TEXT_OUTPUT_FILE, "a") as f:
                f.write(f"Текст с микрофона: {output_text}\n")
            print("microphone")
            print(output_text)

def sc_transcribe():
    global running
    running = True
    
    speaker = threading.Thread(target=speaker_record_stream)
    microphone = threading.Thread(target=microphone_record_stream)
    stop_thread = threading.Thread(target=check_for_stop)
    
    speaker.start()
    microphone.start()
    stop_thread.start()
    
    speaker.join()
    microphone.join()
    stop_thread.join()

def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
):
    ar = f"{sampling_rate}"
    ac = "1"
    if format_for_conversion == "s16le":
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()
    if system == "Linux":
        format_ = "alsa"
        input_ = "default"
    elif system == "Darwin":
        format_ = "avfoundation"
        input_ = ":0"
    elif system == "Windows":
        format_ = "dshow"
        input_ = _get_microphone_name()

    print(input_)
    input_1, input_2 = input_
    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i", 
        input_1,
        "-f", 
        format_,
        "-i", 
        input_2,
        "-filter_complex", 
        "amix=inputs=2",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-fflags",
        "nobuffer",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)
    for item in iterator:
        yield item

def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
):
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s

    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample
    audio_time = datetime.datetime.now()
    delta = datetime.timedelta(seconds=chunk_s)
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        item["sampling_rate"] = sampling_rate
        audio_time += delta
        if datetime.datetime.now() > audio_time + 10 * delta:
            continue
        yield item

def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool = False):
    acc = b""
    stride_left, stride_right = stride
    if stride_left + stride_right >= chunk_len:
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0
    for raw in iterator:
        acc += raw
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}
        else:
            while len(acc) >= chunk_len:
                stride = (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}
                if stream:
                    item["partial"] = False
                yield item
                _stride_left = stride_left
                acc = acc[chunk_len - stride_left - stride_right:]
    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}
        if stream:
            item["partial"] = False
        yield item

def _ffmpeg_stream(ffmpeg_command, buflen: int):
    bufsize = 2**24
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                raw = ffmpeg_process.stdout.read(buflen)
                if raw == b"":
                    break
                yield raw
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error

def _get_microphone_name():
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]

        if microphone_lines:
            microphone_name_1 = microphone_lines[0].split('"')[1]
            microphone_name_2 = microphone_lines[1].split('"')[1]
            print(f"Using microphone: {microphone_name_1}")
            print(f"Using speaker: {microphone_name_2}")
            return (f"audio={microphone_name_1}", f"audio={microphone_name_2}")
    except FileNotFoundError:
        print("ffmpeg was not found. Please install it or make sure it is in your system PATH.")

    return "default"

def transcribe(chunk_length_s=15.0, stream_chunk_s=1.0):
    global running
    running = True
    
    mic = ffmpeg_microphone_live(
        sampling_rate=16000,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking... (Press '1' to stop)")
    stop_thread = threading.Thread(target=check_for_stop)
    stop_thread.start()

    for item in mic:
        if not running:
            break
        response = requests.post("http://localhost:9000/transcribe", json={"audio_data": item["raw"].tolist()})
        output_array = response.json()
        with open(TEXT_OUTPUT_FILE, "a") as f:
            f.write(f"Текст с микрофона: {output_array}\n")
        print(output_array)

    stop_thread.join()
    print("Запись остановлена.")
