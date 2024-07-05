import socket
import time

from audio_utils import ffmpeg_microphone_live


def transcribe(chunk_length_s=15.0, stream_chunk_s=1.0, host='localhost', port=43007):
    mic = ffmpeg_microphone_live(
        sampling_rate=16000,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.settimeout(5)  # Set a timeout for the socket operations
        for item in mic:
            raw_audio = item["raw"]
            print("Sending audio chunk to server...")
            s.sendall(raw_audio.tobytes())
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


# Example usage
transcribe()
