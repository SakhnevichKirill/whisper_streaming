from faster_whisper import WhisperModel
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from pyannote.audio import Pipeline
import numpy as np
import asyncio
import dotenv
import torch
import time
import os

dotenv.load_dotenv()

transcriber = None
vad = None

# Load Faster Whisper model
def load_transcriber():
    global transcriber
    transcriber = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Load VAD model
def load_vad():
    global vad
    
    vad = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                   use_auth_token="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")

# Load models
load_transcriber()
load_vad()

async def transcribe_vad(chunk_length_s=30.0, stream_chunk_s=1, stop_pause_time=2.0, output_file="transcriptions.txt"):
    sampling_rate = 16000  # Faster Whisper default sampling rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Start speaking...")
    speaking_start = None
    all_transcriptions = []
    last_speech_time = time.time()

    with open(output_file, "a", encoding="utf-8") as f:
        for chunk in mic:
            audio_data = chunk['raw']
            waveform = audio_data / np.max(np.abs(audio_data))
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # Add channel dimension

            # Use VAD
            vad_result = vad({"waveform": waveform_tensor, "sample_rate": sampling_rate})
            
            current_time = time.time()
            if vad_result.get_timeline():
                last_speech_time = current_time
                if speaking_start is None:
                    speaking_start = current_time
                    print("Speech detected, starting transcription...")

            if speaking_start is not None:
                segments_numpy = waveform.flatten()
                transcription_segments = transcriber.transcribe(segments_numpy, beam_size=5, language="ru")
                for segment in transcription_segments:
                    text = [seg.text for seg in segment if hasattr(seg, 'text')]
                    print(text)
                    if text:
                        print(f"Transcription: {text}")
                        f.write(text[0] + "\n")
                        f.flush()  # Ensure the text is written to the file immediately
                        all_transcriptions.append(text)

            if current_time - last_speech_time > stop_pause_time:
                if speaking_start is not None:
                    print(f"No speech detected for {stop_pause_time} seconds. Stopping.")
                    break
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overload

    return all_transcriptions

async def main_loop():
    while True:
        transcription = await transcribe_vad()
        print("Transcription complete. Results:")
        for line in transcription:
            print(line)
        print("\nReady for next transcription. Press Ctrl+C to exit.")

if __name__ == "__main__":
    asyncio.run(main_loop())