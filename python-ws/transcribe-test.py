import json
import multiprocessing
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf
import numpy as np
import sounddevice as sd

# PCM Properties
SAMPLE_RATE = 32000  # 32kHz
SAMPLE_WIDTH = 2      # 16-bit PCM (2 bytes per sample)
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
CHUNK_SIZE = 5 * BYTES_PER_SECOND  # 5 seconds = 320,000 bytes

# Shared Queues
audio_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()


def process_chunk(pcm_data):
    """Convert PCM bytes to WAV and return NumPy array"""
    audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np  # Ready for Whisper


def transcriber_process(audio_queue, result_queue):
    """ Process that loads the Whisper model once and transcribes incoming audio """
    print("🔄 Loading Whisper model...")
    model = WhisperModel("tiny", compute_type="int8")  # Load model only once

    while True:
        audio_np, node_id, timestamp = audio_queue.get()
        if audio_np is None:
            break  # Graceful shutdown

        # Transcribe NumPy array
        segments, _ = model.transcribe(audio_np)
        transcription = " ".join(segment.text for segment in segments)

        result_queue.put((transcription, node_id, timestamp))


def play_audio(audio_np):
    """Play audio from NumPy array"""
    sd.play(audio_np, SAMPLE_RATE)
    sd.wait()  # Wait for playback to finish


# Start Transcription Process
# process = multiprocessing.Process(target=transcriber_process, args=(audio_queue, result_queue))
# process.start()

# Read JSON
with open('output.json') as file:
    data = json.loads(file.read())

# Buffer to store small chunks
audio_buffer = bytearray()

for obj in data:
    audio_chunks = data[obj]
    for audio_b64,timestamp in audio_chunks:
        audio_bytes = base64.b64decode(audio_b64)
        audio_buffer.extend(audio_bytes)  # Append chunk data

        # If buffer reaches 5 seconds, process it
        while len(audio_buffer) >= CHUNK_SIZE:
            pcm_chunk = audio_buffer[:CHUNK_SIZE]  # Take 5 sec chunk
            audio_buffer = audio_buffer[CHUNK_SIZE:]  # Remove processed data

            audio_np = process_chunk(pcm_chunk)  # Convert PCM to NumPy
            # audio_queue.put((audio_np, obj["node_id"], obj["timestamp"]))  # Send to Whisper
            play_audio(audio_np)  # Play audio if needed

# Process leftover audio (if any)
if len(audio_buffer) > 0:
    audio_np = process_chunk(audio_buffer)
    # audio_queue.put((audio_np, obj["node_id"], obj["timestamp"]))
    play_audio(audio_np)

# Send termination signal to transcription process
# audio_queue.put(None)
# process.join()

print("✅ Transcription process completed")
