import json
import multiprocessing
from faster_whisper import WhisperModel
import base64
import io
import soundfile as sf
import sounddevice as sd

# PCM Properties
SAMPLE_RATE = 32000  # 32kHz
SAMPLE_WIDTH = 2      # 16-bit PCM (2 bytes per sample)
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
CHUNK_SIZE = 10 * BYTES_PER_SECOND  # 5 seconds = 320,000 bytes

# Shared Queues
audio_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()


def process_chunk(pcm_data):
    """Convert PCM bytes to WAV and return as BytesIO"""
    wav_io = io.BytesIO()
    with sf.SoundFile(wav_io, mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS, format="WAV") as file:
        file.write(memoryview(pcm_data).cast('h'))  # Write PCM as WAV
    wav_io.seek(0)  # Reset pointer
    return wav_io  # Ready for Whisper


def transcriber_process(audio_queue, result_queue):
    """ Process that loads the Whisper model once and transcribes incoming audio """
    print("🔄 Loading Whisper model...")
    model = WhisperModel("tiny", compute_type="int8")  # Load model only once

    while True:
        item = audio_queue.get()
        if item is None:
            break  # Graceful shutdown

        wav_io, node_id = item  # Now safely unpack
        segments, _ = model.transcribe(wav_io,language='en')
        for segment in segments:
            print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

        # result_queue.put((transcription, node_id, timestamp))

# Start Transcription Process
process = multiprocessing.Process(target=transcriber_process, args=(audio_queue, result_queue))
process.start()

# Read JSON
with open('output.json') as file:
    data = json.loads(file.read())

# Buffer to store small chunks
audio_buffer = bytearray()

for obj in data:
    audio_chunks = data[obj]
    for audio_b64 in audio_chunks:
        audio_bytes = base64.b64decode(audio_b64)
        audio_buffer.extend(audio_bytes)  # Append chunk data

        # If buffer reaches 5 seconds, process it
        while len(audio_buffer) >= CHUNK_SIZE:
            pcm_chunk = audio_buffer[:CHUNK_SIZE]  # Take 5 sec chunk
            audio_buffer = audio_buffer[CHUNK_SIZE:]  # Remove processed data

            wav_file = process_chunk(pcm_chunk)  # Convert PCM to WAV
            audio_queue.put((wav_file, obj))  # Send to Whisper

            
if len(audio_buffer) > 0:
    wav_file = process_chunk(audio_buffer)  # Convert remaining PCM to WAV
    audio_queue.put((wav_file, obj))  # Send to Whisper

# Send termination signal to transcription process
audio_queue.put(None)
process.join()

print("✅ Transcription process completed")
