import asyncio
import websockets
import numpy as np
import json
import base64
import sounddevice as sd
from collections import defaultdict
import numpy as np
import time
import multiprocessing
import io
import soundfile as sf
import sounddevice as sd

silence_threshold = 500  # Adjust based on noise levels
silence_start = None  # Track when silence begins
silence_duration = 0  # Track silence length


silence_threshold = 500  # Adjust based on actual noise level
silence_start = None
silence_duration = 0

last_silence_notification = 0  # Track last silence notification timestamp
silence_notification_interval = 60  # Send notification every 60 seconds (1 min)

# PCM Properties
SAMPLE_RATE = 32000  # 32kHz
SAMPLE_WIDTH = 2      # 16-bit PCM (2 bytes per sample)
CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
CHUNK_SIZE = 10 * BYTES_PER_SECOND  # 5 seconds = 320,000 bytes


audio_data = defaultdict(list)  # Automatically creates empty bytearray if key doesn't exist

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

    while True:
        item = audio_queue.get()
        if item is None:
            break  # Graceful shutdown
        # print('New Item')

async def detect_silence(audio_bytes, websocket):
    """Detect silence periods longer than 5 seconds and notify the client every 1 minute"""
    global silence_start, silence_duration, last_silence_notification  # Declare globals

    # Convert raw PCM bytes to NumPy array
    if not audio_bytes:  
        print("⚠️ Warning: Empty audio frame received!")
        return  

    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

    # Compute RMS volume safely
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))  

    # Ensure RMS is a valid number
    if np.isnan(rms):
        print("⚠️ Warning: Computed RMS is NaN!")
        return  

    # Detect silence
    if rms < silence_threshold:
        if silence_start is None:
            silence_start = time.time()
        silence_duration = time.time() - silence_start

        if silence_duration >= 5:  # Silence has lasted more than 5 seconds
            current_time = time.time()
            if last_silence_notification == 0 or (current_time - last_silence_notification >= silence_notification_interval):
                print("⚠️ Silence detected for more than 5 seconds!")
                await websocket.send("⚠️ Warning: Silence detected for more than 5 seconds!")
                last_silence_notification = current_time  # Update last notification time
    else:
        silence_start = None  # Reset silence tracking
        last_silence_notification = 0

async def echo(websocket):
    global json_data
    print(f"New connection: {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):  
                pass
            else:
                data = json.loads(message)  # Parse JSON
                audio_bytes = base64.b64decode(data['audio'])
                # print("New Data Incoming..:")
                audio_data[f"{data['node_id']}_{data['username']}"].append((data['timestamp'],audio_bytes))
                # print(f"recieveing length {len(audio_data[f"{data['node_id']}_{data['username']}"])}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Connection closed: {websocket.remote_address}")
        # with open("output.json", "w") as json_file:
        #     json.dump(json_data, json_file)

async def repeated_task():
    while True:
        for user in audio_data:
            audio_buffer = bytearray()
            # print(f"Audio buffer length {len(audio_buffer)}")
            audio_chunks = audio_data[user][:]
            timestamps = []  # Track timestamps for the buffer

            print(f"chunks for transcription {len(audio_data[user])}")
            for timestamp,audio in audio_chunks:
                audio_buffer.extend(audio)  # Append chunk data
                timestamps.append(timestamp)  # Store timestamp
                # If buffer reaches 5 seconds, process it
                while len(audio_buffer) >= CHUNK_SIZE:
                    pcm_chunk = audio_buffer[:CHUNK_SIZE]  # Take 5 sec chunk
                    audio_buffer = audio_buffer[CHUNK_SIZE:]  # Remove processed data
                    wav_file = process_chunk(pcm_chunk)  # Convert PCM to WAV
                    chunk_time = timestamps.pop(0)
                    audio_queue.put((wav_file, user,chunk_time))
                audio_data[user].remove((timestamp,audio))

            print(f"after deleting the chunks {len(audio_data[user])}")

            if len(audio_buffer) > 0:
                # print(f"Left out chunks {len(audio_buffer)}")
                wav_file = process_chunk(audio_buffer)  # Convert remaining PCM to WAV
                chunk_time = timestamps.pop(0)
                audio_queue.put((wav_file, user,timestamp))  # Send to Whisper
                audio_buffer.clear()  # Clear the buffer after processing
            # print(f"Audio buffer length {len(audio_buffer)}")

        await asyncio.sleep(10)  # Sleep for 10 seconds before running the task again


process = multiprocessing.Process(target=transcriber_process, args=(audio_queue, result_queue))
process.start()

async def main():
    server = await websockets.serve(echo, "0.0.0.0", 8001)
    await repeated_task()
    print("✅ Server started on ws://localhost:8000")
    await server.wait_closed()

asyncio.run(main())
