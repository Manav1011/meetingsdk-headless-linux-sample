import asyncio
import websockets
import numpy as np
import json
import base64
import sounddevice as sd
from collections import defaultdict
import time
import multiprocessing
import io
import soundfile as sf
import psycopg2
import psycopg2.extras  
from queue import Empty  # Fix queue exception handling
from faster_whisper import WhisperModel

# 🔧 Silence Detection Config
silence_threshold = 500  
silence_start = None  
silence_duration = 0  
last_silence_notification = 0  
silence_notification_interval = 60  

# 🔧 PCM Properties
SAMPLE_RATE = 32000  
SAMPLE_WIDTH = 2  
CHANNELS = 1  
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
CHUNK_SIZE = 20 * BYTES_PER_SECOND  

# Data storage
audio_data = defaultdict(list)  
audio_queue = multiprocessing.Queue()
result_queue = multiprocessing.Queue()

NUM_WORKERS = min(6, multiprocessing.cpu_count())  # Adjust based on CPU cores & memory

def process_chunk(pcm_data):
    """Convert PCM bytes to WAV and return as BytesIO"""
    wav_io = io.BytesIO()
    with sf.SoundFile(wav_io, mode='w', samplerate=SAMPLE_RATE, channels=CHANNELS, format="WAV") as file:
        file.write(memoryview(pcm_data).cast('h'))  # Write PCM as WAV
    wav_io.seek(0)  # Reset pointer
    return wav_io  # Ready for Whisper

def transcriber_process(audio_queue, result_queue):
    """Worker process that loads Whisper and transcribes audio chunks."""
    print(f"🔄 Worker {multiprocessing.current_process().name} loading Whisper model...")
    model = WhisperModel("tiny", compute_type="int8")  
    
    # ✅ PostgreSQL Connection
    try:
        conn = psycopg2.connect(
            dbname="dockertestdb",
            user="manav1011",
            password="Manav@1011",
            host="192.168.7.70",
            port=5432
        )
        cur = conn.cursor()
        cur.execute("""
            PREPARE insert_transcription AS 
            INSERT INTO zoom.transcripts (meeting_id, user_id, username, transcript, created_at) 
            VALUES ($1, $2, $3, $4, $5)
        """)
        conn.commit()
        print("✅ Database Connected")
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return  
    
    while True:
        try:
            item = audio_queue.get(timeout=5)
            if item is None:
                break  
            
            wav_file, user, chunk_time = item  
            segments, _ = model.transcribe(wav_file, language='en')
            transcript = " ".join(segment.text for segment in segments)

            if not transcript.strip():  # ✅ Handle Empty Transcriptions
                print("⚠️ Skipping empty transcription")
                continue
            print(f"Transcript: {transcript}")
            meeting_id, user_id, username = user.split("_")  
            cur.execute("EXECUTE insert_transcription (%s, %s, %s, %s, %s)", 
                        (meeting_id, user_id, username, transcript, chunk_time))
            conn.commit()
            print(f"✅ Transcription saved for {username} at {chunk_time}")
        except Empty:
            continue  
        except Exception as e:
            print(f"❌ Error in transcription process: {e}")
            conn.rollback()

async def detect_silence(audio_bytes, websocket):
    """Detect silence and notify the client if it exceeds 5 seconds."""
    global silence_start, silence_duration, last_silence_notification  

    if not audio_bytes:  
        print("⚠️ Warning: Empty audio frame received!")
        return  

    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))  

    if np.isnan(rms):
        print("⚠️ Warning: Computed RMS is NaN!")
        return  

    if rms < silence_threshold:
        if silence_start is None:
            silence_start = time.time()
        silence_duration = time.time() - silence_start

        if silence_duration >= 5:
            current_time = time.time()
            if last_silence_notification == 0 or (current_time - last_silence_notification >= silence_notification_interval):
                print("⚠️ Silence detected for more than 5 seconds!")
                await websocket.send("⚠️ Warning: Silence detected for more than 5 seconds!")
                last_silence_notification = current_time  
    else:
        silence_start = None  
        last_silence_notification = 0  

async def echo(websocket):
    print(f"🔗 New connection: {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):  
                pass
            else:
                data = json.loads(message)  
                audio_bytes = base64.b64decode(data['audio'])
                audio_data[f"{data['meeting_id']}_{data['node_id']}_{data['username']}"].append((data['timestamp'], audio_bytes))
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print(f"🔌 Connection closed: {websocket.remote_address}")

async def repeated_task():
    """Process and send audio chunks to transcriber queue."""
    while True:
        for user in audio_data:
            audio_buffer = bytearray()
            audio_chunks = audio_data[user][:]
            timestamps = []  

            print(f"📦 Chunks for transcription: {len(audio_data[user])}")
            for timestamp, audio in audio_chunks:
                audio_buffer.extend(audio)  
                timestamps.append(timestamp)  

                while len(audio_buffer) >= CHUNK_SIZE:
                    pcm_chunk = audio_buffer[:CHUNK_SIZE]  
                    audio_buffer = audio_buffer[CHUNK_SIZE:]  
                    wav_file = process_chunk(pcm_chunk)
                    chunk_time = timestamps.pop(0)
                    audio_queue.put((wav_file, user, chunk_time))  
                audio_data[user].remove((timestamp, audio))

            if len(audio_buffer) > 0:
                wav_file = process_chunk(audio_buffer)  
                chunk_time = timestamps.pop(0)
                audio_queue.put((wav_file, user, chunk_time))  
                audio_buffer.clear()  

        await asyncio.sleep(25)  

# ✅ Start multiple transcription worker processes
processes = []
for _ in range(NUM_WORKERS):
    p = multiprocessing.Process(target=transcriber_process, args=(audio_queue, result_queue))
    p.start()
    processes.append(p)

async def main():
    server = await websockets.serve(echo, "0.0.0.0", 8001)
    asyncio.create_task(repeated_task())  
    print("✅ Server started on ws://localhost:8001")
    await server.wait_closed()

asyncio.run(main())
