import asyncio
import websockets
import numpy as np
import json
import sounddevice as sd
import numpy as np
import time

silence_threshold = 500  # Adjust based on noise levels
silence_start = None  # Track when silence begins
silence_duration = 0  # Track silence length


silence_threshold = 500  # Adjust based on actual noise level
silence_start = None
silence_duration = 0

last_silence_notification = 0  # Track last silence notification timestamp
silence_notification_interval = 60  # Send notification every 60 seconds (1 min)


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
    print(f"New connection: {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):  
                pass
                # node_id = struct.unpack('!I', message[:4])[0]  # '!I' means big-endian unsigned int
                
                # # Rest is audio data
                # audio_data = message[4:]
                
                # print(f"Received data from node {node_id}, audio data size: {len(audio_data)}")
                # await detect_silence(message,websocket)  # Process without delay
            else:
                data = json.loads(message)  # Parse JSON
                node_id = data["node_id"]
                audio_base64 = data["audio"]
                # Decode Base64 audio
                audio_bytes = base64.b64decode(audio_base64)
                print(f"📥 Received {len(audio_bytes)} bytes of audio from Node {node_id}")
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Connection closed: {websocket.remote_address}")

async def main():
    server = await websockets.serve(echo, "0.0.0.0", 8000)
    print("✅ Server started on ws://localhost:8000")
    await server.wait_closed()

asyncio.run(main())
