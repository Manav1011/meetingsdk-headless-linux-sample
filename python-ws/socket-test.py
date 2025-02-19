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
import psycopg2
import psycopg2.extras
import google.generativeai as genai
genai.configure(api_key="AIzaSyD0vLS27RzUfBTiFb3lv4tbxsS10BL3Sio")
text_model = genai.GenerativeModel("gemini-1.5-flash-8b")
import json
import re

# 🔧 Silence Detection Config
silence_threshold = 2000  
silence_start = None  
silence_duration = 0  
last_silence_notification = 0  
silence_notification_interval = 60*10

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

active_connections = dict()
waiting_connections = dict()
silence_durations = dict()

def split_into_chunks(text, chunk_size=5000, tolerance=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # If there's more text left and we haven't exceeded the tolerance
        if end < len(text) and (end + tolerance) < len(text):
            end = text.rfind(' ', start, end)  # Try to split at a space
            if end == -1:  
                end = min(start + chunk_size, len(text))  # Fallback to hard cut

        chunks.append(text[start:end])
        start = end
    return chunks

async def start_sdk(join_url,meeting_id):
    reader, writer = await asyncio.open_connection('192.168.7.70', 9001)
    command = {"action": "start_sdk","join_url":join_url,'meeting_id':meeting_id}
    writer.write(json.dumps(command).encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()

def generate_icebreakers():
    messages = [
        {
            "role": "system",
            "content": """
                - You are a **meeting facilitator** ensuring discussions stay engaging and on track.
                - Generate **5 creative icebreaker questions** when the meeting becomes unproductive or silent.
                - The questions should be **fun, thought-provoking, or contextually relevant**, encouraging participation.
                - Keep a **friendly, professional tone**, avoiding sensitive or controversial topics.
                - Format output in **beautiful Markdown** with emojis for engagement:
                ```
                ## 🎤 Icebreaker Questions  
                - 🤔 Question 1: [Text]  
                - 💡 Question 2: [Text]  
                - 🎭 Question 3: [Text]  
                - 🌍 Question 4: [Text]  
                - 🚀 Question 5: [Text]  
                ```
            """
        },
        {"role": "system", "content": "temperature: 0.7"},
        {"role": "user", "content": "Generate 5 icebreaker questions in Markdown format."}
    ]
    response = text_model.generate_content(json.dumps(messages))
    return response.text



def query_llm(chunk):
    """
    Sends conversation chunks to the LLM for summarization, ensuring context is preserved 
    even if the order of user turns varies.
    """
    
    # Combine chunks into a single context strin

    # Construct system and user messages
    messages = [
        {
        "role": "system",
        "content": """
            - You are a professional meeting assistant tasked with summarizing the meeting **up to the point** when the user presses the "Get Summary" button.
            - Focus on extracting the most importa
            nt discussion points, decisions, and action items from the meeting so far, while excluding minor details or off-topic conversation.
            - Your summary should include the following sections:
            - **Meeting Summary - [Date]**
            - **Attendees:** List of participants.
            - **Key Discussion Points:** Bullet points summarizing the key topics discussed up until now. Focus on high-level takeaways.
            - **Decisions:** Bullet points listing key decisions made during the meeting.
            - **Action Items:** Bullet points specifying tasks assigned to individuals, clearly labeled with responsible person(s).
            - Use **bold** for key topics, decisions, and action items.
            - Keep the tone professional and concise, suitable for business documentation.
            - Ensure the **date** is automatically filled in based on the provided input.
            - The summary should be generated **dynamically as the meeting progresses**, but finalized only when the user requests the summary.
            - If any part of the meeting is in a language other than English, please **translate it into English** in the summary.
        """
        },
        {"role": "system", "content": "temperature: 0"},  # Higher temperature for better context inference
        {"role": "user", "content": chunk}
    ]

    # Generate response from LLM
    response = text_model.generate_content(json.dumps(messages))
    return response.text
    # Extract JSON from response
    # match = re.search(r"\{.*\}", response.text, re.DOTALL)  # Matches curly braces and content inside
    # if match:
    #     json_string = match.group(0)
    # else:
    #     print("No JSON found in response.")
    #     return None  

    # # Parse JSON safely
    # try:
    #     data = json.loads(json_string)
    #     return data
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding JSON: {e}")
    #     print(f"Raw response: {json_string}")  # Debugging info
    #     return None  

# merge summary for next process
def generate_final_summary(summaries, recursion_depth=0, max_recursion=10):
    print(f"Recursion Depth : {recursion_depth}")
    final_summary = ''
    if len(summaries) == 1:
        final_summary = summaries[0]
        return final_summary

    if recursion_depth >= max_recursion:
        combined_summary = ' '.join(summaries)
        response = query_llm([combined_summary])
        return response

    concated_summary = ' '.join(summaries)
    sub_summaries = split_into_chunks(concated_summary)
    if len(sub_summaries) == 1:
        response = query_llm(sub_summaries[0])
        final_summary = response
        return final_summary
    new_summaries = []
    for summary in sub_summaries:
        response = query_llm(summary)
        new_summaries.append(response)
    return generate_final_summary(new_summaries, recursion_depth + 1)

def generate_summary_till_now(meeting_id):
    conn = psycopg2.connect(
                dbname="dockertestdb",
                user="manav1011",
                password="Manav@1011",
                host="192.168.7.70",
                port=5432
    )

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = """
        SELECT username, transcript, created_at 
        FROM zoom.transcripts 
        WHERE meeting_id = %s 
        AND created_at < NOW()
        ORDER BY created_at ASC;
    """

    cur.execute(query, (meeting_id,))
    records = cur.fetchall()
    conn.close()
    # Format results as JSON
    transcripts = [
        {
            "username": row[0],
            "transcript": row[1],
            "timestamp":row[2]
        }
        for row in records
    ]

    # make whole conversation
    conversation = ''
    for transcript in transcripts:
        text = f"{transcript['timestamp']} - {transcript['username']} : {transcript['transcript']} \n"
        conversation+=text
    if len(conversation.strip()) <= 0:
        print('No conversation provided')
        return
    # Now the chunking of 5000 tokens each
    chunks = split_into_chunks(conversation)
    
    summaries = []
    for chunk in chunks:
        response = query_llm(chunk)
        summaries.append(response)
    final_summary = generate_final_summary(summaries)
    return final_summary

def generate_summary(meeting_id):
    conn = psycopg2.connect(
                dbname="dockertestdb",
                user="manav1011",
                password="Manav@1011",
                host="192.168.7.70",
                port=5432
    )

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = """
        SELECT username,transcript,created_at 
        FROM zoom.transcripts 
        WHERE meeting_id = %s 
        ORDER BY created_at ASC;
    """

    cur.execute(query, (meeting_id,))
    records = cur.fetchall()
    conn.close()

    # Format results as JSON
    transcripts = [
        {
            "username": row[0],
            "transcript": row[1],
            "timestamp":row[2]
        }
        for row in records
    ]

    # make whole conversation
    conversation = ''
    for transcript in transcripts:
        text = f"{transcript['timestamp']} - {transcript['username']} : {transcript['transcript']} \n"
        conversation+=text
    if len(conversation.strip()) <= 0:
        print('No conversation provided')
        return
    # Now the chunking of 5000 tokens each
    chunks = split_into_chunks(conversation)

    summaries = []
    for chunk in chunks:
        response = query_llm(chunk)
        summaries.append(response)

    final_summary = generate_final_summary(summaries)
    print(final_summary)
    # return final_summary

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
            segments, _ = model.transcribe(wav_file)
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

async def detect_silence(meeting_id,audio_bytes,silence_duration_preset):
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
        if silence_duration >= int(silence_duration_preset):
            current_time = time.time()
            if last_silence_notification == 0 or (current_time - last_silence_notification >= silence_notification_interval):
                print(f"⚠️ Silence detected for more than {silence_duration_preset} seconds!")
                if meeting_id in active_connections:
                    clients = [active_connections[meeting_id].get('host',None)] + active_connections[meeting_id].get('participants',[])
                    if len(clients) > 0:
                        # generate_icebreakers
                        excercise = generate_icebreakers()
                        message = {
                            'action': 'notify',
                            'message': f'📢 **Silence Detected!** ⏳ Duration: {silence_duration_preset} seconds',
                            'excercise':excercise
                        }
                        for client in clients:
                            try:
                                if client:
                                    await client.send(json.dumps(message))
                            except Exception as e:
                                print(f"⚠️ Error sending message to a client: {e}")
                last_silence_notification = current_time  
    else:
        silence_start = None  
        last_silence_notification = 0  

class EchoServerProtocol:
    def connection_made(self, transport):
        self.transport = transport
        print("Connection established")

    def datagram_received(self, data, addr):
        try:
            message = data.decode()
            asyncio.create_task(self.handle_datagram(message, addr))
        except Exception as e:
            print(f"Error decoding message: {e}")

    async def handle_datagram(self, message, addr):
        data = json.loads(message)
        if data['action'] == 'stream_mixed':
            audio_bytes = base64.b64decode(data['audio'])
            await detect_silence(data['meeting_id'],audio_bytes,silence_durations[data['meeting_id']])
        if data['action'] == 'stream_individual':
            audio_bytes = base64.b64decode(data['audio'])
            audio_data[f"{data['meeting_id']}_{data['node_id']}_{data['username']}"].append((data['timestamp'], audio_bytes))

def remove_connection(connection):
    global active_connections

    keys_to_remove = []

    for meeting_id, connections in list(active_connections.items()):
        if isinstance(connections, dict):  
            # Remove from 'participants' list if present
            if 'participants' in connections and connection in connections['participants']:
                connections['participants'].remove(connection)
                if not connections['participants']:  # Remove key if empty
                    del connections['participants']

            # Remove if connection is stored directly under 'host' or 'bot'
            if connections.get('host') == connection:
                del connections['host']
            if connections.get('bot') == connection:
                del connections['bot']

            # If the meeting ID has no remaining connections, mark it for removal
            if not connections:
                keys_to_remove.append(meeting_id)

    # Remove empty meeting IDs
    for meeting_id in keys_to_remove:
        del active_connections[meeting_id]


async def echo(websocket):
    global active_connections
    # active_connections.add(websocket)
    print(f"🔗 New connection: {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, bytes):  
                pass
            else:
                data = json.loads(message)
                if data['action'] == 'connection':
                    meeting_id = data['meeting_id']
                    if meeting_id not in active_connections:
                        active_connections[meeting_id] = {}
                        if data['user'] == 'host':
                            # here we have to start the zoom bot
                            active_connections[meeting_id]['host'] = websocket
                            # add the silence duration
                            silence_durations[meeting_id] = data['silence_duration']
                            # add the participatns
                            if meeting_id in waiting_connections:
                                active_connections[meeting_id]['participants'] = []
                                for participant in waitning_connections[meeting_id]:
                                    active_connections[meeting_id]['participants'].append(participant)
                                del waitning_connections[meeting_id]
                            # start the sdk
                            join_url = data['join_url']
                            await start_sdk(join_url,meeting_id)
                        else:
                            # add them to waiting area
                            if meeting_id not in waiting_connections:
                                waiting_connections[meeting_id] = [websocket]
                            else:
                                waiting_connections[meeting_id].append(websocket)
                    else:
                        # if data['user'] == 'ZoomBot':
                        #     active_connections[meeting_id]['bot'] = websocket
                        if data['user'] == 'participant':
                            if 'participants' not in active_connections[meeting_id]:
                                active_connections[meeting_id]['participants'] = [websocket]
                            else:
                                active_connections[meeting_id]['participants'].append(websocket)
                        # if data['user'] == 'host':
                        #     active_connections[meeting_id]['host'] = websocket
                        # active_connections[meeting_id].append(websocket)
                if data['action'] == 'meeting_ended':
                    # here we have to end the audio buffer for meeting
                    del active_connections[meeting_id]
                    asyncio.create_task(process_leftout_buffer(meeting_id=data['meeting_id']))
                    summary_process = multiprocessing.Process(target=generate_summary, args=(data['meeting_id'],))
                    summary_process.start()
                if data['action'] == 'get_summary':
                    await get_summary_till_now(meeting_id=data['meetingNumber'],websocket=websocket)
                    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        remove_connection(websocket)
        print(active_connections)
        print(f"🔌 Connection closed: {websocket.remote_address}")

async def get_summary_till_now(meeting_id,websocket):
    final_summary = generate_summary_till_now(meeting_id)
    message = {'action':'notify','message':final_summary}
    if meeting_id in active_connections:
        clients = [active_connections[meeting_id].get('host',None)] + active_connections[meeting_id].get('participants',[])
        for client in clients:
            try:
                if client:
                    await client.send(json.dumps(message))
            except Exception as e:
                print(f"⚠️ Error sending message to a client: {e}")


async def process_leftout_buffer(meeting_id):
    for user in audio_data:
        if not meeting_id in user:
            continue
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
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(EchoServerProtocol,local_addr=('192.168.7.195', 8080))
    server = await websockets.serve(echo, "0.0.0.0", 8001)
    asyncio.create_task(repeated_task())  
    print("✅ Server started on ws://localhost:8001")
    await server.wait_closed()

asyncio.run(main())
