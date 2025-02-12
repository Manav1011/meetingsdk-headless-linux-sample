import json
import base64
import soundfile as sf
import numpy as np
import librosa
import io


new_data = {'audio':''}

with open('output.json') as file:
    data = json.loads(file.read())
target_rate = 16000
sample_rate = 32000
sample_width =2
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}


merged_pcm = b"".join(base64.b64decode(obj['audio']) for obj in data)

# Convert PCM bytes to NumPy array
audio_np = np.frombuffer(merged_pcm, dtype=dtype_map[sample_width]).astype(np.float32)
audio_np /= np.iinfo(dtype_map[sample_width]).max  # Normalize to [-1, 1]
# Resample to 16,000 Hz for Whisper (if needed)
if sample_rate != target_rate:
    audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=target_rate, res_type="kaiser_best")
# Save as WAV in memory
wav_io = io.BytesIO()
sf.write(wav_io, audio_np, target_rate, format="WAV", subtype="PCM_16")
wav_io.seek(0)  # Reset pointer

with open("merged_audio.wav", "wb") as f:
    f.write(wav_io.read())