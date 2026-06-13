import asyncio
import websockets
import wave
import numpy as np
import speech_recognition as sr
import os
import time
import io
from gtts import gTTS
from pydub import AudioSegment
import http

import static_ffmpeg
static_ffmpeg.add_paths()

# ================= KÜRESEL AYARLAR =================
SAMPLE_RATE = 16000
CHANNELS = 1
# Ses algılama eşiğini 100 civarına çekmek ESP32 verisiyle daha uyumlu çalışacaktır
SILENCE_THRESHOLD = 100    
SILENCE_DURATION = 1.5

is_processing = False

def generate_ai_response(user_text):
    user_text = user_text.lower()
    if "merhaba" in user_text or "selam" in user_text:
        return "Merhaba! Sana nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_text:
        return "Harikayım, teşekkür ederim! Sen nasılsın?"
    elif "kimsin" in user_text:
        return "Ben senin ESP32 tabanlı akıllı asistanınım."
    elif "ışığı aç" in user_text:
        return "Anlaşıldı, ışıkları hemen açıyorum."
    else:
        return f"{user_text} dediğini duydum ama bunu henüz yapamıyorum."

def generate_tts_audio(text):
    filename = "cevap.wav"
    tts = gTTS(text=text, lang='tr', slow=False)
    mp3_data = io.BytesIO()
    tts.write_to_fp(mp3_data)
    mp3_data.seek(0)
    sound = AudioSegment.from_file(mp3_data, format="mp3")
    sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    sound.export(filename, format="wav")
    return filename

def prepare_audio_for_esp32(wav_filename):
    with wave.open(wav_filename, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        return frames

def process_and_save_audio(raw_bytes):
    if len(raw_bytes) == 0: return None
    filename = "canli_kayit.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    return filename

def convert_speech_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data, language="tr-TR")
        except:
            return None

async def handle_response_task(websocket, audio_to_process):
    global is_processing
    try:
        wav_file = process_and_save_audio(audio_to_process)
        user_text = convert_speech_to_text(wav_file)
        if user_text:
            ai_reply = generate_ai_response(user_text)
            reply_wav = generate_tts_audio(ai_reply)
            raw_pcm_bytes = prepare_audio_for_esp32(reply_wav)
            
            # Ses parçalar halinde gönderilir
            chunk_size = 4096
            for i in range(0, len(raw_pcm_bytes), chunk_size):
                await websocket.send(raw_pcm_bytes[i:i+chunk_size])
                await asyncio.sleep(0.02)
    finally:
        is_processing = False

async def health_check(path, request_headers):
    # WebSocket el sıkışması dışındaki istekleri (HEAD/GET) yakala
    if "upgrade" not in request_headers.get("Connection", "").lower():
        return http.HTTPStatus.OK, [], b"OK\n"
    return None

async def audio_stream_handler(websocket):
    global is_processing
    audio_buffer = bytearray()
    is_speaking = False
    silence_start = None

    try:
        async for audio_data in websocket:
            if is_processing: continue
            
            data_chunk = np.frombuffer(audio_data, dtype=np.int16)
            energy = int(np.std(data_chunk))
            
            if energy > SILENCE_THRESHOLD:
                is_speaking = True
                audio_buffer.extend(audio_data)
                silence_start = None
            elif is_speaking:
                audio_buffer.extend(audio_data)
                if silence_start is None: silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    is_processing = True
                    asyncio.create_task(handle_response_task(websocket, bytes(audio_buffer)))
                    audio_buffer.clear()
                    is_speaking = False
                    silence_start = None
    except:
        pass

async def main():
    PORT = int(os.environ.get("PORT", 10000))
    async with websockets.serve(
        audio_stream_handler, "0.0.0.0", PORT,
        process_request=health_check
    ):
        print(f"🚀 Sunucu {PORT} portunda yayında.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
