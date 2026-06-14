import asyncio
import wave
import numpy as np
import speech_recognition as sr
import os
import time
import io
from gtts import gTTS
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import static_ffmpeg

static_ffmpeg.add_paths()

# ================= KÜRESEL AYARLAR =================
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 300   # Biraz artırdık (gürültü için)
SILENCE_DURATION = 1.5

is_processing = False

app = FastAPI()

# ================= YARDIMCI FONKSİYONLAR =================

def generate_ai_response(user_text):
    user_text = user_text.lower()
    if "merhaba" in user_text or "selam" in user_text:
        return "Merhaba! Sana nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_text:
        return "Harikayım, teşekkür ederim! Sen nasılsın?"
    elif "ışığı aç" in user_text:
        return "Anlaşıldı, ışıkları hemen açıyorum."
    else:
        return "Dediğini duydum ama henüz bunu yapamıyorum."

def process_audio_in_memory(raw_bytes):
    """Diske yazmadan bellekte ses işleme."""
    mem_file = io.BytesIO()
    with wave.open(mem_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    mem_file.seek(0)
    return mem_file

async def handle_response_task(websocket: WebSocket, audio_to_process: bytes):
    global is_processing
    try:
        audio_stream = process_audio_in_memory(audio_to_process)
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_stream) as source:
            audio_data = recognizer.record(source)
            # Google STT
            user_text = recognizer.recognize_google(audio_data, language="tr-TR")
            print(f"\n🗣️ SİZ: {user_text}")

            ai_reply = generate_ai_response(user_text)
            print(f"🤖 ASİSTAN: {ai_reply}")

            # TTS ve Streaming
            tts = gTTS(text=ai_reply, lang='tr', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # ESP32'nin istediği formata çevir
            audio = AudioSegment.from_file(fp, format="mp3")
            pcm_data = audio.set_frame_rate(16000).set_channels(1).raw_data
            
            print("📤 Cevap sesi aktarılıyor...")
            chunk_size = 2048
            for i in range(0, len(pcm_data), chunk_size):
                await websocket.send_bytes(pcm_data[i:i+chunk_size])
                await asyncio.sleep(0.02)
            print("✅ Cevap iletildi.")

    except Exception as e:
        print(f"\n⚠️ İşlem hatası: {e}")
    finally:
        is_processing = False

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n⚡ ESP32 Bağlandı!")
    
    global is_processing
    audio_buffer = bytearray()
    is_speaking = False
    silence_start_time = None
    
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            
            if is_processing: continue
            
            energy = int(np.std(np.frombuffer(audio_data, dtype=np.int16)))
            
            if energy > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    audio_buffer = bytearray()
                audio_buffer.extend(audio_data)
                silence_start_time = None
            else:
                if is_speaking:
                    audio_buffer.extend(audio_data)
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_DURATION:
                        print("\n⏱️ Sessizlik algılandı, işleniyor...")
                        is_processing = True
                        asyncio.create_task(handle_response_task(websocket, bytes(audio_buffer)))
                        is_speaking = False
                        audio_buffer = bytearray()
                        
    except WebSocketDisconnect:
        print("\n🔌 Bağlantı koptu.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
