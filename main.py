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
SILENCE_THRESHOLD = 500  # Gürültü eşiği
SILENCE_DURATION = 1.5    # Daha hızlı tepki için 1.5 saniye

app = FastAPI()

def generate_ai_response(user_text):
    user_text = user_text.lower()
    if any(word in user_text for word in ["merhaba", "selam"]):
        return "Merhaba! Sana nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_text:
        return "Harikayım, teşekkür ederim! Sen nasılsın?"
    return "Dediğini duydum ama henüz bunu yapamıyorum."

def process_audio_in_memory(raw_bytes):
    mem_file = io.BytesIO()
    with wave.open(mem_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    mem_file.seek(0)
    return mem_file

async def handle_response_task(websocket: WebSocket, audio_to_process: bytes):
    try:
        audio_stream = process_audio_in_memory(audio_to_process)
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_stream) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data, language="tr-TR")
            print(f"\n🗣️ SİZ: {user_text}")

            ai_reply = generate_ai_response(user_text)
            
            # TTS işlemi
            tts = gTTS(text=ai_reply, lang='tr', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            # Ses formatını ESP32'nin istediği PCM (16kHz, 1 kanal) formatına getir
            audio = AudioSegment.from_file(fp, format="mp3")
            pcm_data = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).raw_data
            
            # Cevabı parçalar halinde gönder
            for i in range(0, len(pcm_data), 2048):
                await websocket.send_bytes(pcm_data[i:i+2048])
                await asyncio.sleep(0.01)
            
            # Konuşma bitti sinyali
            await websocket.send_text("STOP")
            print("✅ Cevap iletildi ve STOP sinyali gönderildi.")
                
    except Exception as e:
        print(f"\n⚠️ İşlem hatası: {e}")

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ ESP32 Bağlandı!")
    
    is_processing = False
    audio_buffer = bytearray()
    is_speaking = False
    silence_start_time = None
    
    try:
        while True:
            # WebSocket'ten veri al
            audio_data = await websocket.receive_bytes()
            
            # --- DEBUG: Veri akışını izlemek için ---
            # print(f"DEBUG: {len(audio_data)} byte alındı.", end="\r")

            if is_processing: continue
            
            data_chunk = np.frombuffer(audio_data, dtype=np.int16)
            energy = int(np.std(data_chunk))
            
            if energy > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    audio_buffer = bytearray()
                audio_buffer.extend(audio_data)
                silence_start_time = None
            elif is_speaking:
                audio_buffer.extend(audio_data)
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_DURATION:
                    print("\n⏱️ Sessizlik algılandı, işlem başlıyor...")
                    is_processing = True
                    # İşlemi arka planda başlat (await etme ki loop devam etsin)
                    asyncio.create_task(handle_response_task(websocket, bytes(audio_buffer)))
                    is_speaking = False
                    audio_buffer = bytearray()
                    is_processing = False # Hemen serbest bırak
                        
    except WebSocketDisconnect:
        print("\n🔌 Bağlantı koptu.")
    except Exception as e:
        print(f"\n⚠️ WebSocket hatası: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
