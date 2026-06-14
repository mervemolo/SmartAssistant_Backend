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
SILENCE_THRESHOLD = 500  # Eşiği biraz yükselttik, gürültüyü ses sanmasın
SILENCE_DURATION = 1.8

is_processing = False

app = FastAPI()

def generate_ai_response(user_text):
    user_text = user_text.lower()
    if "merhaba" in user_text or "selam" in user_text:
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
    global is_processing
    try:
        audio_stream = process_audio_in_memory(audio_to_process)
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_stream) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data, language="tr-TR")
            print(f"\n🗣️ SİZ: {user_text}")

            ai_reply = generate_ai_response(user_text)
            
            # Ses oluşturma
            tts = gTTS(text=ai_reply, lang='tr', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            audio = AudioSegment.from_file(fp, format="mp3")
            pcm_data = audio.set_frame_rate(16000).set_channels(1).raw_data
            
            # Streaming - Paketleri 2048'er byte gönder
            for i in range(0, len(pcm_data), 2048):
                await websocket.send_bytes(pcm_data[i:i+2048])
                await asyncio.sleep(0.01)
            await websocket.send_text("STOP")
            print("✅ Cevap iletildi ve STOP sinyali gönderildi.")
                
    except Exception as e:
        print(f"\n⚠️ İşlem hatası: {e}")
    finally:
        is_processing = False

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ ESP32 Bağlandı ve hazır!")
    
    global is_processing
    audio_buffer = bytearray()
    is_speaking = False
    silence_start_time = None
    
    try:
        while True:
            # Buradaki timeout, bağlantının kopmasını engeller
            audio_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=60.0)
            
            # --- DEBUG: Veri akıyor mu? ---
            # Render loglarında bu satırı görürsen sunucu veriyi alıyor demektir.
            # print(f"DEBUG: {len(audio_data)} byte alındı.", end="\r")

            if is_processing: continue
            
            data_chunk = np.frombuffer(audio_data, dtype=np.int16)
            if len(data_chunk) == 0: continue
            
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
                    asyncio.create_task(handle_response_task(websocket, bytes(audio_buffer)))
                    is_speaking = False
                    audio_buffer = bytearray()
                        
    except asyncio.TimeoutError:
        print("\n⏳ Bağlantı zaman aşımı (ping yok).")
    except WebSocketDisconnect:
        print("\n🔌 Bağlantı koptu.")
    except Exception as e:
        print(f"\n⚠️ Hata: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
