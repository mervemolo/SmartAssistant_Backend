import asyncio
import wave
import numpy as np
import speech_recognition as sr
import os
import time
import io
import json
from datetime import datetime

from gtts import gTTS
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import static_ffmpeg

static_ffmpeg.add_paths()

from openai import AsyncOpenAI
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if GROQ_API_KEY:
    client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
else:
    client = None
    print("⚠️ UYARI: GROQ_API_KEY bulunamadı!")

SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

app = FastAPI()
is_processing = False

# Sensör verilerini tutacağımız global değişken
ev_durumu = "Sensör verisi henüz gelmedi."

async def generate_ai_response(text):
    if not client:
        return "Üzgünüm, yapay zeka anahtarı ayarlanmamış."
        
    today = datetime.now().strftime("%d %B %Y")
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        f"Sen Merve'nin ESP32 tabanlı akıllı ev asistanısın. "
                        f"Bugünün tarihi: {today}. "
                        f"Evin anlık durumu: {ev_durumu}. "
                        "Kullanıcı evle ilgili bir şey sorarsa (sıcaklık, nem, hareket, ışık) bu verileri kullanarak doğal bir şekilde cevap ver. "
                        "Yanıtların her zaman çok kısa, öz ve günlük konuşma dilinde olmalı. Maksimum 1-2 cümle kur."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        answer = response.choices[0].message.content
        return answer.replace("*", "").replace("#", "").replace("_", "")
    except Exception as e:
        print(f"⚠️ AI Hatası: {e}")
        return "Üzgünüm, bağlantı sorunu yaşıyorum."

def create_wav(raw_audio):
    mem = io.BytesIO()
    with wave.open(mem, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)
    mem.seek(0)
    return mem

def create_tts(text):
    mp3 = io.BytesIO()
    tts = gTTS(text=text, lang="tr", slow=False)
    tts.write_to_fp(mp3)
    mp3.seek(0)
    audio = AudioSegment.from_file(mp3, format="mp3")
    audio = audio + 8  
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return audio.raw_data

async def process_audio(websocket, raw_audio):
    global is_processing
    try:
        wav_file = create_wav(raw_audio)
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language="tr-TR")
            print(f"\n🗣️ SİZ: {text}")
        except sr.UnknownValueError:
            print("❌ Ses anlaşılamadı")
            return

        answer = await generate_ai_response(text)
        print(f"🤖 ASİSTAN: {answer}")

        pcm = create_tts(answer)
        chunk_size = 2048
        for i in range(0, len(pcm), chunk_size):
            await websocket.send_bytes(pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)

        await websocket.send_text("STOP")
    except Exception as e:
        print(f"⚠️ İşlem hatası: {e}")
    finally:
        is_processing = False
        print("\n🎤 Dinleniyor...")

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    global is_processing, ev_durumu
    await websocket.accept()
    print("⚡ ESP32 Bağlandı!")

    audio_buffer = bytearray()
    is_speaking = False
    silence_start = None

    try:
        while True:
            # Artık hem metin (sensör) hem de bayt (ses) alabiliyoruz
            message = await websocket.receive()
            
            if "text" in message:
                try:
                    veri = json.loads(message["text"])
                    ev_durumu = f"Sıcaklık {veri.get('sicaklik')} derece, Nem %{veri.get('nem')}. Ortam: {veri.get('isik')}. Hareket durumu: {veri.get('hareket')}."
                    print(f"📊 Sensör Güncellendi: {ev_durumu}")
                except:
                    pass
                continue

            if "bytes" in message:
                if is_processing:
                    continue
                
                data = message["bytes"]
                samples = np.frombuffer(data, dtype=np.int16)
                if len(samples) == 0: continue

                energy = int(np.std(samples))
                
                if energy > SILENCE_THRESHOLD:
                    if not is_speaking:
                        audio_buffer.clear()
                        is_speaking = True
                    audio_buffer.extend(data)
                    silence_start = None
                else:
                    if is_speaking:
                        audio_buffer.extend(data)
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > SILENCE_DURATION:
                            is_processing = True
                            asyncio.create_task(process_audio(websocket, bytes(audio_buffer)))
                            audio_buffer.clear()
                            is_speaking = False

    except WebSocketDisconnect:
        print("\n🔌 ESP32 ayrıldı")
    except Exception as e:
        print(f"\n⚠️ WebSocket hata {e}")

@app.get("/")
def home():
    return {"status": "ESP32 AI Server Running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
