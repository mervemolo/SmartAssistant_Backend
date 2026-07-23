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

# =========================
# GROQ AI AYARLARI
# =========================
from openai import AsyncOpenAI

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if GROQ_API_KEY:
    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )
else:
    client = None
    print("⚠️ UYARI: GROQ_API_KEY bulunamadı!")

# =========================
# AYARLAR
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

app = FastAPI()
is_processing = False

# Global Sensör Belleği
ev_durumu = "Sıcaklık ve nem normal düzeyde, hareket algılanmadı."

# Sohbet Hafızası & 5 Dk Zamanlayıcı Değişkenleri
chat_history = []          
last_interaction_time = 0  
MEMORY_TIMEOUT = 300       

# =========================
# AI CEVAP (GROQ / Llama 3)
# =========================
async def generate_ai_response(text):
    global chat_history, last_interaction_time
    if not client:
        return "Üzgünüm, yapay zeka anahtarı ayarlanmamış."
        
    today = datetime.now().strftime("%d %B %Y")
    current_time = time.time()
    
    # 5 dakika hareketsizlik kontrolü
    if last_interaction_time > 0 and (current_time - last_interaction_time > MEMORY_TIMEOUT):
        chat_history = []
        print("🧹 5 dakika hareketsizlik algılandı: Sohbet hafızası tamamen silindi.")
        
    last_interaction_time = current_time
    
    # 🧠 DÜZELTİLEN YENİ SYSTEM PROMPT (NET VE KOŞULLU EMİRLER)
    system_prompt = {
        "role": "system", 
        "content": (
            f"Sen Merve'nin ESP32 tabanlı akıllı ev asistanısın. Bugünün tarihi: {today}. "
            f"Evin anlık sensör durum raporu: {ev_durumu}. "
            "\n\nKURALLAR:\n"
            "1. EĞER kullanıcı sana odanın sıcaklığını, nemini, ışığını veya sensör değerlerini doğrudan SORARSA (Örn: 'kaç derece?', 'nem nasıl?'), "
            "sana verilen durum raporundaki bilgileri kullanarak DOĞRU VE AÇIK BİR CEVAP VER.\n"
            "2. EĞER kullanıcı sensörlerle ilgili bir şey sormuyorsa (normal sohbet ediyorsa, selam veriyorsa), "
            "cümlelerine durup dururken sıcaklık veya nem bilgilerini ASLA EKLEME. Lafı gereksiz yere sensörlere getirme.\n"
            "3. Sesli asistan olduğun için yanıtların her zaman samimi, doğal ve maksimum 1 veya 2 kısa cümle olmalı."
        )
    }
    
    messages = [system_prompt]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": text})
    
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        clean_text = answer.replace("*", "").replace("#", "").replace("_", "")
        
        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": clean_text})
        
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
            
        return clean_text
        
    except Exception as e:
        print(f"⚠️ AI Hatası: {e}")
        return "Üzgünüm, şu an bağlantı kuramıyorum."


# =========================
# PCM WAV OLUŞTUR
# =========================
def create_wav(raw_audio):
    mem = io.BytesIO()
    with wave.open(mem, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio)
    mem.seek(0)
    return mem


# =========================
# TTS (SES YÜKSELTME İLE)
# =========================
def create_tts(text):
    mp3 = io.BytesIO()
    tts = gTTS(
        text=text,
        lang="tr",
        slow=False
    )
    tts.write_to_fp(mp3)
    mp3.seek(0)

    audio = AudioSegment.from_file(mp3, format="mp3")
    audio = audio + 8  
    
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    return audio.raw_data


# =========================
# SES İŞLEME
# =========================
async def process_audio(websocket, raw_audio):
    global is_processing

    try:
        wav_file = create_wav(raw_audio)
        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(
                audio,
                language="tr-TR"
            )
            print(f"\n🗣️ SİZ: {text}")

        except sr.UnknownValueError:
            print("❌ Ses anlaşılamadı")
            return

        answer = await generate_ai_response(text)
        print(f"🤖 ASİSTAN: {answer}")

        pcm = create_tts(answer)
        print("📤 Ses gönderiliyor...")

        chunk_size = 2048
        for i in range(0, len(pcm), chunk_size):
            await websocket.send_bytes(pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)

        await websocket.send_text("STOP")
        print("✅ Cevap tamamlandı")

    except Exception as e:
        print(f"⚠️ İşlem hatası: {e}")

    finally:
        is_processing = False
        print("\n🎤 Dinleniyor...")


# =========================
# WEBSOCKET
# =========================
@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    global is_processing, ev_durumu
    
    print("DEBUG: Yeni bir bağlantı isteği geldi!")
    await websocket.accept()
    print("⚡ ESP32 Bağlandı!")

    audio_buffer = bytearray()
    is_speaking = False
    silence_start = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    data_str = message["text"]
                    if data_str != "STOP":
                        veri = json.loads(data_str)
                        
                        temp = veri.get("sicaklik", 0.0)
                        hum = veri.get("nem", 0.0)
                        light = veri.get("isik", "GUNDUZ")
                        motion = veri.get("hareket", "YOK")
                        
                        ev_durumu = f"Oda sıcaklığı {temp} derece, nem oranı yüzde {hum}. Şu an ortam durumu: {light}. Odada hareket durumu: {motion}."
                        print(f"📊 Sensör Güncellendi -> Sıcaklık: {temp}°C | Nem: %{hum} | Işık: {light} | Hareket: {motion}")
                except Exception as json_error:
                    print(f"⚠️ Sensör JSON ayrıştırma hatası: {json_error}")
                continue

            if "bytes" in message:
                if is_processing:
                    continue

                data = message["bytes"]
                samples = np.frombuffer(data, dtype=np.int16)
                if len(samples) == 0:
                    continue

                energy = int(np.std(samples))
                print(f"Ses: {energy}    ", end="\r")

                if energy > SILENCE_THRESHOLD:
                    if not is_speaking:
                        print("\n🎙️ Konuşma başladı")
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
                            print("\n⏱️ İşleniyor...")
                            is_processing = True
                            
                            asyncio.create_task(
                                process_audio(websocket, bytes(audio_buffer))
                            )
                            
                            audio_buffer.clear()
                            is_speaking = False

    except WebSocketDisconnect:
        print("\n🔌 ESP32 ayrıldı")
    except Exception as e:
        print(f"\n⚠️ WebSocket hata {e}")

@app.get("/")
def home():
    return {"status": "ESP32 AI Server Running on GROQ"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
