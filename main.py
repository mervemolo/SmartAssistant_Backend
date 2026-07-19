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

# =========================
# GEMINI AI AYARLARI
# =========================
import google.generativeai as genai

# Ortam değişkeninden API anahtarını çekiyoruz
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ UYARI: GEMINI_API_KEY bulunamadı! Lütfen ortam değişkenlerini kontrol edin.")

# Hızlı yanıt için flash modelini kullanıyoruz
model = genai.GenerativeModel("gemini-2.0-flash")


# =========================
# AYARLAR
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

app = FastAPI()

# Aynı anda tek konuşma
is_processing = False


# =========================
# AI CEVAP (GÜNCELLENDİ)
# =========================
async def generate_ai_response(text):
    try:
        # Asistana kim olduğunu ve nasıl yanıt vereceğini belirten sistem komutu (Prompt)
        prompt = (
            "Sen Mehmet'in ESP32 tabanlı akıllı ev asistanısın. "
            "Yanıtların her zaman çok kısa, öz ve günlük konuşma dilinde olmalı. "
            "Sesli asistan olduğun için uzun listeler veya karmaşık cümleler kurma. "
            "Maksimum 1 veya 2 cümle ile cevap ver. "
            f"Kullanıcının söylediği: {text}"
        )
        
        # Gemini'den asenkron (beklemesiz) olarak yanıt alıyoruz
        response = await model.generate_content_async(prompt)
        
        # Gelen yanıttaki markdown işaretlerini temizle (TTS'in garip sesler çıkarmasını önler)
        clean_text = response.text.replace("*", "").replace("#", "").replace("_", "")
        return clean_text
        
    except Exception as e:
        print(f"⚠️ AI Hatası: {e}")
        return "Üzgünüm, şu an internet bağlantısı kuramıyorum."


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
# TTS
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
    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    return audio.raw_data


# =========================
# SES İŞLEME (GÜNCELLENDİ)
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

        # YENİ: Await eklenerek asenkron AI fonksiyonu çağrılıyor
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
    global is_processing

    await websocket.accept()
    print("⚡ ESP32 Bağlandı!")

    audio_buffer = bytearray()
    is_speaking = False
    silence_start = None

    try:
        while True:
            data = await websocket.receive_bytes()

            if is_processing:
                continue

            samples = np.frombuffer(data, dtype=np.int16)
            if len(samples) == 0:
                continue

            energy = int(np.std(samples))
            print(f"Ses: {energy}    ", end="\r")

            # konuşma
            if energy > SILENCE_THRESHOLD:
                if not is_speaking:
                    print("\n🎙️ Konuşma başladı")
                    audio_buffer.clear()
                    is_speaking = True
                
                audio_buffer.extend(data)
                silence_start = None

            # sessizlik
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


# =========================
# RENDER HEALTH
# =========================
@app.get("/")
def home():
    return {"status": "ESP32 AI Server Running"}


# =========================
# START
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
