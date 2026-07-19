import asyncio
import wave
import numpy as np
import speech_recognition as sr
import os
import time
import io
import google.generativeai as genai

from gtts import gTTS
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import static_ffmpeg

static_ffmpeg.add_paths()

# =========================
# AYARLAR & AI KURULUMU
# =========================
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5

# Gemini API Kurulumu
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()
is_processing = False

# =========================
# AI CEVAP (GEMINI)
# =========================

async def generate_ai_response(text):
    try:
        # Asistanın nasıl davranması gerektiğini söyleyen talimat
        prompt = (
            f"Sen bir ev asistanısın. Kullanıcıya çok kısa, öz ve günlük dilde cevap ver. "
            f"Cevabını maksimum 2 cümle ile sınırla. "
            f"Kullanıcı dedi ki: {text}"
        )
        
        # Asenkron çağrı
        response = await model.generate_content_async(prompt)
        
        # Markdown işaretlerini temizle (TTS okurken sıkıntı çıkarabilir)
        clean_text = response.text.replace("*", "").replace("#", "").replace("-", "")
        return clean_text
        
    except Exception as e:
        print(f"⚠️ AI Hatası: {e}")
        return "Üzgünüm, şu an bağlantı kuramadım."

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
    tts = gTTS(text=text, lang="tr", slow=False)
    tts.write_to_fp(mp3)
    mp3.seek(0)
    
    audio = AudioSegment.from_file(mp3, format="mp3")
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
            text = recognizer.recognize_google(audio, language="tr-TR")
            print(f"\n🗣️ SİZ: {text}")
        except sr.UnknownValueError:
            print("❌ Ses anlaşılamadı")
            return

        # AI Yanıtını Bekle
        answer = await generate_ai_response(text)
        print(f"🤖 ASİSTAN: {answer}")

        # TTS Oluştur
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
            print(f"Ses:{energy}", end="\r")

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
                    elif (time.time() - silence_start > SILENCE_DURATION):
                        print("\n⏱️ İşleniyor...")
                        is_processing = True
                        asyncio.create_task(
                            process_audio(websocket, bytes(audio_buffer))
                        )
                        audio_buffer.clear()
                        is_speaking = False

    except WebSocketDisconnect:
        print("🔌 ESP32 ayrıldı")
    except Exception as e:
        print(f"⚠️ WebSocket hata {e}")

# =========================
# START
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
