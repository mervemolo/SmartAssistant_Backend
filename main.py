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

SILENCE_THRESHOLD = 200   
SILENCE_DURATION = 1.5

is_processing = False
# ===================================================

app = FastAPI()

# 🎯 RENDER HEALTH CHECK İÇİN HTTP ROTALARI
@app.get("/")
@app.head("/")
@app.get("/health")
@app.head("/health")
async def health_check():
    return {"status": "Service is live and waiting for WebSocket connection!"}

# --- Ses İşleme Fonksiyonları ---
def generate_ai_response(user_text):
    user_text = user_text.lower()
    if "merhaba" in user_text or "selam" in user_text:
        return "Merhaba! Sana nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_text:
        return "Harikayım, teşekkür ederim! Sen nasılsın?"
    elif "kimsin" in user_text:
        return "Ben senin E S P 32 tabanlı akıllı asistanınım."
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
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16)
    if n_channels == 2:
        data = (data[0::2] // 2 + data[1::2] // 2)
    if framerate != 16000:
        old_indices = np.arange(len(data))
        new_indices = np.linspace(0, len(data) - 1, int(len(data) * 16000 / framerate))
        data = np.interp(new_indices, old_indices, data).astype(np.int16)
    return data.tobytes()

def process_and_save_audio(raw_bytes):
    if len(raw_bytes) == 0:
        return None
    audio_data16 = np.frombuffer(raw_bytes, dtype=np.int16)
    filename = "canli_kayit.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data16.tobytes())
    return filename

def convert_speech_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="tr-TR")
            print(f"\n🗣️  SİZ: {text}")
            return text
        except sr.UnknownValueError:
            print("\n👁️‍🗨️  Ses anlaşılamadı veya gürültü çok yüksek.")
        except sr.RequestError as e:
            print(f"\n⚠️  Google STT Servis hatası: {e}")
    return None

async def handle_response_task(websocket: WebSocket, audio_to_process: bytes):
    global is_processing
    try:
        wav_file = process_and_save_audio(audio_to_process)
        user_text = convert_speech_to_text(wav_file)
        
        if user_text:
            ai_reply = generate_ai_response(user_text)
            print(f"🤖 ASİSTAN: {ai_reply}")
            
            reply_wav = generate_tts_audio(ai_reply)
            raw_pcm_bytes = prepare_audio_for_esp32(reply_wav)
            
            print("📤 Cevap sesi aktarılıyor...")
            chunk_size = 8192  
            start_transmission_time = time.time()
            
            for i in range(0, len(raw_pcm_bytes), chunk_size):
                chunk = raw_pcm_bytes[i:i+chunk_size]
                await websocket.send_bytes(chunk)
                await asyncio.sleep(0.100) 
                
            print("✅ Ses gönderme tamamlandı.")
            
            gercek_ses_suresi = len(raw_pcm_bytes) / 32000.0
            gecen_sure = time.time() - start_transmission_time
            kalan_bekleme = (gercek_ses_suresi + 1.2) - gecen_sure
            
            if kalan_bekleme > 0:
                print(f"⏳ Asistanın konuşması bitiyor, {kalan_bekleme:.2f} sn bekleniyor...")
                await asyncio.sleep(kalan_bekleme)
                
        else:
            print("❌ Boş ses veya gürültü geçişi, iptal edildi.")
            
    except Exception as e:
        print(f"\n⚠️ İşlem sırasında hata oluştu: {e}")
    finally:
        is_processing = False
        print("\n🎤 Yeniden Dinleniyor...\n")

@app.websocket("/")
async def websocket_root_handler(websocket: WebSocket):
    await audio_stream_handler(websocket)

async def audio_stream_handler(websocket: WebSocket):
    global is_processing
    await websocket.accept()
    print("\n⚡ ESP32 Asistan Bağlandı!")
    
    audio_buffer = bytearray()
    is_speaking = False
    silence_start_time = None
    is_processing = False
    last_print_time = time.time() # Hata Düzeltme: Zamanlayıcı yerel olarak tanımlandı
    
    print("🎤 Sunucu dinlemede... Konuşmaya başlayabilirsiniz.")

    try:
        while True:
            audio_data = await websocket.receive_bytes()
            
            if is_processing:
                continue
                
            data_chunk = np.frombuffer(audio_data, dtype=np.int16)
            energy = 0 # Hata Düzeltme: Çökmeyi önlemek için varsayılan değer atandı
            
            if len(data_chunk) > 0:
                energy = int(np.std(data_chunk))
    
            # Hata Düzeltme: Girintiler ve değişken adı senkronize edildi
            if time.time() - last_print_time > 0.5:
                print(f"📊 Ses: {energy:<5} | Durum: {'🗣️ KAYITTA' if is_speaking else '💤 SESSİZ'}    ", end="\r")
                last_print_time = time.time()
                
            if energy > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    print("\n🎙️  Ses algılandı! Yeni kayıt başladı...")
                    audio_buffer = bytearray()
                
                audio_buffer.extend(audio_data)
                silence_start_time = None
            else:
                if is_speaking:
                    audio_buffer.extend(audio_data)
                    
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_DURATION:
                        print("\n⏱️  Sessizlik algılandı, yanıt hazırlanıyor...")
                        is_processing = True
                        asyncio.create_task(handle_response_task(websocket, bytes(audio_buffer)))
                        
                        audio_buffer = bytearray()
                        is_speaking = False
                        silence_start_time = None

    except WebSocketDisconnect:
        print("\n🔌 ESP32 Bağlantıyı kapattı.")
    except Exception as e:
        print(f"\n⚠️ Bilinmeyen bağlantı hatası: {e}")
    finally:
        is_processing = False

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 3000))
    print(f"🚀 Uvicorn Sunucusu Aktif... Port: {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
