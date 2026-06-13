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

import static_ffmpeg
static_ffmpeg.add_paths()    

# ================= GÜNCELLENEN KÜRESEL AYARLAR =================
SAMPLE_RATE = 16000  # AI modellerinin ve ESP32'nin ortak frekansı (16kHz)
CHANNELS = 1         # Mono (Tek Kanal)

SILENCE_THRESHOLD = 1200   
SILENCE_DURATION = 1.5     # Konuşma sonrası beklenecek sessizlik (saniye)

is_processing = False
# ===============================================================

def generate_ai_response(user_text):
    user_text = user_text.lower()
    if "merhaba" in user_text or "selam" in user_text:
        return "Merhaba! Sana nasıl yardımcı olabilirim?"
    elif "nasılsın" in user_text:
        return "Harikayım, teşekkür ederim! Sen nasılsın?"
    elif "kimsin" in user_text:
        return "Ben senin E S P 32 tabanlı akıllı asistanınım."
    elif "Işığı aç" in user_text:
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

async def handle_response_task(websocket, audio_to_process):
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
            chunk_size = 4096  
            start_transmission_time = time.time()
            
            for i in range(0, len(raw_pcm_bytes), chunk_size):
                chunk = raw_pcm_bytes[i:i+chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(0.100) 
                
            print("✅ Ses gönderme tamamlandı.")
            
            # Donanımsal Koruma Zaman Kilidi
            gercek_ses_suresi = len(raw_pcm_bytes) / 32000.0
            gecen_sure = time.time() - start_transmission_time
            kalan_bekleme = (gercek_ses_suresi + 1.2) - gecen_sure
            
            if kalan_bekleme > 0:
                print(f"⏳ Asistanın konuşması bitiyor, donanım senkronizasyonu için {kalan_bekleme:.2f} saniye bekleniyor...")
                await asyncio.sleep(kalan_bekleme)
                
        else:
            print("❌ Boş ses veya gürültü geçişi, iptal edildi.")
            
    except Exception as e:
        print(f"\n⚠️ İşlem sırasında hata oluştu: {e}")
    finally:
        is_processing = False
        print("\n🎤 Yeniden Dinleniyor...\n")

async def audio_stream_handler(websocket):
    global is_processing
    print(f"\n⚡ ESP32 Asistan Bağlandı! ({websocket.remote_address})")
    
    audio_buffer = bytearray()
    is_speaking = False
    silence_start_time = None
    is_processing = False
    
    print("🎤 Sunucu dinlemede... Konuşmaya başlayabilirsiniz.")

    try:
        async for audio_data in websocket:
            if is_processing:
                continue
                
            data_chunk = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(data_chunk) > 0:
                energy = int(np.std(data_chunk))
                
                print(f"📊 Canlı Ses: {energy:<10} | Eşik: {SILENCE_THRESHOLD} | Durum: {'🗣️ KAYITTA' if is_speaking else '💤 SESSİZ'}", end="\r")
                
                if energy > SILENCE_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        print("\n🎙️  Ses algılandı! Eski arka plan gürültüleri temizlendi, yeni kayıt başladı...")
                        audio_buffer = bytearray()  # 🎯 DÜZELTİLDİ: Konuşma başladığı an depo tamamen sıfırlanır!
                    
                    audio_buffer.extend(audio_data)  # Sadece konuşma anındaki veriler eklenir
                    silence_start_time = None
                else:
                    if is_speaking:
                        # 🎯 DÜZELTİLDİ: Konuşma devam ederken aradaki kısa duraksamaları (kelime arası boşlukları) kaydeder
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

    except websockets.exceptions.ConnectionClosed:
        print("\n🔌 ESP32 Bağlantıyı kapattı.")
    finally:
        is_processing = False
        # --- Bu fonksiyonu main() fonksiyonunun üstüne ekleyin ---
async def health_check(path, request_headers):
    # Render, uygulamanın çalışıp çalışmadığını anlamak için HEAD isteği gönderir
    if path == "/":
        return None  # WebSocket isteği ise devam et
    return None # Diğer durumlarda standart davran

# --- main fonksiyonunu bu şekilde güncelleyin ---
async def main():
    PORT = int(os.environ.get("PORT", 10000))
    
    # process_request parametresi sağlık kontrolü isteklerini yönetir
    async with websockets.serve(
        audio_stream_handler, 
        "0.0.0.0", 
        PORT,
        process_request=health_check, 
        ping_interval=None,
        ping_timeout=None
    ):
        print(f"🚀 Sunucu {PORT} portunda başarıyla başlatıldı!")
        await asyncio.Future()

async def main():
    PORT = int(os.environ.get("PORT", 10000))
    
    # Render'da WebSocket bazen direkt HTTP isteği gibi algılanabilir.
    # Bu yüzden host'u "0.0.0.0" olarak bırakmak en güvenlisidir.
    async with websockets.serve(
        audio_stream_handler, 
        "0.0.0.0", 
        PORT,
        ping_interval=None, # Render'da bağlantı kopmasını engellemek için
        ping_timeout=None
    ):
        print(f"🚀 Sunucu {PORT} portunda başarıyla başlatıldı!")
        await asyncio.Future() 

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
