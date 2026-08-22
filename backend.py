import asyncio
import websockets
import json
import os
import subprocess
import whisper

# ============================================================
# SERVER CONFIG
# ============================================================

HOST = "0.0.0.0"

# Cloud platforms provide PORT automatically
PORT = int(os.environ.get("PORT", 8765))

# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_FOLDER = os.path.join(BASE_DIR, "audio")
TRANSCRIPT_FOLDER = os.path.join(BASE_DIR, "transcripts")

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

# ============================================================
# WHISPER
# ============================================================

print("🤖 Loading Whisper model...")

model = whisper.load_model("tiny")

print("✅ Whisper model loaded!")

# ============================================================
# FILE NUMBER
# ============================================================

def get_next_number():

    files = [
        f
        for f in os.listdir(AUDIO_FOLDER)
        if f.startswith("browser_")
        and f.endswith(".webm")
    ]

    return len(files) + 1


# ============================================================
# FFMPEG
# ============================================================

def convert_to_wav(webm_file, wav_file):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        webm_file,
        "-ar",
        "16000",
        "-ac",
        "1",
        wav_file
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("❌ FFmpeg error:")
        print(result.stderr)

        return False

    return True


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(wav_file):

    print("\n🎙️ Transcribing audio...")

    result = model.transcribe(
        wav_file,
        fp16=False
    )

    text = result["text"].strip()

    return text


# ============================================================
# WEBSOCKET CLIENT
# ============================================================

async def handle_client(websocket):

    print("\n✅ Frontend connected!")

    audio_chunks = []
    chunk_count = 0

    try:

        async for message in websocket:

            # ==================================================
            # TEXT MESSAGE
            # ==================================================

            if isinstance(message, str):

                try:

                    data = json.loads(message)

                except json.JSONDecodeError:

                    print("⚠️ Invalid JSON")

                    continue

                # ==============================================
                # AUDIO METADATA
                # ==============================================

                if data.get("type") == "audio":

                    chunk_count += 1

                    print(
                        f"📦 Chunk {chunk_count} metadata received"
                    )

                    await websocket.send(
                        json.dumps({
                            "type": "ready_for_audio",
                            "chunk": chunk_count
                        })
                    )

                # ==============================================
                # STOP
                # ==============================================

                elif data.get("type") == "stop":

                    print("\n🛑 Stop signal received")

                    if not audio_chunks:

                        await websocket.send(
                            json.dumps({
                                "type": "error",
                                "message": "No audio received."
                            })
                        )

                        continue

                    # ==========================================
                    # FILE NUMBER
                    # ==========================================

                    number = get_next_number()

                    # ==========================================
                    # WEBM
                    # ==========================================

                    webm_file = os.path.join(
                        AUDIO_FOLDER,
                        f"browser_{number:03d}.webm"
                    )

                    with open(
                        webm_file,
                        "wb"
                    ) as f:

                        for chunk in audio_chunks:

                            f.write(chunk)

                    print(
                        f"💾 Audio saved: {webm_file}"
                    )

                    # ==========================================
                    # WAV
                    # ==========================================

                    wav_file = os.path.join(
                        AUDIO_FOLDER,
                        f"browser_{number:03d}.wav"
                    )

                    print(
                        "🔄 Converting WebM → WAV..."
                    )

                    converted = convert_to_wav(
                        webm_file,
                        wav_file
                    )

                    if not converted:

                        await websocket.send(
                            json.dumps({
                                "type": "error",
                                "message":
                                "FFmpeg conversion failed."
                            })
                        )

                        continue

                    print(
                        f"✅ WAV created: {wav_file}"
                    )

                    # ==========================================
                    # WHISPER
                    # ==========================================

                    text = transcribe_audio(
                        wav_file
                    )

                    print("\n" + "=" * 50)
                    print("📝 TRANSCRIPTION")
                    print("=" * 50)
                    print(text)
                    print("=" * 50)

                    # ==========================================
                    # SAVE TEXT
                    # ==========================================

                    txt_file = os.path.join(
                        TRANSCRIPT_FOLDER,
                        f"browser_{number:03d}.txt"
                    )

                    with open(
                        txt_file,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        f.write(text)

                    # ==========================================
                    # SEND TO FRONTEND
                    # ==========================================

                    await websocket.send(
                        json.dumps({
                            "type": "transcription",
                            "text": text
                        })
                    )

                    print(
                        "📤 Transcription sent to frontend"
                    )

                    # ==========================================
                    # RESET
                    # ==========================================

                    audio_chunks = []
                    chunk_count = 0

            # ==================================================
            # AUDIO BINARY DATA
            # ==================================================

            elif isinstance(message, bytes):

                audio_chunks.append(message)

                print(
                    f"🎵 Audio received | "
                    f"{len(message)} bytes"
                )

                await websocket.send(
                    json.dumps({
                        "type": "processed",
                        "chunk": chunk_count
                    })
                )

    except websockets.exceptions.ConnectionClosed:

        print("🔴 Frontend disconnected")

    except Exception as e:

        print(
            f"❌ Server error: {e}"
        )


# ============================================================
# SERVER
# ============================================================

async def main():

    print("\n🚀 Starting Voice AI Backend...")

    print(
        f"📡 WebSocket: ws://{HOST}:{PORT}"
    )

    print(
        "🌐 Waiting for frontend..."
    )

    print()

    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        max_size=None
    ):

        await asyncio.Future()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print(
            "\n🛑 Backend stopped."
        )
