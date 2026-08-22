import whisper
import os
import csv

# ==========================================
# SETTINGS
# ==========================================

AUDIO_FOLDER = "audio"
CSV_FILE = "transcriptions.csv"

# ==========================================
# LOAD WHISPER MODEL
# ==========================================

print("🔄 Loading Whisper SMALL model...")
print("⏳ Please wait...\n")

model = whisper.load_model("small")

print("✅ Model loaded!\n")

# ==========================================
# FIND AUDIO FILES
# ==========================================

if not os.path.exists(AUDIO_FOLDER):
    print(f"❌ Audio folder not found: {AUDIO_FOLDER}")
    exit()

files = sorted([
    f for f in os.listdir(AUDIO_FOLDER)
    if f.lower().endswith(".wav")
])

if len(files) == 0:
    print("❌ No WAV files found in the audio folder.")
    exit()

print(f"🎧 Found {len(files)} audio files.\n")

# ==========================================
# CREATE CSV
# ==========================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "transcription"
    ])

    # ======================================
    # TRANSCRIBE EACH FILE
    # ======================================

    for number, file in enumerate(files, start=1):

        filepath = os.path.join(
            AUDIO_FOLDER,
            file
        )

        print("=" * 60)
        print(f"🎙️ [{number}/{len(files)}] {file}")
        print("🔄 Transcribing...")

        try:

            result = model.transcribe(
                filepath,
                fp16=False,
                temperature=0
            )

            text = result["text"].strip()

            print(f"📝 {text}")

            # Save to CSV
            writer.writerow([
                file,
                text
            ])

        except Exception as e:

            print(f"❌ Error: {e}")

            writer.writerow([
                file,
                f"ERROR: {e}"
            ])

# ==========================================
# COMPLETE
# ==========================================

print("\n")
print("=" * 60)
print("🎉 TRANSCRIPTION COMPLETE!")
print("=" * 60)
print(f"📄 CSV created: {CSV_FILE}")
print("=" * 60)