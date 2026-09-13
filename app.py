from flask import Flask, render_template, request, jsonify
import os
import subprocess
import shutil

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    import whisper
except Exception:
    whisper = None

from werkzeug.utils import secure_filename
import imageio_ffmpeg


FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_DIR = os.path.dirname(FFMPEG_EXE)
FFMPEG_ALIAS = os.path.join(FFMPEG_DIR, "ffmpeg.exe")

if not os.path.exists(FFMPEG_ALIAS):
    shutil.copy2(FFMPEG_EXE, FFMPEG_ALIAS)

PATH = os.environ.get("PATH", "")
if FFMPEG_DIR not in PATH.split(os.pathsep):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + PATH

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("tiny") if whisper is not None else None


# =========================
# SENTIMENT WORDS
# =========================

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "happy", "love", "best", "nice", "wonderful",
    "positive", "thank", "thanks", "perfect",
    "helpful", "satisfied", "fantastic", "beautiful"
}

NEGATIVE_WORDS = {
    "bad", "poor", "worst", "hate", "sad",
    "angry", "terrible", "awful", "negative",
    "disappointed", "problem", "issue", "wrong",
    "horrible", "unhappy", "frustrated", "annoying"
}


# =========================
# SENTIMENT ANALYSIS
# =========================


def analyze_sentiment(text):
    words = text.lower().split()

    positive_score = sum(
        word.strip(".,!?;:'\"()[]{}") in POSITIVE_WORDS
        for word in words
    )

    negative_score = sum(
        word.strip(".,!?;:'\"()[]{}") in NEGATIVE_WORDS
        for word in words
    )

    if positive_score > negative_score:
        return "POSITIVE"

    elif negative_score > positive_score:
        return "NEGATIVE"

    return "NEUTRAL"


def translate_text(text, target_language="en"):
    if not text or GoogleTranslator is None:
        return None

    target_language = (target_language or "en").strip()

    if target_language.lower() in {"en", "english"}:
        return None

    try:
        translated = GoogleTranslator(
            source="auto",
            target=target_language
        ).translate(text)

        return translated.strip() if translated else None

    except Exception as e:
        print("TRANSLATION ERROR:", repr(e))
        return None


# =========================
# AUDIO HELPERS
# =========================


def convert_audio_to_wav(input_file, output_file):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", input_file,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_file
    ]

    subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# TEXT ANALYSIS
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True) or {}

    text = str(data.get("text") or "").strip()
    target_language = str(data.get("target_language") or "en").strip()

    if not text:
        return jsonify({
            "error": "No transcription text"
        }), 400

    sentiment = analyze_sentiment(text)
    translation = translate_text(text, target_language)

    return jsonify({
        "transcription": text,
        "sentiment": sentiment,
        "translation": translation
    })


@app.route("/translate", methods=["POST"])
def translate():

    data = request.get_json(silent=True) or {}

    text = str(data.get("text") or "").strip()
    target_language = str(data.get("target_language") or "en").strip()

    if not text:
        return jsonify({
            "error": "No text to translate"
        }), 400

    translated = translate_text(text, target_language)

    return jsonify({
        "transcription": text,
        "translation": translated,
        "target_language": target_language
    })


# =========================
# AUDIO UPLOAD
# =========================

@app.route("/upload", methods=["POST"])
def upload():

    if "audio" not in request.files:
        return "No audio file uploaded", 400

    file = request.files["audio"]

    if file.filename == "":
        return "No audio file selected", 400

    target_language = request.form.get("target_language", "en")
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    wav_path = os.path.join(UPLOAD_FOLDER, "converted_audio.wav")

    file.save(input_path)

    try:
        convert_audio_to_wav(input_path, wav_path)

        if model is None:
            return "Whisper model is unavailable", 500

        result = model.transcribe(wav_path, fp16=False)
        transcription = result["text"].strip()
        sentiment = analyze_sentiment(transcription)
        translation = translate_text(transcription, target_language)

        return render_template(
            "index.html",
            transcription=transcription,
            sentiment=sentiment,
            translation=translation,
            target_language=target_language
        )

    except Exception as e:
        print("AUDIO ERROR:", repr(e))
        return "Audio processing failed: " + str(e), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(wav_path):
            os.remove(wav_path)


# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )