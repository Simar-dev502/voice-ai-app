from flask import Flask, render_template, request
import whisper
import os
import wave
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Small model for Render memory
model = whisper.load_model("tiny")


POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome",
    "happy", "love", "best", "nice", "wonderful",
    "positive", "thank", "thanks", "perfect"
}

NEGATIVE_WORDS = {
    "bad", "poor", "worst", "hate", "sad",
    "angry", "terrible", "awful", "negative",
    "disappointed", "problem", "issue", "wrong"
}


def analyze_sentiment(text):
    words = text.lower().split()

    positive_score = sum(word in POSITIVE_WORDS for word in words)
    negative_score = sum(word in NEGATIVE_WORDS for word in words)

    if positive_score > negative_score:
        return "POSITIVE"
    elif negative_score > positive_score:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def read_wav_file(filepath):
    with wave.open(filepath, "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio /= 32768.0

    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
        audio /= 2147483648.0

    else:
        raise ValueError("Unsupported WAV format")

    # Convert stereo to mono
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "audio" not in request.files:
        return "No audio file uploaded", 400

    file = request.files["audio"]

    if file.filename == "":
        return "No audio file selected", 400

    filename = secure_filename(file.filename)

    if not filename.lower().endswith(".wav"):
        return "Please upload a WAV audio file.", 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        audio = read_wav_file(filepath)

        result = model.transcribe(
            audio,
            fp16=False
        )

        transcription = result["text"].strip()

        sentiment = analyze_sentiment(transcription)

        return render_template(
            "index.html",
            transcription=transcription,
            sentiment=sentiment
        )

    except Exception as e:
        print("TRANSCRIPTION ERROR:", e)
        return "Audio processing failed. Please try again.", 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True)