from flask import Flask, render_template, request
import whisper
import os
import subprocess
import tempfile
import shutil
import numpy as np
from werkzeug.utils import secure_filename
import imageio_ffmpeg
import imageio_ffmpeg
import subprocess

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

print("FFMPEG PATH:", FFMPEG_EXE)

try:
    result = subprocess.run(
        [FFMPEG_EXE, "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("FFMPEG TEST:", result.stdout.splitlines()[0])
except Exception as e:
    print("FFMPEG ERROR:", repr(e))



app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
def convert_audio_to_wav(input_file, output_file):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", input_file,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_file
    ]

    subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )




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

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(input_path)

    wav_path = os.path.join(
        UPLOAD_FOLDER,
        "converted_audio.wav"
    )

    try:

        # Convert M4A / MP3 / WEBM / WAV etc. to WAV
        convert_audio_to_wav(
            input_path,
            wav_path
        )

        # Whisper transcription
        result = model.transcribe(
            wav_path,
            fp16=False
        )

        transcription = result["text"].strip()

        sentiment = analyze_sentiment(
            transcription
        )

        return render_template(
            "index.html",
            transcription=transcription,
            sentiment=sentiment
        )

    except Exception as e:

        print("AUDIO ERROR:", repr(e))

        return (
            "Audio processing failed: "
            + str(e),
            500
        )

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
