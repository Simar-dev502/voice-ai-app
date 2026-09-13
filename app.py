from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import subprocess
import tempfile
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

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise Exception("FFmpeg error: " + result.stderr[-1500:])


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    return result.text.strip()


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

    positive_score = sum(
        word.strip(".,!?") in POSITIVE_WORDS
        for word in words
    )

    negative_score = sum(
        word.strip(".,!?") in NEGATIVE_WORDS
        for word in words
    )

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

    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not configured on Render", 500

    filename = secure_filename(file.filename)

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    wav_path = os.path.join(
        UPLOAD_FOLDER,
        "converted_audio.wav"
    )

    try:
        file.save(input_path)

        print("AUDIO RECEIVED:", filename)

        convert_audio_to_wav(
            input_path,
            wav_path
        )

        print("AUDIO CONVERTED")

        transcription = transcribe_audio(
            wav_path
        )

        print("TRANSCRIPTION:", transcription)

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
            + str(e)
        ), 500

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
