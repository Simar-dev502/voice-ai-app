from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "No text received"
        }), 400

    text = data["text"].strip()

    if not text:
        return jsonify({
            "error": "No transcription text"
        }), 400

    sentiment = analyze_sentiment(text)

    return jsonify({
        "transcription": text,
        "sentiment": sentiment
    })


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

    try:
        file.save(input_path)

        return (
            "Audio uploaded successfully. "
            "For free transcription, please use the microphone option."
        )

    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        return "Audio upload failed: " + str(e), 500

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )