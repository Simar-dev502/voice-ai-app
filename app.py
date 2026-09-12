from flask import Flask, render_template, request
import whisper
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load lightweight Whisper model
model = whisper.load_model("tiny")


# Lightweight sentiment analysis
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["audio"]

    if not file:
        return "No audio file uploaded", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Transcription
    result = model.transcribe(filepath)
    transcribe = result["text"]

    # Sentiment
    sentiment = analyze_sentiment(transcribe)

    return render_template(
        "index.html",
        transcription=transcribe,
        sentiment=sentiment
    )


if __name__ == "__main__":
    app.run(debug=True)