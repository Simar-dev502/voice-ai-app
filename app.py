from flask import Flask, render_template, request
import whisper
import os
from transformers import pipeline

app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load models
model = whisper.load_model("base")
sentiment_model = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["audio"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # transcription
    result = model.transcribe(filepath)
    transcribe = result["text"]

    # sentiment analysis
    sentiment = sentiment_model(transcribe)

    return render_template(
        "index.html",
        transcription=transcribe,
        sentiment=sentiment[0]["label"]
    )

if __name__ == "__main__":
    app.run(debug=True)