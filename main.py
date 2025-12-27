from flask import Flask , Response, jsonify, request, render_template

import sys
import os
sys.path.append(os.path.abspath("."))
from src.utils import PredictExamScore

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predictExamScore():
    obj = PredictExamScore()
    data = request.form
    print(data)
    predictedScore = obj.predict_score(data)
    print(predictedScore)
    return f"{predictedScore}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8000)